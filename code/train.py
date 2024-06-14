import os
import time
import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from utils.log import get_logger
from utils.dataset import TrainSet, mixup
from model.build import build_model
from utils.util import cal_psnr_and_ssim, build_optimizer, build_scheduler
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser('train processing')
    
    # model
    parser.add_argument('--sisr', action='store_true')
    parser.add_argument('--arch', type=str, default='swin')
    parser.add_argument('--depths', type=str, default='4*6', help='(depths[0],) * depths[1]')
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_feats', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=9)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--num_cats', type=int, default=0)
    parser.add_argument('--upscale', type=int, default=4)
    parser.add_argument('--use_checkpoint', action='store_true', help='saving GPU memory during training')
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    # log
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--log_every_iter', type=int, default=400)
    parser.add_argument('--save_dir', type=str, default='demo')   # log_dir/save_dir/train
    parser.add_argument('--save_every_iter', type=int, default=10000)
    parser.add_argument('--tb_log', action='store_true')

    # device
    parser.add_argument('--gpu_ids', type=str, default='5', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=2)

    # datasets
    parser.add_argument('--dataset_dir', type=str, default='../Dataset/Stereo_48x96/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mixup', type=float, default=0)

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='for AdamW')
    parser.add_argument('--beta2', type=float, default=0.999, help='for AdamW')

    # scheduler
    parser.add_argument('--scheduler_name', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--periods', type=str, default='5e4,1e5', help='for cosine')
    parser.add_argument('--min_lrs', type=str, default='1e-4,1e-6', help='for cosine')
    parser.add_argument('--decay_step', type=int, default=50000, help='for step')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='for step')
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_steps', type=int, default=5000)

    # train
    parser.add_argument('--current_iter', type=int, default=0)
    parser.add_argument('--current_epoch', type=int, default=1)
    parser.add_argument('--total_iter', type=int, default=150000)
    parser.add_argument('--load_from', type=str, help='load only parameters', default=None)
    parser.add_argument('--resume_from', type=str, help='continue to train from checkpoint', default=None)

    # mode
    parser.add_argument('--mode', type=str, default='train')

    args = parser.parse_args()

    # args.local_rank = os.environ['LOCAL_RANK']
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)

    args.periods = list(map(lambda x: int(float(x)), args.periods.split(',')))
    args.min_lrs = list(map(float, args.min_lrs.split(',')))

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        set_random_seed(123)
        self.device = torch.device(args.gpu_ids[args.local_rank]) if len(args.gpu_ids) != 0 else torch.device('cpu')
        if len(args.gpu_ids) != 0:
            torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = True

        if len(args.gpu_ids) > 1: 
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = -1
            self.world_size = 1

        trainset = TrainSet(args.dataset_dir, args.upscale)
        if len(args.gpu_ids) > 1:
            assert args.batch_size % self.world_size == 0
            train_sampler = DistributedSampler(trainset)
            self.train_loader = DataLoader(trainset, batch_size=args.batch_size//self.world_size, num_workers=args.num_workers, sampler=train_sampler)
        else:
            self.train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        self.model = build_model(args).to(self.device)
        self.model_without_ddp = self.model
        if self.rank <= 0:
            self.logger = get_logger(args.log_dir, args.save_dir, args.mode, saved_args=args)
            if args.tb_log:
                self.tb_logger = SummaryWriter(log_dir=os.path.join(args.log_dir, args.save_dir, args.mode, 'tb_log'))
            total = sum([param.nelement() for param in self.model.parameters() if param.requires_grad])
            self.logger.info("Number of parameter: %.2fM" % (total / 1e6))
            self.logger.info(f'Model type: {args.arch}')
            if args.resume_from or args.load_from:
                pretrain = args.resume_from if args.resume_from else args.load_from
                self.logger.warning(f'Load model parameters from: {pretrain}...')
                checkpoint = torch.load(pretrain, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.logger.warning('Train from scratch...')

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=(args.drop_path_rate > 0))
            self.model_without_ddp = self.model.module
        
        self.optimizer = build_optimizer(self.model, args)
        self.scheduler = build_scheduler(self.optimizer, args)
        if args.resume_from:
            checkpoint = torch.load(args.resume_from, map_location=self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.args.current_epoch = checkpoint['epoch'] + 1
            self.args.current_iter = checkpoint['iter']
            if self.rank <= 0:
                self.logger.warning(f'Load optimizer...')
                self.logger.warning(f'Resume epoch to {self.args.current_epoch}, iter to {self.args.current_iter}...')

        self.criterion = torch.nn.L1Loss().to(self.device)

    def train_one_epoch(self):
        self.model.train()
        loss_per_iter = []
        load_time_list = []
        t_load = time.time()
        for batch_idx, sample in enumerate(self.train_loader):
            t_load = time.time() - t_load
            load_time_list.append(t_load)
            self.args.current_iter += 1
            if self.args.current_iter > self.args.total_iter:
                break

            self.scheduler.step(self.args.current_iter-1)  # update learning rate

            sample = {k : sample[k].to(self.device) for k in sample}
            sample = mixup(sample, args.mixup)

            if self.args.sisr:
                sr_left = self.model(sample['lr0'])
                sr_right = self.model(sample['lr1'])
            else:
                sr_left, sr_right = self.model(sample['lr0'], sample['lr1'])

            loss = self.criterion(sr_left, sample['hr0']) + self.criterion(sr_right, sample['hr1'])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.rank <= 0:
                loss_per_iter.append(loss.detach().cpu().item())
                if self.args.tb_log:
                    self.tb_logger.add_scalar('Learning rate', self.optimizer.param_groups[0]["lr"], self.args.current_iter)
                    self.tb_logger.add_scalar('Loss/rec', loss_per_iter[-1], self.args.current_iter)

                if batch_idx % self.args.log_every_iter == 0:   # log message
                    self.logger.debug(f'epoch: {self.args.current_epoch},  '
                                      f'batch: {batch_idx+1}/{len(self.train_loader)},  '
                                      f'iter: {self.args.current_iter}/{self.args.total_iter},  '
                                      f'lr: {self.optimizer.param_groups[0]["lr"]:.4e},  '
                                      f'loss: {loss:.5f},  ' 
                                      f'average load time: {np.mean(load_time_list):.2f} s')
                    load_time_list = []

                if self.args.current_iter % self.args.save_every_iter == 0:  # save model
                    save_path = os.path.join(self.args.log_dir, self.args.save_dir, self.args.mode, 'checkpoint')
                    save_path = os.path.join(save_path, f'x{self.args.upscale}_iter_{int(self.args.current_iter/1000)}k.tar')
                    torch.save({'epoch': self.args.current_epoch,
                                'iter': self.args.current_iter,
                                'model_state_dict': self.model_without_ddp.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                }, save_path)

            t_load = time.time()

        if self.rank <= 0:
            self.logger.info(f'epoch: {self.args.current_epoch},  '
                             f'iter: {self.args.current_iter}/{self.args.total_iter},  '
                             f'lr: {self.optimizer.param_groups[0]["lr"]:.4e},  '
                             f'loss_avg: {np.mean(loss_per_iter):.5f}')
        self.args.current_epoch += 1

    def train(self):
        while self.args.current_iter <= self.args.total_iter:
            if self.world_size > 1:
                self.train_loader.sampler.set_epoch(self.args.current_epoch)
            self.train_one_epoch()


if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()