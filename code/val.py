import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.log import get_logger
from utils.dataset import TestSet
from model.build import build_model
from utils.util import cal_psnr_and_ssim
from torch.utils.data import DataLoader
from thop import profile


def get_args():
    parser = argparse.ArgumentParser('val processing')

    # TODO:
    parser.add_argument('--drop_path_rate', type=float, default=0)
    parser.add_argument('--arch', type=str, default='steremer')
    parser.add_argument('--depths', type=str, default='2*11')
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_feats', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--num_cats', type=int, default=0)
    parser.add_argument('--upscale', type=int, default=4)
    parser.add_argument('--use_checkpoint', action='store_true')
    # --------------------

    # dir setting
    parser.add_argument('--dataset_dir', type=str, default='../Dataset/Stereo_48x96/')
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--save_dir', type=str, default='demo')

    # device
    parser.add_argument('--device', type=str, default='cuda:0')

    # test setting
    parser.add_argument('--start_iter', type=int, default=150000)
    parser.add_argument('--end_iter', type=int, default=200000)
    parser.add_argument('--step', type=int, default=10000)

    # mode
    parser.add_argument('--mode', type=str, default='val')

    args = parser.parse_args()
    return args


def update_state_dict(model, pretrained_dict):
    own_state_dict = model.state_dict()
    # filter parameters
    pretrained_dict = {name: param for name, param in pretrained_dict.items() if name in own_state_dict and 'mask' not in name}
    own_state_dict.update(pretrained_dict)
    model.load_state_dict(own_state_dict)


def test(model, logger, test_loader, device):
    model.eval()
    with torch.no_grad():
        psnr_crop, ssim_crop = [], []
        psnr_all, ssim_all = [], []
        for idx, sample in enumerate(test_loader):
            lr_left, lr_right, hr_left, hr_right = sample['lr0'].to(device), sample['lr1'].to(device), \
                                                   sample['hr0'].to(device), sample['hr1'].to(device)

            sr_left, sr_right = model(lr_left, lr_right)
            sr_left = torch.clamp(sr_left, 0., 1.)
            sr_right = torch.clamp(sr_right, 0., 1.)

            psnr_c, ssim_c = cal_psnr_and_ssim(sr_left[:, :, :, 64:], hr_left[:, :, :, 64:])
            psnr_l, ssim_l = cal_psnr_and_ssim(sr_left, hr_left)
            psnr_r, ssim_r = cal_psnr_and_ssim(sr_right, hr_right)

            psnr_crop.append(psnr_c)
            ssim_crop.append(ssim_c)
            psnr_all.append((psnr_l + psnr_r) / 2.)
            ssim_all.append((ssim_l + ssim_r) / 2.)

        psnr_crop = float(np.array(psnr_crop).mean())
        ssim_crop = float(np.array(ssim_crop).mean())
        psnr_all = float(np.array(psnr_all).mean())
        ssim_all = float(np.array(ssim_all).mean())

        logger.debug(f'The average psnr: left -- {psnr_crop:.5f},  left+right -- {psnr_all:.5f}')
        logger.debug(f'The average ssim: left -- {ssim_crop:.5f},  left+right -- {ssim_all:.5f}')

        return psnr_all, ssim_all


if __name__ == '__main__':
    args = get_args()
    logger = get_logger(args.log_dir, args.save_dir, args.mode, args)
    
    datasets = ['Flickr1024', 'Flickr1024_val', 'KITTI2012', 'KITTI2015', 'Middlebury', 'ETH3D']
    # datasets = ['Flickr1024_val', 'ETH3D']
    iters = np.arange(args.start_iter, args.end_iter+1, args.step)

    model = build_model(args).to(args.device)
    torch.backends.cudnn.benchmark = True

    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    logger.debug("Number of parameter: %.2fM" % (total / 1e6))
    logger.debug(f'Model type: {args.arch}')

    psnr = {dataset: np.zeros(len(iters)) for dataset in datasets}
    ssim = {dataset: np.zeros(len(iters)) for dataset in datasets}
    for idx, iteration in enumerate(iters):
        model_file = os.path.join(args.log_dir, args.save_dir, 'train/checkpoint', f'x{args.upscale}_iter_{int(iteration/1000)}k.tar')
        logger.info(f'Load model: {model_file}...')
        checkpoint = torch.load(model_file, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # update_state_dict(model, checkpoint['model_state_dict'])

        for dataset in datasets:
            logger.debug(f'Current dataset: {dataset}')
            args.testset = dataset
            testset = TestSet(args.dataset_dir, args.testset, args.upscale)
            test_loader = DataLoader(testset, batch_size=1, num_workers=4, shuffle=False)
            current_psnr, current_ssim = test(model, logger, test_loader, args.device)
            psnr[dataset][idx] = current_psnr
            ssim[dataset][idx] = current_ssim
        logger.debug('-----------------------------------------------------------\n')

    logger.debug('===========================================================')
    for dataset in datasets:
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax1.plot(iters, psnr[dataset], label='PSNR', c='r')
        ax1.scatter(iters, psnr[dataset], c='c')
        ax1.set_xticks(iters)
        # ax1.tick_params(labelsize=22)
        ax1.set_xlabel('Iteration', fontsize=22)
        ax1.set_ylabel('PSNR', fontsize=22)
        ax1.set_title(dataset, fontsize=24)

        ax2 = ax1.twinx()
        ax2.plot(iters, ssim[dataset], label='SSIM', c='b')
        ax2.scatter(iters, ssim[dataset], c='m')
        ax2.set_ylabel('SSIM', fontsize=22)
        # ax2.tick_params(labelsize=22)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc=4, fontsize=14)

        fig.savefig(os.path.join(args.log_dir, args.save_dir, args.mode, 'result', f'{dataset}_result.jpg'))

        logger.info(f'Dataset: {dataset}')
        logger.info(f'Best PSNR: {np.max(psnr[dataset]):.2f}, iter: {iters[np.argmax(psnr[dataset])]}')
        logger.info(f'Best SSIM: {np.max(ssim[dataset]):.4f}, iter: {iters[np.argmax(ssim[dataset])]}')
    logger.debug('===========================================================\n')

    flops, params = profile(model, inputs=(torch.rand((1, 3, 256, 256)).to(args.device), torch.rand((1, 3, 256, 256)).to(args.device)))
    logger.debug(f'FLOPs for 256x256x3 input: {flops/1e9} G')
