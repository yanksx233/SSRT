import os
import torch
import argparse
import numpy as np
from utils.log import get_logger
from utils.dataset import TestSet
from model.build import build_model
from utils.util import cal_psnr_and_ssim
from torchvision import transforms
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser('test processing')

    # TODO:
    parser.add_argument('--drop_path_rate', type=float, default=0)
    parser.add_argument('--arch', type=str, default='ssrt')
    parser.add_argument('--depths', type=str, default='2*12')
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
    parser.add_argument('--checkpoint', type=str, default=None)

    # device
    parser.add_argument('--device', type=str, default='cuda:0')

    # mode
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--sisr', action='store_true')

    args = parser.parse_args()
    return args


def update_state_dict(model, pretrained_dict):
    """force load parameter"""
    own_state_dict = model.state_dict()
    # filter parameters
    pretrained_dict = {name: param for name, param in pretrained_dict.items() if name in own_state_dict}
    own_state_dict.update(pretrained_dict)
    model.load_state_dict(own_state_dict)


def test(args, logger, model):
    testset = TestSet(args.dataset_dir, args.testset, args.upscale)
    test_loader = DataLoader(testset, batch_size=1, num_workers=4, shuffle=False)

    result_path = os.path.join(args.log_dir, args.save_dir, args.mode, 'result', args.testset)
    os.mkdir(result_path)

    model.eval()
    with torch.no_grad():
        logger.debug(f'Current dataset: {args.testset}')
        psnr_crop, ssim_crop = [], []
        psnr_all, ssim_all = [], []
        for idx, sample in enumerate(test_loader):
            lr_left, lr_right, hr_left, hr_right = sample['lr0'].to(args.device), sample['lr1'].to(args.device), \
                                                   sample['hr0'].to(args.device), sample['hr1'].to(args.device)

            if args.sisr:
                sr_left = model(lr_left)
                sr_right = model(lr_right)
            else:
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
            
            sr_left = transforms.ToPILImage()(torch.squeeze(sr_left.data.cpu(), 0))
            sr_left.save(result_path + '/' + str(idx) + '_L.png')
            sr_right = transforms.ToPILImage()(torch.squeeze(sr_right.data.cpu(), 0))
            sr_right.save(result_path + '/' + str(idx) + '_R.png')

        psnr_crop = float(np.array(psnr_crop).mean())
        ssim_crop = float(np.array(ssim_crop).mean())
        psnr_all = float(np.array(psnr_all).mean())
        ssim_all = float(np.array(ssim_all).mean())

        logger.debug(f'The average PSNR/SSIM: left -- {psnr_crop:.2f}/{ssim_crop:.4f},  left+right -- {psnr_all:.2f}/{ssim_all:.4f}\n')


if __name__ == '__main__':
    args = get_args()
    logger = get_logger(args.log_dir, args.save_dir, args.mode, args)
    # dataset = ['Flickr1024', 'Flickr1024_val', 'KITTI2012', 'KITTI2015', 'Middlebury', 'ETH3D']
    dataset = ['Middlebury', 'ETH3D']

    model = build_model(args).to(args.device)
    torch.backends.cudnn.benchmark = True
    
    logger.info(f'Load model: {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    # update_state_dict(model, checkpoint['model_state_dict'])  # force load
    model.load_state_dict(checkpoint['model_state_dict'])

    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    logger.info("Number of parameter: %.2fM" % (total / 1e6))
    
    for i in range(len(dataset)):
        args.testset = dataset[i]
        model.num_cats = args.num_cats if args.testset == 'ETH3D' else 0
        test(args, logger, model)