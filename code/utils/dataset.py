import os
import torch
import imageio
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class ToTensor(object):
    def __call__(self, sample):
        for k in sample:
            sample[k] = torch.Tensor(np.transpose(sample[k], (2, 0, 1))) / 255.

        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if np.random.randint(2) == 1:
            for k in sample:
                sample[k] = np.fliplr(sample[k])

        if np.random.randint(2) == 1:
            for k in sample:
                sample[k] = np.flipud(sample[k])

        return sample


class RGBFlip(object):
    def __call__(self, sample):
        idx = np.random.permutation(3)
        for k in sample:
            sample[k] = sample[k][:, :, idx]

        return sample


def mixup(sample, prob=1.0, alpha=0.6):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return sample

    B = sample['lr0'].shape[0]
    rand_indices = torch.randperm(B)
    lam = torch.distributions.beta.Beta(alpha, alpha).rsample((B, 1, 1, 1)).to(sample['lr0'].device)

    for k in sample:
        sample[k] = lam * sample[k] + (1 - lam) * sample[k][rand_indices]

    return sample


class TrainSet(Dataset):
    def __init__(self, dataset_dir, upscale, transform=transforms.Compose([RandomFlip(), RGBFlip(), ToTensor()])):
        self.path = dataset_dir + 'train/patches_x' + str(upscale)
        self.file_list = os.listdir(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.path + '/' + self.file_list[idx]
        lr_left = imageio.imread(file_path + '/lr0.png')
        lr_right = imageio.imread(file_path + '/lr1.png')
        hr_left = imageio.imread(file_path + '/hr0.png')
        hr_right = imageio.imread(file_path + '/hr1.png')

        sample = {'lr0': lr_left,
                  'lr1': lr_right,
                  'hr0': hr_left,
                  'hr1': hr_right}

        if self.transform:
            sample = self.transform(sample)

        return sample


class TestSet(Dataset):
    def __init__(self, dataset_dir, testset, upscale, transform=transforms.Compose([ToTensor()])):
        self.upscale = upscale
        self.path = dataset_dir + f'test/' + testset
        self.file_list = sorted(os.listdir(self.path + '/lr_x' + str(upscale)))

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        hr_left = imageio.imread(self.path + '/hr/' + self.file_list[idx] + '/hr0.png')
        hr_right = imageio.imread(self.path + '/hr/' + self.file_list[idx] + '/hr1.png')

        lr_left = imageio.imread(self.path + '/lr_x' + str(self.upscale) + '/' + self.file_list[idx] + '/lr0.png')
        lr_right = imageio.imread(self.path + '/lr_x' + str(self.upscale) + '/' + self.file_list[idx] + '/lr1.png')

        sample = {'lr0': lr_left,
                  'lr1': lr_right,
                  'hr0': hr_left,
                  'hr1': hr_right}

        if self.transform:
            sample = self.transform(sample)

        return sample