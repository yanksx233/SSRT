import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

class EDSR(nn.Module):
    def __init__(self, upscale_factor):
        super(EDSR, self).__init__()
        self.init_feature = nn.Conv2d(3, 256, 3, 1, 1)
        self.body = ResidualGroup(256, 32)
        if upscale_factor == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(256, 256 * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 3, 3, 1, 1))
        if upscale_factor == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(256, 256 * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 256 * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 3, 3, 1, 1))


    def forward(self, x):
        buffer = self.init_feature(x)
        buffer = self.body(buffer)
        out = self.upscale(buffer)

        return out


class ResB(nn.Module):
    def __init__(self, n_feat):
        super(ResB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
            if i == 0: modules_body.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = 0.1 * self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            ResB(n_feat) \
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

if __name__ == "__main__":
    net = EDSR(upscale_factor=2).cuda()
    from thop import profile
    input = torch.randn(1, 3, 128, 128).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of parameters: %.2fM' % (params/1e6))
    print('   Number of FLOPs: %.2fG' % (flops*2/1e9))
