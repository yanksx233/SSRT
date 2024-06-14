import math
import torch
import numbers
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from natten.functional import natten2dav, natten2dqkrpb
from torch.nn.init import trunc_normal_


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    # images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches   # [N, C*k*k, L], L is the total number of such blocks


def to_3d(x):
    B, C, H, W = x.shape
    x = x.view(B, C, H * W).transpose(1, 2).contiguous()
    return x


def to_4d(x, H, W):
    B, L, C = x.shape
    x = x.transpose(1, 2).contiguous().view(B, C, H, W)
    return x


def window_partition(x, window_size):
    windows = rearrange(x, 'b c (h ws1) (w ws2) -> (b h w) c ws1 ws2', ws1=window_size, ws2=window_size)
    return windows


def window_reverse(windows, H, W):
    window_size = windows.shape[-1]
    x = rearrange(windows, '(b h w) c ws1 ws2 -> b c (h ws1) (w ws2)', h=H//window_size, w=W//window_size)
    return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()

#         hidden_features = int(dim*ffn_expansion_factor)
#         self.body = nn.Sequential(
#             nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
#             nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias),
#             nn.GELU(),
#             nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#         )

#     def forward(self, x):
#         x = self.body(x)
#         return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MultiHeadTransposedAttention(nn.Module):
    def __init__(self, num_heads):
        super(MultiHeadTransposedAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
    def forward(self, qkv):
        _, _, h, w = qkv.shape

        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return out


class DualWinTransposedAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size):
        super(DualWinTransposedAttention, self).__init__()
        assert dim % 2 == 0 and num_heads % 2 == 0, 'Number of dimension and head should be even'
        self.window_size = window_size
        self.shift_size = window_size // 2
        
        self.qkv = nn.Sequential(nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias),
                                nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias))
        self.trans_att1 = MultiHeadTransposedAttention(num_heads//2)
        self.trans_att2 = MultiHeadTransposedAttention(num_heads//2)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        h, w = x.shape[-2:]

        qkv = self.qkv(x)
        qkv1, qkv2 = qkv.chunk(2, dim=1)

        windows1 = window_partition(qkv1, self.window_size)   # (num_windows*B, C'*3, window_size, window_size)
        windows1 = self.trans_att1(windows1)
        x1 = window_reverse(windows1, h, w)   # (B, C', H, W)

        shifted_qkv2 = torch.roll(qkv2, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))
        windows2 = window_partition(shifted_qkv2, self.window_size)
        windows2 = self.trans_att2(windows2)
        shifted_x2 = window_reverse(windows2, h, w)
        x2 = torch.roll(shifted_x2, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

        out = torch.cat([x1, x2], dim=1)
        out = self.project_out(out)
        return out


class TransposedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, window_size, LayerNorm_type):
        super(TransposedTransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = DualWinTransposedAttention(dim, num_heads, bias, window_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.beta
        x = x + self.ffn(self.norm2(x)) * self.gamma

        return x


class NeighborhoodAttention(nn.Module):
    def __init__(self, dim, num_heads, kernel_size, dilation=1, rpb=True, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Sequential(nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias),
                                nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=qkv_bias))

        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)

        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W, = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, 3, self.num_heads, self.head_dim, H, W)
            .permute(1, 0, 2, 4, 5, 3)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = natten2dqkrpb(q, k, self.rpb, self.kernel_size, self.dilation) * self.temperature
        attn = attn.softmax(dim=-1)
        x = natten2dav(attn, v, self.kernel_size, self.dilation)   # [B, head, H, W, C]
        x = x.permute(0, 1, 4, 2, 3).reshape(B, C, H, W)

        return self.proj(x)


class NeighborhoodTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size, LayerNorm_type, ffn_expansion_factor, bias):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = NeighborhoodAttention(dim, num_heads, kernel_size)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) * self.beta
        x = x + self.ffn(self.norm2(x)) * self.gamma

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size, ffn_expansion_factor, bias, window_size, LayerNorm_type):
        super().__init__()
        self.body = nn.Sequential(
            NeighborhoodTransformerBlock(dim, dim//32, kernel_size, LayerNorm_type, ffn_expansion_factor, bias),
            TransposedTransformerBlock(dim, num_heads, ffn_expansion_factor, bias, window_size, LayerNorm_type),
        )

    def forward(self, x):
        x = self.body(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(CrossAttention, self).__init__()
        self.reduction = reduction
        self.match = nn.Sequential(nn.Conv2d(dim, dim//reduction, 1),
                                   nn.Conv2d(dim//reduction, dim//reduction, 3, 1, 1, groups=dim//reduction))
        self.value = nn.Sequential(nn.Conv2d(dim, dim, 1),
                                   nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))

    def adjacent(self, x, num_cats, stride=1):
        # x: [B, C, H, W]
        # num_cats: the number of adjacent
        x_cat = x
        for i in range(1, num_cats+1):
            x_up = torch.cat((x[:, :, i*stride:, :], x[:, :, :i*stride, :]), dim=2)
            x_down = torch.cat((x[:, :, -i*stride:, :], x[:, :, :-i*stride, :]), dim=2)
            x_cat = torch.cat((x_cat, x_up, x_down), dim=3)
        return x_cat

    def index_select(self, unfold_features, dim, index):
        # unfold_features: [B*H, k*k*c, W1]
        # dim: which dim is W in return
        # index: [B*H, W]
        # return: [B*H, k*k*c, W]

        views = [unfold_features.size(0)] + [-1 if i == dim else 1 for i in range(1, len(unfold_features.size()))]
        expanse = list(unfold_features.size())
        expanse[0] = -1
        expanse[dim] = -1   # [-1, k*k*c, -1]
        index = index.view(views).expand(expanse)   # [B*H, W] -> [B*H, 1, W] -> [B*H, k*k*c, W]
        return torch.gather(unfold_features, dim, index)

    def forward(self, x, y, num_cats):
        x_match = self.match(x)     # [B, c, H, W1]
        y_match = self.match(y)     # [B, c, H, W2]
        x_match = F.interpolate(x_match, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False) # [B, c, h, w1]
        y_match = F.interpolate(y_match, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=False) # [B, c, h, w2]
        y_value = self.value(y)     # [B, C, H, W2]

        B, c, h, w1 = x_match.shape
        _, _, _, w2 = y_match.shape
        _, C, H, W1 = x.shape
        _, _, _, W2 = y.shape
        m = 3 * 3 * c      # k*k*c
        M = 6 * 6 * c * self.reduction  # K*K*C

        y_match = self.adjacent(y_match, num_cats, stride=1)  # [B, c, h, w2*(1+cats*2)]

        x_match_unfold = extract_image_patches(x_match, ksizes=(3, 3), strides=(1, 1), rates=(1, 1), padding='same')    # [B, m, h*w1]
        y_match_unfold = extract_image_patches(y_match, ksizes=(3, 3), strides=(1, 1), rates=(1, 1), padding='same')    # [B, m, h*w2*n]

        x_match_unfold = x_match_unfold.view(B, m, h, w1).permute(0, 2, 1, 3).contiguous().view(B*h, m, w1)  # [B*h, m, w1]
        y_match_unfold = y_match_unfold.view(B, m, h, w2*(1+num_cats*2)).permute(0, 2, 1, 3).contiguous().view(B*h, m, w2*(1+num_cats*2))   # [B*h, k*k*c, w2*n]

        x_match_unfold = F.normalize(x_match_unfold, dim=1)
        y_match_unfold = F.normalize(y_match_unfold, dim=1)

        sim_map = torch.einsum('ijk,ijl->ikl', x_match_unfold, y_match_unfold)  # [B*h, w1, w2*n]
        score, index = torch.max(sim_map, dim=2)    # [B*h, w1]

        y_value = self.adjacent(y_value, num_cats, stride=2)  # [B, C, H, W2*(1+num_cats*2)]
        
        y_value_unfold = extract_image_patches(y_value, ksizes=(6, 6), strides=(2, 2), rates=(1, 1), padding='same')  # [B, M, h*w2*n]
        y_value_unfold = y_value_unfold.view(B, M, h, w2*(1+num_cats*2)).permute(0, 2, 1, 3).contiguous().view(B*h, M, w2*(1+num_cats*2))

        trans_feats = self.index_select(y_value_unfold, 2, index).view(B, h, M, w1).permute(0, 2, 1, 3).contiguous().view(B, M, h*w1) * score.view(B, 1, h*w1)

        divisor = extract_image_patches(torch.ones_like(x), ksizes=(6, 6), strides=(2, 2), rates=(1, 1), padding='same')
        divisor = F.fold(divisor, (H, W1), kernel_size=6, stride=2, padding=2)
        trans_feats = F.fold(trans_feats, (H, W1), kernel_size=6, stride=2, padding=2) / divisor

        if num_cats == 0:     #  symmetry case, have same sim_map for left and right image
            x_value = self.value(x)     # [B, C, H, W1]
            score2, index2 = torch.max(sim_map, dim=1)    # [B*h, w2]

            x_value_unfold = extract_image_patches(x_value, ksizes=(6, 6), strides=(2, 2), rates=(1, 1), padding='same')  # [B, M, h*w1]
            x_value_unfold = x_value_unfold.view(B, M, h, w1).permute(0, 2, 1, 3).contiguous().view(B*h, M, w1)

            trans_feats2 = self.index_select(x_value_unfold, 2, index2).view(B, h, M, w2).permute(0, 2, 1, 3).contiguous().view(B, M, h*w2) * score2.view(B, 1, h*w2)

            if x.shape == y.shape:
                divisor2 = divisor
            else:
                divisor2 = extract_image_patches(torch.ones_like(y), ksizes=(6, 6), strides=(2, 2), rates=(1, 1), padding='same')
                divisor2 = F.fold(divisor2, (H, W2), kernel_size=6, stride=2, padding=2)
            trans_feats2 = F.fold(trans_feats2, (H, W2), kernel_size=6, stride=2, padding=2) / divisor2

            return trans_feats, trans_feats2

        return trans_feats


class CrossMerging(nn.Module):
    def __init__(self, dim):
        super(CrossMerging, self).__init__()
        self.ho_att = CrossAttention(dim)
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, left, right, num_cats):
        if num_cats == 0:
            trans_l, trans_r = self.ho_att(left, right, 0)
        else:
            trans_l = self.ho_att(left, right, num_cats)
            trans_r = self.ho_att(right, left, num_cats)

        feats_l = left + trans_l * self.beta
        feats_r = right + trans_r * self.gamma

        return feats_l, feats_r


class BasicLayer(nn.Module):
    def __init__(self, dim, num_heads, depth, ffn_expansion_factor, bias, kernel_size, window_size, LayerNorm_type, merge):
        super(BasicLayer, self).__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(dim, num_heads, kernel_size, ffn_expansion_factor, bias, window_size, LayerNorm_type)
                                     for i in range(depth)])
        self.merging = CrossMerging(dim) if merge else None

    def forward(self, left, right, num_cats):
        left = self.blocks(left)
        right = self.blocks(right)

        if self.merging is not None:
            left, right = self.merging(left, right, num_cats)
        
        return left, right
        

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, num_feats, scale):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feats, 4 * num_feats, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feats, 9 * num_feats, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'Scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and torch.rand(1) < self.drop_rate:
            return feats[:2]

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats[:2]


class SSRT(nn.Module):
    def __init__(self, in_channel=3, dim=64, upscale=4, num_heads=[2, 2, 2, 2, 2, 2], depths=[4, 4, 4, 4, 4, 4],
                ffn_expansion_factor=2.66, bias=True, kernel_size=9, window_size=16, LayerNorm_type='WithBias', drop_path_rate=0.1,
                num_cats=0, use_checkpoint=False):
        super().__init__()
        self.upscale = upscale
        self.window_size = window_size
        self.num_cats = num_cats
        self.use_checkpoint = use_checkpoint

        self.conv = nn.Conv2d(in_channel, dim, 3, 1, 1)

        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = BasicLayer(dim, num_heads=num_heads[i], depth=depths[i], ffn_expansion_factor=ffn_expansion_factor, 
                               bias=bias, kernel_size=kernel_size, window_size=window_size, LayerNorm_type=LayerNorm_type, 
                               merge=False if i == len(depths)-1 else True,
                              )
            layer = DropPath(drop_path_rate, layer)
            self.layers.append(layer)
        
        self.up = nn.Sequential(Upsample(dim, upscale),
                                nn.Conv2d(dim, in_channel, 3, 1, 1))

    def check_image_size(self, x):
        divisor = self.window_size
        _, _, H, W = x.shape
        mod_pad_h = (divisor - H % divisor) % divisor
        mod_pad_w = (divisor - W % divisor) % divisor
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, left, right):
        H0, W0 = left.shape[-2:]
        H1, W1 = right.shape[-2:]
        _left = self.check_image_size(left)
        _right = self.check_image_size(right)

        _left = self.conv(_left)
        _right = self.conv(_right)
        
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                _left, _right = checkpoint.checkpoint(layer, _left, _right, torch.tensor(self.num_cats))
            else:
                _left, _right = layer(_left, _right, self.num_cats)

        _left = self.up(_left)[..., :H0*self.upscale, :W0*self.upscale]
        _right = self.up(_right)[..., :H1*self.upscale, :W1*self.upscale]

        left_sr = _left + F.interpolate(left, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        right_sr = _right + F.interpolate(right, scale_factor=self.upscale, mode='bicubic', align_corners=False)

        return left_sr, right_sr


if __name__ == '__main__':

    B, H, W1 = 1, 48, 96
    W2 = W1 + 0
    upscale = 4

    device = torch.device("cuda:1")

    depths = [1] * 12
    heads = [4] * len(depths)


    model = SSRT(depths=depths, num_heads=heads, upscale=upscale).to(device)
    total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print("Number of parameter: %.2fM" % (total / 1e6))

    # model.load_state_dict(checkpoint['model_state_dict'])
    # print(checkpoint['model_state_dict'].keys())


    
    # print([p for p in model.layers[0].module.merging.ho_att.match.parameters()][0][0,0])

    cri = nn.MSELoss().to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad is True], lr=1e-3)

    x1 = torch.rand((B, 3, H, W1), device=device)
    x2 = torch.rand((B, 3, H, W2), device=device)
    gt1 = torch.rand((B, 3, H*upscale, W1*upscale), device=device)
    gt2 = torch.rand((B, 3, H*upscale, W2*upscale), device=device)

    model.train()
    # model.eval()
    print('input:', x1.shape, x2.shape)
    # with torch.no_grad():
    y1, y2 = model(x1, x2)
    print('output:', y1.shape, y2.shape)
    
    opt.zero_grad()
    loss = cri(y1, gt1) + cri(y2, gt2)
    loss.backward()
    opt.step()

    model.num_cats = 1
    y1, y2 = model(x1, x2)
    opt.zero_grad()
    loss = cri(y1, gt1) + cri(y2, gt2)
    loss.backward()
    opt.step()


    # print([p for p in model.layers[0].module.merging.ho_att.match.parameters()][0][0,0])
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # # print(model)

    # model.eval()
    # y1, y2 = model(x1, x2)
    # y3, y4 = model(x1, x2)
    # print(y3[0,0,:5,:5])
    # print(y1[0,0,:5,:5])
