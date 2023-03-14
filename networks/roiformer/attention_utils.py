import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math

import numpy as np
import matplotlib.pyplot as plt


def conv_bn_relu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    #padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=True),
            nn.ReLU(inplace=True),
        )

class FixedPositionEmbedding(nn.Module):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super(FixedPositionEmbedding, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x,):
        # x: B x C x H x W
        eps = 1e-6
        size_h, size_w = x.shape[-2:]
        y_embed = torch.arange(1, size_h + 1, dtype=x.dtype, device=x.device)
        x_embed = torch.arange(1, size_w + 1, dtype=x.dtype, device=x.device)
        y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")
        x_embed = x_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)
        y_embed = y_embed.unsqueeze(0).repeat(x.shape[0], 1, 1)

        if self.normalize:
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (
            2 * dim_t.div(2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2)

        return pos


class Linear(nn.Module):
    def __init__(self, in_dim, hid_dim=None, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hid_dim, in_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads=4, mlp_ratio=2., qkv_bias=False,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        #self.pos_enc = FixedPositionEmbedding(dim//2)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Linear(in_dim=dim, hid_dim=mlp_hidden_dim, act_layer=act_layer)


    def forward(self, x, pos_x):
        #print(x.shape, self.pos_enc(x).shape)
        norm_x = self.norm1(x +  pos_x)
        x = x + self.attn(norm_x)
        x = x + self.mlp(self.norm2(x))

        return x

class BlockNaive(nn.Module):

    def __init__(self, dim, num_heads=4, mlp_ratio=2., qkv_bias=False,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias)

    def forward(self, x):
        #print(x.shape, self.pos_enc(x).shape)
        x = x + self.attn(x)
        return x


def plot_attention_box(center, size, h=12, w=40, n_head=8):
    # center : B, L, n_head, 1, 2
    for k in range(10):
        cens = center[k,:,:,0,:].data.cpu().numpy()
        for i in range(n_head):
            cen = cens[:,0,:]
            plt.scatter(cen[:,1],cen[:,0], c=np.random.rand(3,))
        plt.show()



if __name__ == "__main__":
    net = Block(256, num_heads=8, mlp_ratio=2)
    net = net.cuda()

    x = torch.rand(2, 240, 256).cuda()
    net(x)