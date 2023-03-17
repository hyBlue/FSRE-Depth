import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from utils.depth_utils import ConvBlock
from .attention_utils import FixedPositionEmbedding, plot_attention_box, conv_bn_relu


def W(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels)

    )


class AnchorDeformAtt(nn.Module):
    
    def __init__(self, in_channels, num_head, num_att_points=16, min_a=0.25, max_a=0.75, adaptive_attn=False):
        super(AnchorDeformAtt, self).__init__()
        self.num_head = num_head
        self.num_att_points = num_att_points
        self.head_dim = in_channels//num_head
        self.min = min_a
        self.max = max_a
        self.adaptive_attn = adaptive_attn

        # print(f'adaptive_attn={self.adaptive_attn}')

        self.size_deform = nn.Sequential(
                    nn.Conv2d(in_channels, num_head*2, kernel_size=(1,1), stride=(1,1), bias=True),
                    nn.Sigmoid()
            )
        self.anchor_deform = nn.Sequential(
                nn.Conv2d(in_channels, num_head*self.num_att_points*2, kernel_size=(1,1), stride=(1,1), bias=True),
                nn.Sigmoid()
        )
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1), stride=(1,1), bias=True)
        if self.adaptive_attn:
            self.query_proj = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        else:
            self.anchor_att = nn.Conv2d(in_channels, num_head*self.num_att_points, kernel_size=(1,1), stride=(1,1), bias=True)

        self.out_proj = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(in_channels)
                    )

    def _where_to_attend(self, query, ref_windows):
        b, l = ref_windows.shape[:2]  
        # ref_windows B, HW, n_head, 4

        offset_ = self.anchor_deform(query) #, B, n_head*16*2, H, W
        offset_ = offset_.flatten(2).permute(0, 2, 1).reshape(b, l, self.num_head, self.num_att_points, 2)

        ref_windows = ref_windows.unsqueeze(-2)
        center, size = ref_windows.split(2, dim=-1)  # B, HW, n_head, 1, 2
        grid = center - 0.5*size + offset_ * size
        grid = torch.clamp(grid, min=0.0, max=1.0)
        return grid.contiguous()  # B, HW, n_head, num_points, 2 (0,1)

    def _create_ref_windows(self, tensor):

        eps = 1e-6
        size_h, size_w = tensor.shape[-2:]
        y_embed = torch.arange(
            1, size_h + 1, dtype=tensor.dtype, device=tensor.device
        )
        x_embed = torch.arange(
            1, size_w + 1, dtype=tensor.dtype, device=tensor.device
        )
        x_, y_ = torch.meshgrid(x_embed, y_embed, indexing="ij")
        y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing="ij")
        x_embed = x_embed.unsqueeze(0).repeat(tensor.shape[0], 1, 1)
        y_embed = y_embed.unsqueeze(0).repeat(tensor.shape[0], 1, 1)

        y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps)
        x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps)
        center = torch.stack([x_embed, y_embed], dim=-1).flatten(1, 2) # B, HW, 2
        center = center.unsqueeze(-2).repeat(1,1, self.num_head, 1) # B, HW, n_head, 2

        size = torch.clamp(self.size_deform(tensor), min=self.min, max=self.max) # B, n_head*2, H, W
        B,_, H, W = size.shape
        size = size.flatten(2).permute(0, 2, 1).reshape(B, H*W, self.num_head, 2)
        #print(size, center)
        ref_box = torch.cat([center, size], dim=-1) # cx, cy, H, W
        return ref_box  # B, HW, n_head, 4
    
    def anchor_attn(self, value, sampling_locations, query):
        """
        Params:
            :value: (B, C, H, W), value pool
            :sampling_locations: (B, L1, nheads, npoints, 2)
            :query: (B, C, H, W), query

        Return:
            :output: (B, C, H, W)
        """
        b, C, h, w = value.shape
        _, l1, nheads, npoints, _ = sampling_locations.shape
        assert l1 == h * w

        # (B*nhead, C/nhead, h, w)
        value_pool = value.view(b * nheads, -1, h, w)
        grid = sampling_locations.transpose(1, 2)
        # (B*nhead, l1, npoint, 2)
        grid = 2 * grid.contiguous().view(b * nheads, l1, npoints, 2) - 1.
        # (B*nheads, C/nheads, l1, npoints)
        sampled_v = F.grid_sample(value_pool, grid, align_corners=True)
        # (B, l1, nhead, C/nhead, npoints)
        sampled_v = sampled_v.reshape(b, nheads, -1, l1, npoints).permute(0, 3, 1, 2, 4).contiguous()

        if self.adaptive_attn:
            # (B, l1, nhead, C/nhead)
            p_query = self.query_proj(query).reshape(b, nheads, -1, l1).permute(0, 3, 1, 2).contiguous()
            # (B, l1, nhead, 1, C/nhead)
            p_query = p_query.unsqueeze(-2)
            # (B, l1, nhead, 1, npoints)
            attn_weights = torch.matmul(p_query, sampled_v)
        else:
            # (B, l1, nhead, 1, npoints)
            attn_weights = self.anchor_att(query).flatten(2).permute(0,2,1).contiguous().reshape(b, l1, nheads, npoints).unsqueeze(-2)
        # Logit to prob
        attn_weights = F.softmax(attn_weights, dim=-1)
        # B, l1, nhead, C/nhead
        value_out = (attn_weights * sampled_v).sum(dim=-1)
        # B, C, h, w
        value_out = value_out.view(b, l1, C).permute(0,2,1).contiguous().reshape(b, C, h, w)
        return value_out

    def forward(self, feat_sd):
        memory = self.value_proj(feat_sd)  # B, C, H, W

        ref_windows = self._create_ref_windows(feat_sd)  # B, HW, n_head, 4
        grid = self._where_to_attend(feat_sd, ref_windows)  # B, HW, nhead, num_att_points, 2
        #print(grid.shape)

        value = self.anchor_attn(memory, grid, query=feat_sd)  # B, C, H, W
        value = self.out_proj(value)
        return value


class LocalTransformer(nn.Module):
    def __init__(self, in_channels, num_head, batch_norm=True, with_sem=True, num_att_points=16, num_att_layers = 2, min_a=0.25, max_a=0.75,
                adaptive_attn=False):
        super(LocalTransformer, self).__init__()
        
        self.with_sem = with_sem
        self.num_head = num_head
        self.num_att_points = num_att_points
        self.head_dim = in_channels//num_head
        self.num_att_layers = num_att_layers

        self.anchor_att_dep = nn.ModuleList(
                    [AnchorDeformAtt(in_channels, num_head, num_att_points=num_att_points, min_a=min_a, max_a=max_a, adaptive_attn=adaptive_attn)
                        for i in range(num_att_layers)]
        )
        if self.with_sem:
            self.anchor_att_seg = nn.ModuleList(
                        [AnchorDeformAtt(in_channels, num_head, num_att_points=num_att_points, min_a=min_a, max_a=max_a, adaptive_attn=adaptive_attn)
                            for i in range(num_att_layers)]
            )
        
        if self.with_sem:
            self.pre_enc1 = conv_bn_relu(batch_norm, in_channels, in_channels//2, kernel_size=1, padding=0)
            self.pre_enc2 = conv_bn_relu(batch_norm, in_channels, in_channels//2, kernel_size=1, padding=0)
        self.pre_enc = conv_bn_relu(batch_norm, in_channels, in_channels, kernel_size=1, padding=0)
        self.post_enc = conv_bn_relu(batch_norm, in_channels, in_channels, kernel_size=1, padding=0)

 
        self.fuse1 = nn.Sequential(ConvBlock(in_channels * 2 + in_channels, in_channels),
                                  nn.Conv2d(in_channels, in_channels, kernel_size=1))
        if self.with_sem:
            self.fuse2 = nn.Sequential(ConvBlock(in_channels * 2 + in_channels, in_channels),
                                  nn.Conv2d(in_channels, in_channels, kernel_size=1))       

    def forward(self, inputs):
        if self.with_sem:
            return self.forward_cross(inputs[0], inputs[1])
        else:
            return self.forward_self(inputs[0])

    def forward_self(self, feat_d):
        feat_d_ = self.pre_enc(feat_d)
        feat_d_ = torch.cat([
                self.aspp1(feat_d_), self.aspp1(feat_d_),
                self.aspp1(feat_d_), self.aspp1(feat_d_),
                self.aspp1(feat_d_),
        ], 1)  # B, C//4*5 , H, W
        feat_d_ = self.post_enc(feat_d_)
        value = feat_d
        return self.fuse1(torch.cat([value, feat_d], 1))


    def forward_cross(self, feat_d, feat_s): 
        feat_sd = torch.cat([
                self.pre_enc1(feat_d),
                self.pre_enc2(feat_s)
        ], 1)
        feat_sd = self.pre_enc(feat_sd)
        value_dep = self.post_enc(feat_sd)
        value_seg = value_dep.clone()
        for layer in self.anchor_att_dep:
            value_dep = layer(value_dep)

        for layer in self.anchor_att_seg:
            value_seg = layer(value_seg)


        return self.fuse1(torch.cat([feat_d, feat_sd, value_dep], 1)), self.fuse2(torch.cat([feat_s, feat_sd, value_seg], 1))


