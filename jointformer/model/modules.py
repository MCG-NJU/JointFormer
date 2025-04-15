"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from jointformer.model.group_modules import *
from jointformer.model.cbam import CBAM
from einops import rearrange


class FeatureFusionBlock(nn.Module):
    def __init__(self, f16_dim, enhanced_f16_dim, g_mid_dim, g_out_dim):
        super().__init__()

        # self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(f16_dim+enhanced_f16_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, f16, enhanced_f16):
        """

        @param f16: [bs_obj, f16_dim, h, w]
        @param enhanced_f16: [bs_obj, enhanced_f16_dim, h, w]
        @return:
            [bs_obj, g_out_dim, h, w]
        """

        g = torch.cat([f16, enhanced_f16], dim=1)
        g = self.block1(g)
        r = self.attention(g)
        g = self.block2(g + r)

        return g


class FeatureFuser(nn.Module):
    # Used after the decoder, multi-scale feature
    def __init__(self, g_dims, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g16_conv = GConv2D(g_dims[0], hidden_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], hidden_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], hidden_dim, kernel_size=1)

        # self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)
        #
        # nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g16, g8, g4):
        """

        @param g16: [bs_obj, g_dims[0]=384, h16, w16]
        @param g8: [bs_obj, g_dims[1]=256, h8, w8]
        @param g4: [bs_obj, g_dims[2]=256+1, h4, w4]
        @return:
            g = [bs_obj, hidden_dim=2563=768, h16, w16]

        """

        # g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1 / 2)) + \
        #     self.g4_conv(downsample_groups(g[2], ratio=1 / 4))

        g16 = self.g16_conv(g16)
        g8 = self.g8_conv(downsample_groups(g8, ratio=1/2))
        g4 = self.g4_conv(downsample_groups(g4, ratio=1/4))

        g = torch.cat([g16, g8, g4], dim=1) # # [bs_obj, hidden_dim=256*3=768, h16, w16]

        return g


class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        # self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        """

        @param skip_f: [bs_obj, skip_dim, h2×, w2×]
        @param up_g: [bs_obj, g_up_dim, h, w]
        @return:    [bs_obj, g_out_dim, h2×, w2×]
        """
        skip_f = self.skip_conv(skip_f)
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = skip_f + g
        g = self.out_conv(g)
        return g


class Decoder(nn.Module):
    def __init__(self, f16_dim=768, f8_dim=384, f4_dim=256):
        super().__init__()

        self.fuser_16 = FeatureFusionBlock(f16_dim=f16_dim, enhanced_f16_dim=f16_dim, g_mid_dim=f8_dim, g_out_dim=f8_dim)

        self.up_16_8 = UpsampleBlock(skip_dim=f8_dim, g_up_dim=f8_dim, g_out_dim=f4_dim)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(skip_dim=f4_dim, g_up_dim=f4_dim, g_out_dim=256)  # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

        self.fuser_all = FeatureFuser(g_dims=[384, 256, 256 + 1], hidden_dim=256)

    def forward(self, multi_scale_features, enhanced_f16):

        """

        @param multi_scale_features: f16, f8, f4, [bs, obj_num, c, h, w]
        @param enhanced_f16: [bs, obj_num, c, h, w]
        @return:
            logits [bs, obj_num, H, W]
            fusion_feature [bs, obj_num, 256*3=768, h16, w16]
        """
        f16, f8, f4 = multi_scale_features
        batch_size, num_objects = f16.shape[:2]

        # [bs, obj_num, c, h, w] -> [bs_obj, c, h, w]
        enhanced_f16 = rearrange(enhanced_f16, 'b o c h w -> (b o) c h w')
        f16 = rearrange(f16, 'b o c h w -> (b o) c h w')
        f8 = rearrange(f8, 'b o c h w -> (b o) c h w')
        f4 = rearrange(f4, 'b o c h w -> (b o) c h w')

        g16 = self.fuser_16(f16=f16, enhanced_f16=enhanced_f16)  # [bs_obj, f8_dim=384, h16, w16]

        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)
        logits = self.pred(F.relu(g4))

        fusion_feature = self.fuser_all(g16=g16, g8=g8, g4=torch.cat([g4, logits], dim=1))  # [bs_obj, 256*3, h16, w16]
        fusion_feature = fusion_feature.view(batch_size, num_objects, *fusion_feature.shape[1:])

        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return logits, fusion_feature
