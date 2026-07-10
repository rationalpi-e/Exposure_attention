"""
Stage 5: Multi-scale correction + entropy-guided attention fusion.

Identical to Stage 4 (models/multiscale.py) except the uniform average is
replaced with EntropyAttentionFusion. Keeping Stage 4's file untouched on
purpose -- you now have both a "uniform fusion" and "attention fusion" model
to directly compare in your ablation table.
"""

import torch.nn as nn
import torch.nn.functional as F

from models.pyramid import build_gaussian_pyramid
from models.retinex import RetinexExposureNet
from models.attention_fusion import EntropyAttentionFusion


class MultiScaleAttentionExposureNet(nn.Module):
    def __init__(self, channels=32, iters=8, levels=2, fusion_channels=16, patch=8, bins=16):
        super().__init__()
        self.retinex_net = RetinexExposureNet(channels=channels, iters=iters)
        self.levels = levels
        self.fusion = EntropyAttentionFusion(
            num_scales=levels + 1, channels=fusion_channels, patch=patch, bins=bins
        )

    def forward(self, x):
        pyramid = build_gaussian_pyramid(x, self.levels)

        corrected, diagnostics = [], []
        for level_img in pyramid:
            out, R, L, L_corrected = self.retinex_net(level_img)
            corrected.append(out)
            diagnostics.append((R, L, level_img))

        full_size = pyramid[0].shape[-2:]
        upsampled = [
            F.interpolate(c, size=full_size, mode="bilinear", align_corners=False)
            for c in corrected
        ]

        fused, attn_weights, temperature = self.fusion(upsampled, x)

        return fused, upsampled, diagnostics, attn_weights, temperature