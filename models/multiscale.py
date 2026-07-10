"""
Stage 4: Multi-scale correction + uniform fusion.

Builds a Gaussian pyramid (full-res / half-res / quarter-res -- the "zoom levels"),
runs the SAME Retinex correction network on each level (weight-shared: the physical
correction principle doesn't change with scale, only the input resolution does),
upsamples all corrected outputs back to full resolution, and combines them with
plain averaging.

No attention yet -- that's Stage 5. Keeping fusion "dumb" (uniform average) here
on purpose, so the ablation table can later separate "did multi-scale help at all"
from "did the entropy-attention fusion help on top of that."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pyramid import build_gaussian_pyramid
from models.retinex import RetinexExposureNet


class MultiScaleExposureNet(nn.Module):
    def __init__(self, channels=32, iters=8, levels=2):
        super().__init__()
        # one shared RetinexExposureNet applied at every scale -- same weights,
        # different input resolution each time
        self.retinex_net = RetinexExposureNet(channels=channels, iters=iters)
        self.levels = levels  # number of extra downsampled scales beyond full-res

    def forward(self, x):
        pyramid = build_gaussian_pyramid(x, self.levels)  # [full, half, quarter, ...]

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

        fused = torch.stack(upsampled, dim=0).mean(dim=0).clamp(0, 1)  # uniform fusion

        return fused, upsampled, diagnostics