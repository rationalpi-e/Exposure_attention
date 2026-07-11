"""
Stage 6: Remaining loss terms.

Full loss stack (matches the roadmap):
    L = L_pixel
      + lambda1 * (L_recon + L_smooth + L_reflectance)
      + lambda2 * L_exposure
      + lambda3 * L_perceptual

L_pixel, L_recon, L_smooth already exist in models/retinex.py.
This file adds the three still missing: reflectance consistency, exposure
control, and perceptual loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def reflectance_consistency_loss(R_low, R_high):
    """
    Same scene -> reflectance should be roughly the same regardless of exposure.
    R_low, R_high: (B,3,H,W), decomposed from the low and high image of the SAME pair.
    """
    return F.l1_loss(R_low, R_high)


def exposure_control_loss(out, patch=16, target=0.6):
    """
    Zero-DCE style: local patch mean intensity should sit near a well-exposed
    target (~0.6 -- not 1.0; pure white is blown out, not "well exposed").
    """
    gray = out.mean(1, keepdim=True)
    pooled = F.avg_pool2d(gray, patch)
    return ((pooled - target) ** 2).mean()


class VGGPerceptualLoss(nn.Module):
    """
    L1 distance between VGG16 features of prediction vs. ground truth, at a
    mid-level layer -- captures texture/structure, not just raw pixel values.

    NOTE: first run downloads pretrained VGG16 weights (~528MB) -- needs an
    internet connection the first time this is instantiated.
    """

    def __init__(self, layer_index=16):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential(*[vgg[i] for i in range(layer_index)]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, out, target):
        out_n = (out - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        f_out = self.slice(out_n)
        f_target = self.slice(target_n)
        return F.l1_loss(f_out, f_target)