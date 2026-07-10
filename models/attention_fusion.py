"""
Stage 5: Entropy-guided, temperature-scaled attention fusion.

Standard softmax attention is already a Boltzmann distribution:
    weight_i  propto  exp(score_i / T)

Here T is not a fixed hyperparameter -- it's derived from the local Shannon
entropy of the input image. Low-entropy regions (flat, badly exposed) get a
HIGH temperature -> attention spreads out, the network doesn't over-trust any
single scale. High-entropy regions (already well-exposed, more visual detail)
get a LOW temperature -> attention sharpens onto whichever scale the fusion
head is most confident about.

This replaces Stage 4's uniform average with something that adapts per pixel,
per region, based on a measurable property of the image rather than being
learned as a pure black box.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def local_entropy_map(gray, patch=8, bins=16):
    """
    gray: (B,1,H,W) in [0,1]
    Returns (B,1,H,W): per-pixel entropy, computed per (patch x patch) block and
    upsampled back to full resolution (nearest, since entropy is a block statistic,
    not something to smoothly interpolate).
    """
    B, C, H, W = gray.shape
    pad_h = (patch - H % patch) % patch
    pad_w = (patch - W % patch) % patch
    gray_p = F.pad(gray, (0, pad_w, 0, pad_h), mode="replicate")
    Hp, Wp = gray_p.shape[-2:]

    patches = gray_p.unfold(2, patch, patch).unfold(3, patch, patch)          # (B,1,nH,nW,patch,patch)
    patches = patches.contiguous().view(B, 1, Hp // patch, Wp // patch, patch * patch)

    quant = (patches.clamp(0, 1) * (bins - 1)).round().long()                  # (B,1,nH,nW,patch*patch)
    onehot = F.one_hot(quant, num_classes=bins).float()                       # (...,patch*patch,bins)
    probs = onehot.mean(dim=-2).clamp(min=1e-8)                                # (B,1,nH,nW,bins)
    entropy = -(probs * probs.log()).sum(dim=-1)                               # (B,1,nH,nW)

    return F.interpolate(entropy, size=(H, W), mode="nearest")


def temperature_from_entropy(entropy, entropy_max, T_min=0.3, T_max=3.0):
    """Low entropy -> high T (spread attention). High entropy -> low T (sharp attention)."""
    H_norm = (entropy / (entropy_max + 1e-8)).clamp(0, 1)
    return T_max - (T_max - T_min) * H_norm


class EntropyAttentionFusion(nn.Module):
    def __init__(self, num_scales, channels=16, patch=8, bins=16, T_min=0.3, T_max=3.0):
        super().__init__()
        self.patch = patch
        self.bins = bins
        self.T_min = T_min
        self.T_max = T_max
        self.entropy_max = math.log(bins)  # theoretical max entropy for `bins` categories

        in_ch = 3 * num_scales
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_scales, 3, padding=1),
        )

    def forward(self, scale_images, reference_image):
        """
        scale_images: list of N tensors (B,3,H,W), already upsampled to the same resolution
        reference_image: (B,3,H,W) -- the original low-light input; entropy is computed
                          from this, since it's what tells us how "hard" a region is.
        """
        stacked = torch.stack(scale_images, dim=1)          # (B,N,3,H,W)
        concat = torch.cat(scale_images, dim=1)              # (B,3N,H,W)
        attn_logits = self.head(concat)                       # (B,N,H,W)

        gray = reference_image.mean(1, keepdim=True)          # (B,1,H,W)
        entropy = local_entropy_map(gray, patch=self.patch, bins=self.bins)
        T = temperature_from_entropy(entropy, self.entropy_max, self.T_min, self.T_max)  # (B,1,H,W)

        attn_weights = F.softmax(attn_logits / T, dim=1)      # (B,N,H,W) -- Boltzmann-style fusion

        fused = (stacked * attn_weights.unsqueeze(2)).sum(dim=1)  # (B,3,H,W)
        return fused.clamp(0, 1), attn_weights, T