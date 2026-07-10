"""
Stage 3: Retinex split + smoothness.

Physical model: I(x) = R(x) * L(x)  (reflectance * illumination)

Instead of correcting the raw RGB image directly (Stage 2), we now:
  1. Estimate the illumination map L from the input
  2. Derive reflectance R = I / L
  3. Apply the brightening curve ONLY to L
  4. Reconstruct: out = R * L_corrected

This keeps R (color/texture/identity) untouched, which is why Retinex-based
correction tends to distort color less than brightening raw RGB directly.

Two new loss terms live here too:
  - retinex_recon_loss: enforces that R * L actually reconstructs the input
    (i.e. the decomposition is physically valid, not degenerate)
  - structure_aware_smoothness: a discretized variational smoothness term on L,
    weighted so it doesn't blur across real edges in the image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.curve_net import le_curve


class IlluminationNet(nn.Module):
    """Predicts a single-channel illumination map L(x) in (~0, 1]."""

    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        L = torch.sigmoid(self.conv3(h))
        return L.clamp(min=1e-2)  # avoid near-zero illumination -> division blowup in R = I/L


class IllumCurveNet(nn.Module):
    """Predicts the per-pixel curve parameter map used to correct L (not the full RGB image)."""

    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, L):
        h = self.relu(self.conv1(L))
        h = self.relu(self.conv2(h))
        alpha = torch.tanh(self.conv3(h))
        return alpha


class RetinexExposureNet(nn.Module):
    def __init__(self, channels=32, iters=8):
        super().__init__()
        self.illum_net = IlluminationNet(channels)
        self.curve_net = IllumCurveNet(channels)
        self.iters = iters

    def forward(self, x):
        L = self.illum_net(x)                 # (B,1,H,W)
        R = x / (L + 1e-4)                     # (B,3,H,W) -- physical decomposition
        alpha = self.curve_net(L)
        L_corrected = le_curve(L, alpha, self.iters)
        out = (R * L_corrected).clamp(0, 1)
        return out, R, L, L_corrected


def retinex_recon_loss(R, L, I):
    """R * L should reconstruct the original image I. Penalizes a degenerate decomposition."""
    return F.l1_loss(R * L, I)


def structure_aware_smoothness(L, I, eps=1e-3):
    """
    Discretized variational smoothness term: penalizes gradients in L, but
    weighted down wherever I itself has a strong edge, so real edges aren't blurred.
    """
    gray = I.mean(1, keepdim=True)

    dLx = L[..., :, 1:] - L[..., :, :-1]
    dIx = gray[..., :, 1:] - gray[..., :, :-1]
    wx = torch.exp(-dIx.abs() / eps)

    dLy = L[..., 1:, :] - L[..., :-1, :]
    dIy = gray[..., 1:, :] - gray[..., :-1, :]
    wy = torch.exp(-dIy.abs() / eps)

    return (wx * dLx.abs()).mean() + (wy * dLy.abs()).mean()
