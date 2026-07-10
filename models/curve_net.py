"""
Stage 2: Baseline curve network.

A small CNN predicts a per-pixel curve parameter map A(x), which is applied
iteratively to brighten the image:

    LE_n(x) = LE_{n-1}(x) + A(x) * LE_{n-1}(x) * (1 - LE_{n-1}(x))

This is the Zero-DCE-style correction curve: bounded, monotonic, differentiable.
No Retinex split, no multi-scale, no attention yet — that's intentional. This
stage only proves the basic "network predicts a curve that brightens the image
correctly" mechanism works, on a single scale, with one loss term.
"""

import torch
import torch.nn as nn


class CurveNet(nn.Module):
    """Predicts a per-pixel curve parameter map A(x) in (-1, 1)."""

    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        alpha = torch.tanh(self.conv4(x3))   # (-1, 1), per-pixel, per-channel
        return alpha


def le_curve(x, alpha, iters=8):
    """Applies the bounded brightening curve iteratively."""
    for _ in range(iters):
        x = x + alpha * x * (1 - x)
    return x.clamp(0, 1)


class BaselineExposureNet(nn.Module):
    """CurveNet + the iterative curve, wrapped as a single forward pass."""

    def __init__(self, channels=32, iters=8):
        super().__init__()
        self.curve_net = CurveNet(channels)
        self.iters = iters

    def forward(self, x):
        alpha = self.curve_net(x)
        out = le_curve(x, alpha, self.iters)
        return out, alpha