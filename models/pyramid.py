"""
Stage 4: Pyramid utilities.

Two pyramid types, both built on the same Gaussian pyramid underneath:

- build_gaussian_pyramid: [full-res, half-res, quarter-res, ...]  the multi-scale model corrects independently.

- build_pyramid / reconstruct: the Laplacian pyramid (detail bands + coarse base).
  This one must be mathematically lossless when nothing is modified --
  reconstruct(build_pyramid(x)) == x -- as reconstruction is just algebraic
  inversion of the construction step. If test_pyramid.py doesn't confirm that,
  something is wrong here, not in the training loop.
"""

import torch.nn.functional as F


def build_gaussian_pyramid(img, levels):
    """Returns [full-res, half-res, quarter-res, ...] -- levels+1 images total."""
    gauss = [img]
    for _ in range(levels):
        gauss.append(F.avg_pool2d(gauss[-1], 2))   #average polling layer
    return gauss


def build_pyramid(img, levels=2):
    """Laplacian pyramid: [detail_0 (finest), detail_1, ..., base (coarsest)]."""
    gauss = build_gaussian_pyramid(img, levels)
    lap = []
    for i in range(levels):
        up = F.interpolate(gauss[i + 1], size=gauss[i].shape[-2:], mode="bilinear", align_corners=False)   #upsampling of image
        lap.append(gauss[i] - up)
    lap.append(gauss[-1])
    return lap


def reconstruct(lap):
    img = lap[-1]
    for detail in reversed(lap[:-1]):
        img = F.interpolate(img, size=detail.shape[-2:], mode="bilinear", align_corners=False) + detail
    return img