"""
Stage 2 diagnostic: is the loss plateau caused by the curve running out of range?

Checks two concrete things instead of guessing:
  1. Has `alpha` (CurveNet's output) saturated near the tanh bounds (+-1)?
     If a large fraction of pixels are pinned near +-1, the brightening curve
     has maxed out what it can do -- a real capacity ceiling, not an
     optimization problem.
  2. Where is the remaining pixel error concentrated -- in dark ground-truth
     regions, bright ones, or spread evenly? This tells you whether it's
     specifically a brightening problem or something more general.

Usage:
    python diagnose_alpha.py --root /path/to/LOLdataset \
        --checkpoint runs/baseline/checkpoints/epoch_200.pt --overfit_n 8
"""

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import LowLightPairDataset
from models.curve_net import BaselineExposureNet


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # augment=False -- must match how the model was actually trained/overfit,
    # otherwise you're diagnosing against different crops than it learned on
    ds = LowLightPairDataset(args.root, split="our485", crop_size=args.crop_size,
                              train=True, augment=False)
    if args.overfit_n:
        ds = Subset(ds, list(range(min(args.overfit_n, len(ds)))))

    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    low, high, names = next(iter(loader))
    low, high = low.to(device), high.to(device)

    model = BaselineExposureNet(channels=args.channels, iters=args.iters).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        out, alpha = model(low)

    frac_saturated = (alpha.abs() > 0.95).float().mean().item()
    pixel_err = (out - high).abs()

    print(f"alpha stats:  min={alpha.min().item():.4f}  max={alpha.max().item():.4f}  "
          f"mean={alpha.mean().item():.4f}  std={alpha.std().item():.4f}")
    print(f"fraction of alpha with |alpha| > 0.95 (near tanh saturation): {frac_saturated:.2%}")
    print()
    print(f"per-pixel abs error:  mean={pixel_err.mean().item():.4f}  max={pixel_err.max().item():.4f}")

    high_gray = high.mean(1, keepdim=True)
    err_gray = pixel_err.mean(1, keepdim=True)
    dark_mask = high_gray < 0.3
    bright_mask = high_gray > 0.7
    mid_mask = ~dark_mask & ~bright_mask
    for label, mask in [("DARK (gt<0.3)", dark_mask), ("MID (0.3-0.7)", mid_mask), ("BRIGHT (gt>0.7)", bright_mask)]:
        if mask.any():
            print(f"mean error where ground truth is {label}: {err_gray[mask].mean().item():.4f}  "
                  f"({mask.float().mean().item():.1%} of pixels)")

    print()
    if frac_saturated > 0.3:
        print("[LIKELY CAUSE] A large fraction of alpha is saturated -- the curve has run out of "
              "range. Try increasing --iters, or check if some target pixels need MORE than 8 "
              "iterations of correction to reach.")
    else:
        print("[NOT SATURATION] alpha isn't broadly saturated, so the curve still has headroom. "
              "The plateau is more likely a genuine capacity/receptive-field limit of this small "
              "network -- try --channels 64, or test --overfit_n 1 to isolate capacity from "
              "cross-image interference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--overfit_n", type=int, default=8)
    args = parser.parse_args()
    main(args)