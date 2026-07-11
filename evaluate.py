"""
Stage 6: Evaluation script.

Computes PSNR, SSIM, and LPIPS on the eval15 test split for a trained checkpoint.
Run this after training any stage's model to get real numbers instead of eyeballing
sample grids -- this is what your ablation table (Section 12 of the roadmap PDF)
is built from.

Usage:
    python evaluate.py --root /path/to/LOLdataset --checkpoint runs/stage6/checkpoints/epoch_100.pt

Requires: pip install torchmetrics lpips
"""

import argparse

import lpips
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from data.dataset import LowLightPairDataset
from models.multiscale_attention import MultiScaleAttentionExposureNet


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    test_ds = LowLightPairDataset(args.root, split="eval15", train=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Evaluating on {len(test_ds)} images from eval15.")

    model = MultiScaleAttentionExposureNet(
        channels=args.channels, iters=args.iters, levels=args.levels,
        fusion_channels=args.fusion_channels, patch=args.entropy_patch, bins=args.entropy_bins,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net="alex").to(device)

    psnr_total, ssim_total, lpips_total, n = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for low, high, names in test_loader:
            low, high = low.to(device), high.to(device)
            fused, *_ = model(low)

            psnr_total += psnr_metric(fused, high).item()
            ssim_total += ssim_metric(fused, high).item()
            lpips_total += lpips_metric(fused * 2 - 1, high * 2 - 1).item()  # lpips expects [-1,1]
            n += 1

    print()
    print(f"Results over {n} images:")
    print(f"  PSNR:  {psnr_total / n:.3f} dB   (higher is better)")
    print(f"  SSIM:  {ssim_total / n:.4f}       (higher is better, max 1.0)")
    print(f"  LPIPS: {lpips_total / n:.4f}      (lower is better)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to folder containing our485/ and eval15/")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--fusion_channels", type=int, default=16)
    parser.add_argument("--entropy_patch", type=int, default=8)
    parser.add_argument("--entropy_bins", type=int, default=16)
    args = parser.parse_args()
    main(args)