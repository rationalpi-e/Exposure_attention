"""
Evaluation script -- updated to support every stage's model, so it can produce
the ablation table (Section 12 of the roadmap: Baseline -> +Retinex ->
+Multi-scale -> +Entropy-attention -> +Full loss).

Every model's forward() returns the corrected/fused prediction as the FIRST
element of its output tuple (BaselineExposureNet, RetinexExposureNet,
MultiScaleExposureNet, MultiScaleAttentionExposureNet all do this), so this
script doesn't need separate unpacking logic per model -- model(low)[0] works
for all four.

Usage (one call per checkpoint you want in the table):
    python evaluate.py --root /path/to/LOLdataset --model_type baseline \
        --checkpoint runs/baseline/checkpoints/epoch_100.pt

    python evaluate.py --root /path/to/LOLdataset --model_type retinex \
        --checkpoint runs/stage3/checkpoints/epoch_100.pt

    python evaluate.py --root /path/to/LOLdataset --model_type multiscale \
        --checkpoint runs/stage4/checkpoints/epoch_100.pt

    python evaluate.py --root /path/to/LOLdataset --model_type multiscale_attention \
        --checkpoint runs/stage5/checkpoints/epoch_100.pt

    python evaluate.py --root /path/to/LOLdataset --model_type multiscale_attention \
        --checkpoint runs/stage6/checkpoints/epoch_100.pt
"""

import argparse

import lpips
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from data.dataset import LowLightPairDataset
from models.curve_net import BaselineExposureNet
from models.retinex import RetinexExposureNet
from models.multiscale import MultiScaleExposureNet
from models.multiscale_attention import MultiScaleAttentionExposureNet


def build_model(args):
    if args.model_type == "baseline":
        return BaselineExposureNet(channels=args.channels, iters=args.iters)
    elif args.model_type == "retinex":
        return RetinexExposureNet(channels=args.channels, iters=args.iters)
    elif args.model_type == "multiscale":
        return MultiScaleExposureNet(channels=args.channels, iters=args.iters, levels=args.levels)
    elif args.model_type == "multiscale_attention":
        return MultiScaleAttentionExposureNet(
            channels=args.channels, iters=args.iters, levels=args.levels,
            fusion_channels=args.fusion_channels, patch=args.entropy_patch, bins=args.entropy_bins,
        )
    raise ValueError(f"Unknown model_type: {args.model_type}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    test_ds = LowLightPairDataset(args.root, split="eval15", train=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Evaluating '{args.model_type}' ({args.checkpoint}) on {len(test_ds)} images from eval15.")

    model = build_model(args).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net="alex").to(device)

    psnr_total, ssim_total, lpips_total, n = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for low, high, names in test_loader:
            low, high = low.to(device), high.to(device)
            pred = model(low)[0].clamp(0, 1)  # first element of every model's output = the prediction

            psnr_total += psnr_metric(pred, high).item()
            ssim_total += ssim_metric(pred, high).item()
            lpips_total += lpips_metric(pred * 2 - 1, high * 2 - 1).item()  # lpips expects [-1,1]
            n += 1

    print()
    print(f"Results for '{args.model_type}' over {n} images:")
    print(f"  PSNR:  {psnr_total / n:.3f} dB   (higher is better)")
    print(f"  SSIM:  {ssim_total / n:.4f}       (higher is better, max 1.0)")
    print(f"  LPIPS: {lpips_total / n:.4f}      (lower is better)")
    print()
    print(f"| {args.model_type} | {psnr_total / n:.3f} | {ssim_total / n:.4f} | {lpips_total / n:.4f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to folder containing our485/ and eval15/")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True,
                         choices=["baseline", "retinex", "multiscale", "multiscale_attention"])
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--fusion_channels", type=int, default=16)
    parser.add_argument("--entropy_patch", type=int, default=8)
    parser.add_argument("--entropy_bins", type=int, default=16)
    args = parser.parse_args()
    main(args)