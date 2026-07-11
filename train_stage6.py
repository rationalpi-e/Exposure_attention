"""
Stage 6: Full loss stack + tuning.

Same model as Stage 5 (MultiScaleAttentionExposureNet). What's new is the loss:
adds reflectance consistency, exposure control, and perceptual loss on top of
the pixel/recon/smooth terms from Stage 5.

DO NOT enable all three new lambdas at once on your first run. Add them one at
a time, in this order, watching validation PSNR (see evaluate.py) after each:

  1. --lambda_exposure only (cheap, no VGG download needed, stabilizes brightness)
  2. + --lambda_reflect
  3. + --lambda_perceptual (needs internet on first run, downloads VGG16 weights)

Set any lambda to 0 to disable that term entirely -- that's how you do the
one-term-at-a-time ablation the plan calls for, using this same script.

1. OVERFIT TEST:
   python train_stage6.py --root /path/to/LOLdataset --overfit_n 8 --epochs 200 --lambda_perceptual 0

2. FULL TRAINING (once terms are tuned individually):
   python train_stage6.py --root /path/to/LOLdataset --epochs 100

Outputs go to --out_dir:
    runs/stage6/checkpoints/epoch_N.pt
    runs/stage6/samples/epoch_N.png
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset

from data.dataset import LowLightPairDataset
from models.multiscale_attention import MultiScaleAttentionExposureNet
from models.retinex import retinex_recon_loss, structure_aware_smoothness
from models.losses import reflectance_consistency_loss, exposure_control_loss, VGGPerceptualLoss


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_ds = LowLightPairDataset(args.root, split="our485", crop_size=args.crop_size, train=True)

    if args.overfit_n:
        idx = list(range(min(args.overfit_n, len(train_ds))))
        train_ds = Subset(train_ds, idx)
        print(f"[overfit mode] training on {len(train_ds)} images only")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)

    model = MultiScaleAttentionExposureNet(
        channels=args.channels, iters=args.iters, levels=args.levels,
        fusion_channels=args.fusion_channels, patch=args.entropy_patch, bins=args.entropy_bins,
    ).to(device)

    vgg_loss_fn = None
    if args.lambda_perceptual > 0:
        print("Loading VGG16 for perceptual loss (downloads weights on first run)...")
        vgg_loss_fn = VGGPerceptualLoss(layer_index=args.vgg_layer).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        totals = {"total": 0.0, "pixel": 0.0, "recon": 0.0, "smooth": 0.0,
                  "reflect": 0.0, "exposure": 0.0, "perceptual": 0.0}
        last_batch = None

        for low, high, names in train_loader:
            low, high = low.to(device), high.to(device)

            fused, upsampled, diagnostics, attn_weights, temperature = model(low)

            pixel_loss = F.l1_loss(fused, high)
            recon_loss = sum(retinex_recon_loss(R, L, lvl_img) for R, L, lvl_img in diagnostics) / len(diagnostics)
            smooth_loss = sum(structure_aware_smoothness(L, lvl_img) for _, L, lvl_img in diagnostics) / len(diagnostics)

            # reflectance consistency: compare full-res R_low (already computed) against R_high
            R_low_full = diagnostics[0][0]
            L_high = model.retinex_net.illum_net(high)   # decompose the ground-truth image too
            R_high = high / (L_high + 1e-4)
            reflect_loss = reflectance_consistency_loss(R_low_full, R_high)

            exposure_loss = exposure_control_loss(fused, patch=args.exposure_patch, target=args.exposure_target)

            perceptual_loss = vgg_loss_fn(fused, high) if vgg_loss_fn is not None else torch.tensor(0.0, device=device)

            loss = (
                pixel_loss
                + args.lambda_recon * recon_loss
                + args.lambda_smooth * smooth_loss
                + args.lambda_reflect * reflect_loss
                + args.lambda_exposure * exposure_loss
                + args.lambda_perceptual * perceptual_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = low.size(0)
            totals["total"] += loss.item() * bs
            totals["pixel"] += pixel_loss.item() * bs
            totals["recon"] += recon_loss.item() * bs
            totals["smooth"] += smooth_loss.item() * bs
            totals["reflect"] += reflect_loss.item() * bs
            totals["exposure"] += exposure_loss.item() * bs
            totals["perceptual"] += perceptual_loss.item() * bs
            last_batch = (low, high, fused)

        scheduler.step()
        n = len(train_loader.dataset)
        msg = "  ".join(f"{k}={v / n:.4f}" for k, v in totals.items())
        print(f"epoch {epoch + 1}/{args.epochs}  {msg}  lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            save_samples(last_batch, epoch, sample_dir)

        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")


def save_samples(last_batch, epoch, sample_dir, n=4):
    low, high, fused = last_batch
    n = min(n, low.size(0))
    grid = vutils.make_grid(
        torch.cat([low[:n].cpu(), fused[:n].clamp(0, 1).cpu(), high[:n].cpu()], dim=0), nrow=n
    )
    vutils.save_image(grid, os.path.join(sample_dir, f"epoch_{epoch + 1}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to folder containing our485/ and eval15/")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--fusion_channels", type=int, default=16)
    parser.add_argument("--entropy_patch", type=int, default=8)
    parser.add_argument("--entropy_bins", type=int, default=16)
    parser.add_argument("--lambda_recon", type=float, default=0.1)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_reflect", type=float, default=0.1)
    parser.add_argument("--lambda_exposure", type=float, default=1.0)
    parser.add_argument("--lambda_perceptual", type=float, default=0.05)
    parser.add_argument("--exposure_patch", type=int, default=16)
    parser.add_argument("--exposure_target", type=float, default=0.6)
    parser.add_argument("--vgg_layer", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--overfit_n", type=int, default=0,
                         help="If >0, train on only this many images (run this first)")
    parser.add_argument("--out_dir", type=str, default="runs/stage6")
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=20)
    args = parser.parse_args()
    main(args)