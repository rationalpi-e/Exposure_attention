"""
Stage 5: Entropy-attention fusion training script.

1. OVERFIT TEST:
   python train_stage5.py --root /path/to/LOLdataset --overfit_n 8 --epochs 200

2. FULL TRAINING:
   python train_stage5.py --root /path/to/LOLdataset --epochs 100

Outputs go to --out_dir:
    runs/stage5/checkpoints/epoch_N.pt
    runs/stage5/samples/epoch_N.png              -- rows: low / fused prediction / high
    runs/stage5/samples/epoch_N_attn_weights.png -- rows: attention weight per scale
    runs/stage5/samples/epoch_N_temperature.png  -- entropy-derived temperature map

THE IMPORTANT CHECK FOR THIS STAGE (more important than the loss number):
  - epoch_N_temperature.png should NOT look flat/uniform. It should be visibly
    brighter (= higher temperature, after normalization for display) over dark,
    low-detail regions of the low-light input, and darker (= lower temperature)
    over regions that already have more visual structure.
  - epoch_N_attn_weights.png has one row per scale. If one row is ~white and the
    others are ~black everywhere, the fusion collapsed onto a single scale and
    isn't doing anything meaningful -- that's a real failure mode, not just a
    cosmetic issue, and usually means lambda weights or fusion_channels need
    adjusting.
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_pixel, epoch_recon, epoch_smooth = 0.0, 0.0, 0.0, 0.0
        last_batch, last_extra = None, None

        for low, high, names in train_loader:
            low, high = low.to(device), high.to(device)

            fused, upsampled, diagnostics, attn_weights, temperature = model(low)

            pixel_loss = F.l1_loss(fused, high)
            recon_loss = sum(retinex_recon_loss(R, L, lvl_img) for R, L, lvl_img in diagnostics) / len(diagnostics)
            smooth_loss = sum(structure_aware_smoothness(L, lvl_img) for _, L, lvl_img in diagnostics) / len(diagnostics)

            loss = pixel_loss + args.lambda_recon * recon_loss + args.lambda_smooth * smooth_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = low.size(0)
            epoch_loss += loss.item() * bs
            epoch_pixel += pixel_loss.item() * bs
            epoch_recon += recon_loss.item() * bs
            epoch_smooth += smooth_loss.item() * bs
            last_batch = (low, high, fused)
            last_extra = (attn_weights.detach(), temperature.detach())

        scheduler.step()
        n = len(train_loader.dataset)
        print(f"epoch {epoch + 1}/{args.epochs}  "
              f"total={epoch_loss / n:.4f}  pixel={epoch_pixel / n:.4f}  "
              f"recon={epoch_recon / n:.4f}  smooth={epoch_smooth / n:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            save_samples(last_batch, epoch, sample_dir)
            save_attention_maps(last_extra, epoch, sample_dir)

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


def save_attention_maps(last_extra, epoch, sample_dir, n=4):
    attn_weights, temperature = last_extra
    n = min(n, attn_weights.size(0))
    N = attn_weights.size(1)

    attn_rows = [attn_weights[:n, s:s + 1].cpu().repeat(1, 3, 1, 1) for s in range(N)]
    attn_grid = vutils.make_grid(torch.cat(attn_rows, dim=0), nrow=n)
    vutils.save_image(attn_grid, os.path.join(sample_dir, f"epoch_{epoch + 1}_attn_weights.png"))

    T = temperature[:n].cpu()
    T_norm = (T - T.min()) / (T.max() - T.min() + 1e-8)
    T_grid = vutils.make_grid(T_norm.repeat(1, 3, 1, 1), nrow=n)
    vutils.save_image(T_grid, os.path.join(sample_dir, f"epoch_{epoch + 1}_temperature.png"))

    print(f"saved attention weight grid (rows=scales) and temperature map for epoch {epoch + 1}")


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
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--overfit_n", type=int, default=0,
                         help="If >0, train on only this many images (run this first)")
    parser.add_argument("--out_dir", type=str, default="runs/stage5")
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=20)
    args = parser.parse_args()
    main(args)