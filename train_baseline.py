"""
Stage 2: Baseline training script.

Two modes, and you should run them in this order:

1. OVERFIT TEST (do this first, always):
   python train_baseline.py --root /path/to/LOLdataset --overfit_n 8 --epochs 200

   If the model can't drive the loss near zero on 8 memorized images, nothing
   downstream (Retinex, multi-scale, attention) will work either. This is the
   cheapest bug-catching step available — don't skip it.

2. FULL TRAINING (only after the overfit test passes):
   python train_baseline.py --root /path/to/LOLdataset --epochs 100

Outputs go to --out_dir:
    runs/baseline/checkpoints/epoch_N.pt   -- model weights
    runs/baseline/samples/epoch_N.png      -- low | prediction | high, side by side
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset

from data.dataset import LowLightPairDataset
from models.curve_net import BaselineExposureNet


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

    model = BaselineExposureNet(channels=args.channels, iters=args.iters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        last_batch = None

        for low, high, names in train_loader:
            low, high = low.to(device), high.to(device)

            out, alpha = model(low)
            loss = F.l1_loss(out, high)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * low.size(0)
            last_batch = (low, high, out)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            save_samples(last_batch, epoch, sample_dir)

        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")


def save_samples(last_batch, epoch, sample_dir, n=4):
    low, high, out = last_batch
    n = min(n, low.size(0))
    grid = vutils.make_grid(
        torch.cat([low[:n].cpu(), out[:n].clamp(0, 1).cpu(), high[:n].cpu()], dim=0),
        nrow=n,
    )
    out_path = os.path.join(sample_dir, f"epoch_{epoch + 1}.png")
    vutils.save_image(grid, out_path)
    print(f"saved sample grid (rows = low / prediction / high): {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to folder containing our485/ and eval15/")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32, help="CurveNet width")
    parser.add_argument("--iters", type=int, default=8, help="Curve iterations")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--overfit_n", type=int, default=0,
                         help="If >0, train on only this many images (run this first)")
    parser.add_argument("--out_dir", type=str, default="runs/baseline")
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=20)
    args = parser.parse_args()
    main(args)