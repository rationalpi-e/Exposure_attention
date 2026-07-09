import argparse

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset import LowLightPairDataset


def main(root, split, n, crop_size):
    ds = LowLightPairDataset(root, split=split, crop_size=crop_size, train=True)
    print(f"[OK] Found {len(ds)} paired images in '{split}'.")

    loader = DataLoader(ds, batch_size=n, shuffle=True)
    low, high, names = next(iter(loader))

    print("low  batch shape:", tuple(low.shape), " min/max:", round(low.min().item(), 3), round(low.max().item(), 3))
    print("high batch shape:", tuple(high.shape), " min/max:", round(high.min().item(), 3), round(high.max().item(), 3))

    assert low.shape == high.shape, "low/high batch shapes don't match — something is wrong in the Dataset class"
    assert 0.0 <= low.min() and low.max() <= 1.0, "low tensor is not in [0,1] range"
    assert 0.0 <= high.min() and high.max() <= 1.0, "high tensor is not in [0,1] range"
    print("[OK] Shapes and value ranges look correct.")

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        axes[0, i].imshow(low[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"low: {names[i]}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(high[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title(f"high: {names[i]}", fontsize=8)
        axes[1, i].axis("off")
  
    out_path = "sanity_check.png"
    plt.savefig(out_path, dpi=120)
    print(f"[OK] Saved {out_path} — open it and confirm each low/high column shows the SAME scene.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                         help="Path to the folder containing our485/ and eval15/")
    parser.add_argument("--split", type=str, default="our485", choices=["our485", "eval15"])
    parser.add_argument("--n", type=int, default=4, help="Number of pairs to display")
    parser.add_argument("--crop_size", type=int, default=256)
    args = parser.parse_args()
    main(args.root, args.split, args.n, args.crop_size)
