"""
Stage 4a: Pyramid unit test -- run this FIRST, before touching train_stage4.py.

Confirms reconstruct(build_pyramid(x)) == x when nothing in the pyramid is modified.
This has to hold by construction (reconstruction is algebraic inversion of the build
step), so if max_err isn't tiny (~1e-5, floating point noise), the bug is in
models/pyramid.py itself -- fix it here before it contaminates anything downstream.

Usage:
    python test_pyramid.py
"""

import torch

from models.pyramid import build_pyramid, reconstruct


def main():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 256, 256)  # random values are fine -- this checks the math, not image content

    print(f"{'levels':<8}{'max_err':<14}{'mean_err':<14}{'result'}")
    all_ok = True
    for levels in [1, 2, 3]:
        lap = build_pyramid(x, levels=levels)
        recon = reconstruct(lap)
        max_err = (recon - x).abs().max().item()
        mean_err = (recon - x).abs().mean().item()
        ok = max_err < 1e-4
        all_ok = all_ok and ok
        print(f"{levels:<8}{max_err:<14.8f}{mean_err:<14.8f}{'OK' if ok else 'FAIL'}")

    print()
    if all_ok:
        print("[PASS] Pyramid build/reconstruct is lossless. Safe to proceed to train_stage4.py.")
    else:
        print("[FAIL] Fix models/pyramid.py before proceeding -- do not train on top of this.")


if __name__ == "__main__":
    main()