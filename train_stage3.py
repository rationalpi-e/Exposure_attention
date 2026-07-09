import argparse
import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader , Subset

from data.dataset import LowLightPairDataset
from models.retinex import RetinexExposureNet , retinex_recon_loss , structure_aware_smoothness

def main(args):
    device = "cuda"  if torch.cuda.is_available() else "cpu"
    print("device : ", device)

    train_ds = LowLightPairDataset(args.root , split= "our485", crop_size = args.crop_size , train = True)

    if args.overfit_n:
        idx = list(range(min(args.overfit_n , len(train_ds))))
        train_ds = Subset(train_ds , idx)
        print(f"overfit mode training on {len(train_ds)} images only")

    train_loader = DataLoader(train_ds , batch_size = args.batch_size , shuffle = True, num_workers= args.num_workers , drop_last = True)
    model = RetinexExposureNet(channels = args.channels, iters = args.iters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer , T_max = args.epochs)

    ckpt_dir = os.path.join(args.out_dir , "checkpoints")
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(ckpt_dir , exist_ok = True)
    os.makedirs(sample_dir , exist_ok = True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss , epoch_pixel , epoch_recon , epoch_smooth = 0 ,0 , 0 , 0
        last_batch =  None

        for low, high, names in train_loader:
            low , high = low.to(device), high.to(device)

            out , R , L , L_corrected = model(low)
            pixel_loss = F.l1_loss(out , high)
            recon_loss = retinex_recon_loss(R , L , low)
            smooth_loss = structure_aware_smoothness(L , low)
            loss = pixel_loss + args.lambda_recon * recon_loss + args.lambda_smooth * smooth_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.paramters(),max_norm = 5.0)
            optimizer.step()

            bs = low.size(0)
            epoch_loss += loss.item() * bs
            epoch_pixel += pixel_loss.item() * bs
            epoch_recon += recon_loss.item() * bs
            epoch_smooth += smooth_loss.item() * bs
            last_batch = (low , high , out ,R , L)

        scheduler.step()
        n = len(train_loader.dataset)
        print(f"epoch {epoch + 1}/{args.epochs}  "
            f"total={epoch_loss / n:.4f}  pixel={epoch_pixel / n:.4f}  "
            f"recon={epoch_recon / n:.4f}  smooth={epoch_smooth / n:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}")
 
        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            save_samples(last_batch, epoch, sample_dir)
 
        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")

def save_samples(last_batch, epoch , sample_dir , n = 4):
    low, high, out, R, L = last_batch
    n = min(n, low.size(0))
 
    grid = vutils.make_grid(
        torch.cat([low[:n].cpu(), out[:n].clamp(0, 1).cpu(), high[:n].cpu()], dim=0), nrow=n
    )
    vutils.save_image(grid, os.path.join(sample_dir, f"epoch_{epoch + 1}.png"))
 
    # R and L visualized separately -- check these are NOT degenerate
    # (degenerate case: L looks like flat gray ~1.0 everywhere, R looks identical to low)
    L_vis = L[:n].repeat(1, 3, 1, 1).cpu()  # expand single-channel L to 3 channels for viewing
    R_vis = R[:n].clamp(0, 1).cpu()
    rl_grid = vutils.make_grid(torch.cat([R_vis, L_vis], dim=0), nrow=n)
    vutils.save_image(rl_grid, os.path.join(sample_dir, f"epoch_{epoch + 1}_RL.png"))
 
    print(f"saved samples for epoch {epoch + 1} (main grid + R/L grid)")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to folder containing our485/ and eval15/")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--lambda_recon", type=float, default=0.1, help="Weight on Retinex reconstruction loss")
    parser.add_argument("--lambda_smooth", type=float, default=0.05, help="Weight on illumination smoothness loss")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--overfit_n", type=int, default=0,
                         help="If >0, train on only this many images (run this first)")
    parser.add_argument("--out_dir", type=str, default="runs/stage3")
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=20)
    args = parser.parse_args()
    main(args)
