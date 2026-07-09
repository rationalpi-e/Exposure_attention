'''
This program check all the files and images load properly in the given format.
    LOLDataset/
            our485/
                low/    *.png
                high/   *.png
            eval15/
                low/    *.png
                high/   *.png
 
our485 = training pairs, eval15 = test/validation pairs.

Pairs are matched by filename (low/xyz.png <-> high/xyz.png).

'''

import random
from pathlib import Path

from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class LowLightPairDataset(Dataset):
    def __init__(self, root, split="our485", crop_size=256, train=True):
        """
        root: path to the LOL dataset root (the folder that contains our485/ and eval15/)
        split: "our485" (train) or "eval15" (test)
        crop_size: random-crop size during training; ignored if train=False
        train: whether to apply random crop + flip augmentation
        """
        self.root = Path(root)
        self.split_dir = self.root / split
        self.low_dir = self.split_dir / "low"
        self.high_dir = self.split_dir / "high"
        self.crop_size = crop_size
        self.train = train

        if not self.low_dir.exists() or not self.high_dir.exists():
            raise FileNotFoundError(
                f"Expected '{self.low_dir}' and '{self.high_dir}' to exist. "
                f"Check that --root points to the folder containing our485/ and eval15/."
            )

        low_files = sorted(f.name for f in self.low_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        high_files = set(f.name for f in self.high_dir.iterdir() if f.suffix.lower() in IMG_EXTS)

        self.pairs = [f for f in low_files if f in high_files]
        missing = sorted(set(low_files) - high_files)
        if missing:
            print(f"[WARNING] {len(missing)} low-light images have no matching high-light pair "
                  f"and will be skipped, e.g. {missing[:5]}")
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No matching low/high pairs found in {self.split_dir}. "
                f"low/ has {len(low_files)} files, high/ has {len(high_files)} files."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name = self.pairs[idx]
        low = Image.open(self.low_dir / name).convert("RGB")
        high = Image.open(self.high_dir / name).convert("RGB")

        if low.size != high.size:
            raise ValueError(f"Size mismatch for {name}: low={low.size}, high={high.size}")

        if self.train:
            low, high = self._random_crop(low, high, self.crop_size)
            low, high = self._random_flip(low, high)

        low = TF.to_tensor(low)    # float tensor in [0,1], shape (3,H,W)
        high = TF.to_tensor(high)
        return low, high, name

    @staticmethod
    def _random_crop(low, high, size):
        '''crops the images randomly and if the images are too small, it adds extra black pixels to it,applied to both low and high'''
        w, h = low.size
        if w < size or h < size:
            # upscale small images rather than failing on out-of-bounds crop
            new_size = (max(w, size), max(h, size))
            low = low.resize(new_size)
            high = high.resize(new_size)
            w, h = low.size
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        box = (x, y, x + size, y + size)
        return low.crop(box), high.crop(box)

    @staticmethod
    def _random_flip(low, high):
        '''flips images left left to right or vice versa applied to both  low and high'''
        if random.random() < 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        return low, high
