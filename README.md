# Exposure-Attention Network — Stage 1: Data Pipeline

This is the first piece of the project. The goal of this stage is narrow on purpose: 
**confirm the data loads correctly before any model code is written.**

## 1. Get the LOL dataset

Download LOLv1 (search "LOL dataset low light Wei et al."). 
After extracting, you should have:

```
LOLdataset/
    our485/
        low/    (485 training low-light images)
        high/   (485 training normal-light images)
    eval15/
        low/    (15 test low-light images)
        high/   (15 test normal-light images)
```

If your download has different folder names, just rename or symlink them to match the above mentioned as `dataset.py` expects exactly this layout.

## 2. Install requirements

```bash
pip install torch torchvision matplotlib pillow
```

## 3. Run the sanity check

```bash
python check_data.py --root /path/to/LOLdataset --split our485 --n 4
```

This will:
- Load the dataset and report how many pairs were found
- Print tensor shapes and value ranges (should be `(N,3,256,256)` and `[0,1]`)
- Save `sanity_check.png` — a grid of low/high pairs side by side

**Open `sanity_check.png` and manually confirm each low/high column is the same scene,
just at different brightness.** 

Also run it once on `eval15` to make sure the test split loads too:

```bash
python check_data.py --root /path/to/LOLdataset --split eval15 --n 3
```

## What "done" looks like for this stage

- [ ] Script runs with no errors on both `our485` and `eval15`
- [ ] `sanity_check.png` visually confirms correct pairing
- [ ] No `[WARNING]` about unmatched files (or you understand why some exist)

Once all three are true, this stage is genuinely finished — move to Stage 2

## Files

- `data/dataset.py` — `LowLightPairDataset`, a PyTorch `Dataset` for paired low/high images
- `check_data.py` — the sanity-check script above
