"""Effective rank of each block's pooled features under each intervention, from a guidance_capture.py --conditions run.

Writes <run>/analysis/mech_blocks.csv: one row per (condition, block, step) with the effective rank of the 16 seeds' pooled
conditional features, averaged over the 8 classes. guidance_mech_figures.py plots it relative to w = 4 (report Figure 3).
Run from the project root.
usage: guidance_mech.py --run <dir>
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
from guidance_analysis import effective_rank, svd_basis, write_csv


def condition_dirs(run):
    """Every condition folder under run that holds at least one finished batch (skips analysis/ and half-written folders)."""
    return sorted(p for p in run.iterdir() if p.is_dir() and any(p.glob("seed_*/metadata.json")))


def seeds_of(folder):
    """The seeds finished in a condition folder, as sorted integers (folder names look like seed_16)."""
    return sorted(int(p.parent.name[5:]) for p in folder.glob("seed_*/metadata.json"))


def blocks_stage(run, out):
    """Per condition, block and step: effective rank of the conditional pooled features across seeds."""
    rows = []
    for folder in condition_dirs(run):
        seeds = seeds_of(folder)
        pooled = np.stack([np.load(folder / f"seed_{s}" / "pooled.npy", mmap_mode="r") for s in seeds], 0)  # (S, T, 24, 2, 8, 1024)
        t = np.load(folder / f"seed_{seeds[0]}" / "trajectory.npz")["s_t"]
        for b in range(pooled.shape[2]):
            x = pooled[:, :, b].astype(np.float64)  # (S, T, 2, 8, 1024)
            for k in range(len(t)):
                cond = x[:, k, 0]  # branch 0 = conditional: (S, 8, 1024)
                # Center each class's seeds, take the singular values, and average the effective rank over the 8 classes.
                rows.append(dict(condition=folder.name, block=b, step=k, t=t[k],
                                 eff_rank=float(np.mean([effective_rank(svd_basis(cond[:, c] - cond[:, c].mean(0))[0]) for c in range(8)]))))
        print("blocks", folder.name, flush=True)
    write_csv(out / "mech_blocks.csv", rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    a = p.parse_args()
    run = Path(a.run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    blocks_stage(run, out)


if __name__ == "__main__":
    main()
