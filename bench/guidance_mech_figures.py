"""Report Figure 3: effective rank of pooled activations by block and time, under each intervention minus w = 4.

Reads <run>/analysis/mech_blocks.csv from guidance_mech.py and writes <run>/analysis/figures/m3b_block_rank.png.
usage: guidance_mech_figures.py --run <dir>
"""
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to files without a display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, "bench")
from guidance_figures import DIVERGING, INK, MUTED, SURFACE, header, read, select

CONDITIONS = ["score_1", "cond_0.5", "score_1+cond_0.5"]  # one panel each
TITLE = "All three raise effective rank, most in the later blocks; the combination raises it most"


def fig_block_rank(a, out):
    """One (time x block) heat map per condition of effective rank minus w=4's, on one shared symmetric color scale."""
    b = read(a / "mech_blocks.csv")
    ref = select(b, condition="w4")
    t = np.unique(ref["t"])
    # (24 blocks, T) array of column k, looked up one (block, step) cell at a time.
    grid = lambda d, k: np.array([[d[k][(d["block"] == blk) & (d["step"] == s)][0] for s in range(len(t))] for blk in range(24)])
    base = grid(ref, "eff_rank")
    maps = [grid(select(b, condition=c), "eff_rank") - base for c in CONDITIONS]
    limit = max(np.nanmax(np.abs(m)) for m in maps)  # one symmetric color scale, so the three conditions compare directly
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True, sharey=True, dpi=170, facecolor=SURFACE, squeeze=False)
    fig.subplots_adjust(top=0.8, bottom=0.14, left=0.06, right=0.9)  # before the colorbar, so it matches the panels
    for col, (c, m) in enumerate(zip(CONDITIONS, maps)):
        ax = axes[0, col]
        im = ax.imshow(m, aspect="auto", origin="lower", extent=(t[0], t[-1], -0.5, 23.5), cmap=DIVERGING,  # blue = higher than w=4, red = lower
                       norm=TwoSlopeNorm(0, -limit, limit), interpolation="nearest")
        ax.axvline(0.3, color=INK, lw=0.8, ls=":")  # end of the intervention window
        ax.tick_params(colors=MUTED, length=0, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{c.replace('+', ' + ')}: effective rank, minus w=4", loc="left", color=INK, fontsize=9)
        if col == 0:
            ax.set_ylabel("block", color=MUTED, fontsize=9)
        ax.set_xlabel("t  (0 = noise, 1 = data)", color=MUTED, fontsize=9)
    bar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.015, pad=0.01)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, length=0, labelsize=8)
    header(fig, TITLE, "Pooled conditional features. Effective rank of the 16 seeds per class (max 15), averaged over 8 classes. Blue = higher than w=4, red = lower.", 0.93)
    fig.savefig(out / "m3b_block_rank.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    a = p.parse_args()
    analysis = Path(a.run) / "analysis"
    out = analysis / "figures"
    out.mkdir(exist_ok=True)
    fig_block_rank(analysis, out)


if __name__ == "__main__":
    main()
