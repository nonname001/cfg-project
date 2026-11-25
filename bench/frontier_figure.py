"""Figure: semantic diversity against real-image fidelity on held-out seeds, with the condition-damping curve as reference.

x = precision retention, y = DINO recovery (as in paired_vs_cond.py), on seeds 24-47 of SiT-L_2_night, which no method was chosen
on. Gray line: w4 -> cond 0.5 -> cond 0.75 -> cond 1. Points: the methods, with 95% seed-bootstrap intervals on both axes.
Precision uses the real images in $REAL (default outputs/real_imagenet_val; the report figure uses outputs/real_imagenet_train).
usage: [REAL=<dir>] frontier_figure.py [--out <png>] [--title <text>]     (from the project root)
"""
import argparse, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
from common import LABELS
from guidance_analysis import pairwise_cos_distance
from real_fidelity import knn_radius

RUN = Path("outputs/guidance_delta/SiT-L_2_night")
SEEDS = range(24, 48)
CURVE = [("w4", "w = 4"), ("cond_0.5", "cond 0.5"), ("cond_0.75", "cond 0.75"), ("cond_1", "cond 1")]
METHODS = [("wall_3", "w = 3"), ("score_1", "score damping"), ("mcond_1.3333+mchunks_9", "shift damping"),
           ("mcond_1.3333+mchunks_27", "shift+scale damping")]
SURFACE, INK, MUTED, GRID, SERIES, CURVE_INK = "#fcfcfb", "#1d1d1b", "#6b6a64", "#e6e5e0", "#2a78d6", "#8c8b84"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/guidance_delta/figures/frontier_heldout.png")
    p.add_argument("--title", default="Held-out seeds 24-47: methods vs the condition-damping curve (gray)")
    a = p.parse_args()
    real = np.load(os.environ.get("REAL", "outputs/real_imagenet_val") + "/dino.npz")
    R = {c: real["dino"][real["labels"] == c].astype(np.float64) for c in LABELS}
    rad = {c: knn_radius(R[c]) for c in LABELS}
    names = ["w1"] + [c for c, _ in CURVE + METHODS]
    D = {c: np.stack([np.load(RUN / "embeddings" / c / f"seed_{s}.npz")["dino"] for s in SEEDS]).astype(np.float64) for c in names}

    # Whether each sample lies inside its class's real manifold depends only on the real images: compute once, average per resample.
    inside = {c: np.array([[(np.linalg.norm(D[c][s, i][None] - R[l], axis=-1) <= rad[l]).any() for i, l in enumerate(LABELS)]
                           for s in range(len(SEEDS))]) for c in names}

    def raw(c, idx):
        return inside[c][idx].mean(), np.mean([pairwise_cos_distance(D[c][idx, i]) for i in range(8)])

    def scaled(idx):
        m = {c: raw(c, idx) for c in names}
        return {c: ((m[c][0] - m["w1"][0]) / (m["w4"][0] - m["w1"][0]), (m[c][1] - m["w4"][1]) / (m["w1"][1] - m["w4"][1])) for c in names}

    point = scaled(np.arange(len(SEEDS)))
    rng = np.random.default_rng(0)
    boots = [scaled(np.unique(rng.choice(len(SEEDS), len(SEEDS)))) for _ in range(300)]
    ci = {c: np.percentile([[b[c][0], b[c][1]] for b in boots], [2.5, 97.5], axis=0) for c in names}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs, ys = zip(*[point[c] for c, _ in CURVE])
    ax.plot(xs, ys, color=CURVE_INK, lw=2, zorder=1)
    for (c, label), x, y in zip(CURVE, xs, ys):
        ax.scatter([x], [y], s=36, color=CURVE_INK, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.annotate(label, (x, y), xytext=(-6, 6), textcoords="offset points", ha="right", fontsize=8, color=MUTED)
    # Labels sit beside their own point, away from the neighbours (score damping is left of shift damping on the inverted axis).
    offsets = {"wall_3": (9, -14), "score_1": (-9, 6), "mcond_1.3333+mchunks_9": (9, 6), "mcond_1.3333+mchunks_27": (9, -4)}
    for c, label in METHODS:
        x, y = point[c]
        (xl, yl), (xh, yh) = ci[c]
        ax.errorbar([x], [y], xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="none", ecolor=SERIES, alpha=0.45, lw=1.2, zorder=3)
        ax.scatter([x], [y], s=64, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=4)
        dx, dy = offsets[c]
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha="left" if dx > 0 else "right", fontsize=9, color=INK)
    ax.set_xlabel("Precision retention (real-image fidelity, 1 = as at w = 4)", fontsize=9, color=INK)
    ax.set_ylabel("DINO diversity recovery (0 = w = 4, 1 = w = 1)", fontsize=9, color=INK)
    ax.set_title(a.title, fontsize=10, color=INK, loc="left")
    ax.invert_xaxis()  # fidelity falls to the right, so moving right = trading fidelity for diversity
    ax.grid(color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, facecolor=SURFACE)
    print(a.out)


if __name__ == "__main__":
    main()
