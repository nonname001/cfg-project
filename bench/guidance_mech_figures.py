"""Figures and a summary table for guidance_mech.py output. Writes <run>/analysis/figures/ and mech_summary.csv.

usage: guidance_mech_figures.py --run <dir>

Figures:
  m1   dose response: how each metric moves as damping strength lambda grows, score family vs condition family
  m2   direct vs through-x_t change in the class signal per block, from mech_patch.csv
  m3   block x time maps of spread V and class-signal sharing S, relative to w = 4
  m3b  block x time map of effective rank, relative to w = 4
  m4   per-step guidance statistics for every condition
Colors: blue shades = score damping, orange shades = condition damping, green = the mixes; darker = stronger.
"""
import argparse, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, "bench")
from guidance_analysis import write_csv
from guidance_figures import DIVERGING, GRID, INK, MUTED, SURFACE, header, read, save, select, style

LATE_BLOCKS = list(range(13, 24))
# Each family starts at w4 (strength 0), so its dose-response line begins at the undamped point.
SCORE = ["w4", "score_0.5", "score_1", "score_2"]
COND = ["w4", "cond_0.25", "cond_0.5", "cond_1"]
MIX = ["score_1+cond_0.25", "score_1+cond_0.5"]
STRENGTH = {"w4": 0.0, "score_0.5": 0.5, "score_1": 1.0, "score_2": 2.0, "cond_0.25": 0.25, "cond_0.5": 0.5, "cond_1": 1.0}
# Sequential steps of one hue per family (blue for score, orange for condition), light to dark with strength.
BLUES = {"score_0.5": "#86b6ef", "score_1": "#2a78d6", "score_2": "#104281"}
ORANGES = {"cond_0.25": "#f5a37f", "cond_0.5": "#eb6834", "cond_1": "#a83a12"}
MIXC = {"score_1+cond_0.25": "#1baf7a", "score_1+cond_0.5": "#0b6e4a"}
COLORS = {"w1": INK, "w4": MUTED, **BLUES, **ORANGES, **MIXC}


def summary(a):
    """One row per condition: per-step and per-block metrics averaged over the relevant window, plus final diversity and fidelity."""
    curves, blocks =read(a / "mech_curves.csv"), read(a / "mech_blocks.csv")
    finals = {r["condition"]: r for r in csv.DictReader(open(a / "finals" / "intervention_results.csv"))}
    rows = []
    for c in finals:
        cv, bl = select(curves, condition=c), select(blocks, condition=c)
        late, early = cv["t"] >= 0.3, cv["t"] < 0.3  # after and during the intervention window
        lb = np.isin(bl["block"], LATE_BLOCKS) & (bl["t"] >= 0.3)
        f = finals[c]
        rows.append(dict(condition=c, S_f_cond_late=cv["S_f_cond"][late].mean(), S_delta_early=cv["S_delta"][early].mean(),
                         ratio_early=cv["ratio"][early].mean(), high_share_late=cv["high_share"][late].mean(),
                         late_block_eff_rank=bl["eff_rank"][lb].mean(), late_block_V=bl["V"][lb].mean(),
                         late_block_signal_S=bl["signal_S"][lb].mean(), late_block_signal_norm=bl["signal_norm"][lb].mean(),
                         V_low=float(f["V_low"]), dino=float(f["dino"]), clip=float(f["clip"]), p_target=float(f["p_target"]), top1=float(f["top1"])))
    write_csv(a / "mech_summary.csv", rows)
    return {r["condition"]: r for r in rows}


def fig_dose(s, out):
    """m1: six summary metrics against damping strength; mixes are single points at lambda = 1, w = 1 a dotted reference line."""
    keys =[("S_f_cond_late", "S(f_cond), t ≥ 0.3"), ("late_block_eff_rank", "effective rank, blocks 13–23, t ≥ 0.3"),
            ("late_block_signal_S", "class signal shared across seeds (S), blocks 13–23"), ("V_low", "final layout diversity (low-band V)"),
            ("dino", "final DINOv2 diversity"), ("p_target", "fidelity: classifier p(target)")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), dpi=170, facecolor=SURFACE)
    for ax, (k, label) in zip(axes.ravel(), keys):
        for chain, color, name in ((SCORE, "#2a78d6", "score λ"), (COND, "#eb6834", "condition λ")):
            ax.plot([STRENGTH[c] for c in chain], [s[c][k] for c in chain], color=color, lw=2, marker="o", ms=5, mec=SURFACE, label=name)
        for c, marker in zip(MIX, ("^", "v")):
            ax.scatter([1.0], [s[c][k]], color=MIXC[c], marker=marker, s=50, zorder=4, label=c.replace("+", " + "))
        ax.axhline(s["w1"][k], color=INK, lw=1, ls=":")
        style(ax, label, "damping strength λ")
        ax.set_xlim(-0.05, 2.1)
        ax.set_title(label, loc="left", color=INK, fontsize=9.5)
        ax.set_ylabel("")
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    axes[0, 0].text(2.05, s["w1"]["S_f_cond_late"], "w = 1", color=INK, fontsize=7.5, ha="right", va="bottom")
    header(fig, DOSE_TITLE, "Held-out seeds 16–31, 8 classes. λ = 0 is plain w = 4; dotted line is w = 1. Mixes use score λ = 1 plus condition λ shown.", 0.955)
    save(fig, out, "m1_dose_response.png", (0, 0, 1, 0.925))


def fig_decomposition(a, out):
    """m2: per block, how much of each intervention's change to the class signal is direct and how much comes through x_t."""
    p = read(a / "mech_patch.csv")
    act = {k: v[p["site"] != "delta"] for k, v in p.items()}  # block rows only; the Delta rows are not plotted here
    conditions = ["score_1", "cond_0.5", "score_1+cond_0.5"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharey="row", dpi=170, facecolor=SURFACE)
    for col, c in enumerate(conditions):
        d = select(act, condition=c)
        for row, (window, label) in enumerate(((d["t"] < 0.3, "during the window (t < 0.3)"), (d["t"] >= 0.3, "after the window (t ≥ 0.3)"))):
            ax = axes[row, col]
            blocks = np.arange(24)
            for key, color, name in (("direct_signal", "#eb6834", "direct"), ("trajectory_signal", "#2a78d6", "through x_t"), ("total_signal", INK, "total")):
                y = [d[key][window & (d["block"] == b)].astype(float).mean() for b in blocks]
                ax.plot(blocks, y, color=color, lw=2 if key != "total_signal" else 1.2, ls="-" if key != "total_signal" else ":", label=name)
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID)
            ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{c.replace('+', ' + ')}: {label}", loc="left", color=INK, fontsize=9.5)
            if row == 1:
                ax.set_xlabel("block", color=MUTED, fontsize=9.5)
            if col == 0:
                ax.set_ylabel("change in class signal\n(fraction of its w=4 size)", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8)
    header(fig, DECOMP_TITLE, "Class signal = pooled conditional minus unconditional activation. Direct: intervention on, same x_t. Through x_t: plain model on the intervened trajectory.", 0.955)
    save(fig, out, "m2_direct_vs_trajectory.png", (0, 0, 1, 0.92))


def block_maps(a, out, rows, title, subtitle, name):
    """2-D (time x block) maps of each row's metric for three conditions, relative to w=4. rows: (key, label, compare)."""
    b = read(a / "mech_blocks.csv")
    ref = select(b, condition="w4")
    t = np.unique(ref["t"])
    # (24 blocks, T) array of column k, looked up one (block, step) cell at a time.
    grid = lambda d, k: np.array([[d[k][(d["block"] == blk) & (d["step"] == s)][0] for s in range(len(t))] for blk in range(24)])
    conditions = ["score_1", "cond_0.5", "score_1+cond_0.5"]
    single = len(rows) == 1
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 4.3 if single else 3.5 * len(rows)), sharex=True, sharey=True, dpi=170, facecolor=SURFACE, squeeze=False)
    fig.subplots_adjust(top=0.8 if single else 0.88, bottom=0.14 if single else 0.08, left=0.06, right=0.9)  # before the colorbars, so they match the panels
    for row, (k, label, compare) in enumerate(rows):
        base = grid(ref, k)
        maps = [compare(grid(select(b, condition=c), k), base) for c in conditions]
        limit = max(np.nanmax(np.abs(m)) for m in maps)  # one symmetric color scale per row, so the three conditions compare directly
        for col, (c, m) in enumerate(zip(conditions, maps)):
            ax = axes[row, col]
            im = ax.imshow(m, aspect="auto", origin="lower", extent=(t[0], t[-1], -0.5, 23.5), cmap=DIVERGING,  # blue = higher than w=4, red = lower, in every row
                           norm=TwoSlopeNorm(0, -limit, limit), interpolation="nearest")
            ax.axvline(0.3, color=INK, lw=0.8, ls=":")
            ax.tick_params(colors=MUTED, length=0, labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{c.replace('+', ' + ')}: {label}", loc="left", color=INK, fontsize=9)
            if col == 0:
                ax.set_ylabel("block", color=MUTED, fontsize=9)
            if row == len(rows) - 1:
                ax.set_xlabel("t  (0 = noise, 1 = data)", color=MUTED, fontsize=9)
        bar = fig.colorbar(im, ax=axes[row].tolist(), fraction=0.015, pad=0.01)
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=MUTED, length=0, labelsize=8)
    header(fig, title, subtitle, 0.93 if single else 0.958)
    fig.savefig(out / name, facecolor=SURFACE)
    plt.close(fig)


# How a condition's map is compared with w=4's: a log2 ratio for V (a spread, where halving and doubling should look alike),
# a plain difference for quantities that already live on a fixed scale (S in [0, 1], effective rank in [1, 15]).
LOG_RATIO = lambda m, base: np.log2(m / base)
DIFF = lambda m, base: m - base


def fig_blocks(a, out):
    """m3: cross-seed spread V (top row) and class-signal sharing S (bottom row), relative to w=4."""
    block_maps(a, out, (("V", "scale-free spread V, log2 vs w=4", LOG_RATIO), ("signal_S", "class-signal S, minus w=4", DIFF)), BLOCKS_TITLE,
               "Pooled conditional features, 16 seeds per class. Blue = higher than w=4, red = lower. Top: cross-seed spread. Bottom: how much of the class signal seeds share.",
               "m3_block_maps.png")


def fig_block_rank(a, out):
    """m3b: effective rank of the 16 seeds per class, minus w=4's, by block and time."""
    block_maps(a, out, (("eff_rank", "effective rank, minus w=4", DIFF),), RANK_TITLE,
               "Pooled conditional features. Effective rank of the 16 seeds per class (max 15), averaged over 8 classes. Blue = higher than w=4, red = lower.",
               "m3b_block_rank.png")


def fig_curves(a, out):
    """m4: four guidance statistics over time; top row the score family, bottom row the condition family and mixes."""
    cv = read(a / "mech_curves.csv")
    keys = (("S_f_cond", "S(f_cond)"), ("S_delta", "S(Δ)"), ("ratio", "‖Δ‖ / ‖f_uncond‖"), ("V_x_hat", "cross-seed V of x̂"))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), sharex=True, dpi=170, facecolor=SURFACE)
    for row, family in enumerate((["w1", "w4", "score_0.5", "score_1", "score_2"], ["w1", "w4", "cond_0.25", "cond_0.5", "cond_1", *MIX])):
        for col, (k, label) in enumerate(keys):
            ax = axes[row, col]
            for c in family:
                d = select(cv, condition=c)
                ax.plot(d["t"], d[k], color=COLORS[c], lw=1.8 if c not in ("w1", "w4") else 1.3, ls="--" if c == "w1" else "-", label=c.replace("+", " + "))
            ax.axvline(0.3, color=MUTED, lw=0.8, ls=":")
            style(ax, label if col == 0 else None, "t" if row == 1 else None)
            ax.set_title(label, loc="left", color=INK, fontsize=9.5)
        axes[row, 0].legend(frameon=False, fontsize=7, loc="upper left")
    header(fig, CURVES_TITLE, "Top row: score family. Bottom row: condition family and mixes. Dotted vertical line: end of the intervention window.", 0.958)
    save(fig, out, "m4_curves.png", (0, 0, 1, 0.93))


# Default figure titles. --titles replaces the first four, so finding-style titles can be set after looking at the results.
DOSE_TITLE = "Dose response of score and condition damping"
DECOMP_TITLE = "Where each intervention changes the class signal inside the network"
BLOCKS_TITLE = "Internal changes relative to w = 4, by block and time"
CURVES_TITLE = "Guidance statistics over time for every condition"
RANK_TITLE = "All three raise effective rank, most in the later blocks; the combination raises it most"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--titles", nargs=4, help="finding-style titles for figures m1-m4")
    a = p.parse_args()
    analysis = Path(a.run) / "analysis"
    out = analysis / "figures"
    out.mkdir(exist_ok=True)
    global DOSE_TITLE, DECOMP_TITLE, BLOCKS_TITLE, CURVES_TITLE  # the fig_* functions read these module-level names
    if a.titles:
        DOSE_TITLE, DECOMP_TITLE, BLOCKS_TITLE, CURVES_TITLE = a.titles
    s = summary(analysis)
    fig_dose(s, out)
    fig_decomposition(analysis, out)
    fig_blocks(analysis, out)
    fig_block_rank(analysis, out)
    fig_curves(analysis, out)


if __name__ == "__main__":
    main()
