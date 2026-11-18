"""Plot report Figures 1 and 2 from the tables of guidance_analysis.py into <run>/analysis/figures/.

  02_alignment_within_class.png   Figure 1: within-class agreement of Delta, f_cond and f_uncond over time (alignment_shared.csv)
  03_effective_rank_pooled.png    Figure 2: effective rank by block and time at w = 1, w = 4, and their difference (subspace_pooled.csv)

The helpers here (palette, read, select, header) are shared with guidance_mech_figures.py so every figure looks the same.
usage: guidance_figures.py --run <dir>
"""
import argparse, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # render to files without a display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

# Reference palette: blue/orange categorical pair, blue sequential ramp, blue<->red diverging pair with a gray midpoint.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
W_STYLE = {4: dict(color=BLUE, linestyle="-"), 1: dict(color=ORANGE, linestyle="--")}
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#104281"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
W_LEGEND = [Line2D([], [], lw=2, label=f"w = {w}", **s) for w, s in W_STYLE.items()]
T_LABEL = "t  (0 = noise, 1 = data)"


def read(path):
    """Load a CSV as {column: numpy array}, converting columns to float wherever every cell is a number or blank (NaN)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # A column is numeric only if every non-blank cell is. Text columns, like condition names, stay as text.
    table = {}
    for k in rows[0]:
        cells = [r[k] for r in rows]
        numeric = all(c == "" or _number(c) for c in cells)
        table[k] = np.array([np.nan if c == "" else float(c) for c in cells]) if numeric else np.array(cells)
    return table


def _number(text):
    """True if the string parses as a float."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def select(table, **conditions):
    """Rows of a read() table where every given column equals its value, e.g. select(t, w=4, branch="cond")."""
    mask = np.ones(len(next(iter(table.values()))), bool)
    # Text columns compare as strings, numeric columns as floats, so select(t, w=4) and select(t, w="4") both work.
    for key, value in conditions.items():
        mask &= table[key] == (value if isinstance(table[key].flat[0], str) else float(value))
    return {k: v[mask] for k, v in table.items()}


def style(ax, ylabel=None, xlabel=None):
    """House style for a line plot: light background, horizontal grid only, no box, muted labels, x axis fixed to t in [0, 1]."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5)
    ax.set_xlim(0, 1)


def header(fig, title, subtitle, top):
    """Left-aligned bold title stating the finding, with a muted subtitle (the method) placed at height `top`."""
    fig.suptitle(title, x=0.06, y=0.985, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.text(0.06, top, subtitle, color=MUTED, fontsize=8.5, va="top")


def save(fig, out, name, rect):
    """Lay out the panels inside `rect` (leaving room for the header), write the PNG and free the figure's memory."""
    fig.tight_layout(rect=rect)
    fig.savefig(out / name, facecolor=SURFACE)
    plt.close(fig)


def w_lines(ax, table, y, low=None, high=None):
    """Plot column y against t once per w (blue solid w=4, orange dashed w=1), with an optional low-high band."""
    for w, s in W_STYLE.items():
        d = select(table, w=w)
        ok = np.isfinite(d[y].astype(float))
        if low:
            ax.fill_between(d["t"][ok], d[low][ok].astype(float), d[high][ok].astype(float), color=s["color"], alpha=0.14, linewidth=0)
        ax.plot(d["t"][ok], d[y][ok].astype(float), lw=2, **s)


def fig_alignment(a, out):
    """Figure 1: mean pairwise cosine of Delta, and the shared proportion S of Delta, f_cond and f_uncond, within each class."""
    d = read(a / "alignment_shared.csv")
    floor = 1 / 16  # S of 16 independent zero-mean vectors: the "nothing shared" reference line
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, dpi=170, facecolor=SURFACE)
    # (column, y label, reference line): the cosine panel's reference is 0, the S panels' is 1/16.
    panels = [("pairwise_cos_delta_class", r"Mean pairwise $\cos(\Delta^{(i)}, \Delta^{(j)})$", 0),
              ("S_delta_class", r"$S(\Delta)$", floor), ("S_f_cond_class", r"$S(f_{cond})$", floor),
              ("S_f_uncond_class", r"$S(f_{uncond})$", floor)]
    for ax, (key, label, reference) in zip(axes.ravel(), panels):
        # Line: mean over the 8 classes. Band: the range across classes.
        w_lines(ax, d, key, f"{key}_min", f"{key}_max")
        ax.axhline(reference, color=MUTED, lw=1, ls=":")
        style(ax, label)
        low, high = ax.get_ylim()
        ax.set_ylim(low - 0.06 * (high - low), high)  # room under the reference line for its label
        ax.text(0.99, reference, f"{'no shared part ≈ 1/16' if reference else 'independent = 0'} ", color=MUTED, fontsize=7.5, ha="right", va="top")
    for ax in axes[1]:
        ax.set_xlabel(T_LABEL, color=MUTED, fontsize=9.5)
    axes[0, 0].legend(handles=W_LEGEND, frameon=False, loc="upper right", fontsize=9)
    header(fig, "Seeds of one class share a guidance direction only early on",
           "i = 16 seeds within a class, averaged over classes. Band: range across the 8 classes.", 0.955)
    save(fig, out, "02_alignment_within_class.png", (0, 0, 1, 0.925))


def heat(ax, values, t, cmap, norm=None, label=""):
    """Heat map with time on x and block on y, plus a colorbar. values: (24 blocks, T)."""
    im = ax.imshow(values, aspect="auto", origin="lower", extent=(t[0], t[-1], -0.5, values.shape[0] - 0.5), cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([0, 4, 8, 12, 16, 20, 23])
    bar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, length=0, labelsize=8)
    bar.set_label(label, color=MUTED, fontsize=8.5)
    return im


def grid(d, key):
    """(24 layers, T) grid of one column of subspace_pooled.csv."""
    steps = sorted(set(d["step"].astype(int)))
    g = np.full((24, len(steps)), np.nan)
    g[d["layer"].astype(int), d["step"].astype(int)] = d[key].astype(float)
    return g


def fig_rank(a, out):
    """Figure 2: effective rank of the conditional pooled features at w=1, at w=4, and their difference, per block and time."""
    p = select(read(a / "subspace_pooled.csv"), branch="cond")
    t = np.unique(p["t"].astype(float))
    r1, r4 = grid(p, "eff_rank_w1"), grid(p, "eff_rank_w4")
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, dpi=170, facecolor=SURFACE)
    # The w=1 and w=4 maps share one color scale; the difference uses a symmetric diverging scale centered on 0.
    vmax = max(r1.max(), r4.max())
    vmin = min(r1.min(), r4.min())
    heat(axes[0], r1, t, SEQUENTIAL, plt.Normalize(vmin, vmax), "effective rank")
    heat(axes[1], r4, t, SEQUENTIAL, plt.Normalize(vmin, vmax), "effective rank")
    diff = r4 - r1
    limit = np.abs(diff).max()
    heat(axes[2], diff, t, DIVERGING, TwoSlopeNorm(0, -limit, limit), "w=4 minus w=1")
    for ax, name in zip(axes, ("w = 1", "w = 4", "Difference (red: w = 4 has lower rank)")):
        ax.set_title(name, loc="left", color=INK, fontsize=10)
        ax.set_ylabel("block", color=MUTED, fontsize=9.5)
    axes[-1].set_xlabel(T_LABEL, color=MUTED, fontsize=9.5)
    header(fig, "Guidance lowers effective rank in the later blocks and slightly raises it in the first few",
           "Pooled conditional features. Effective rank from 16 seeds per class, averaged over 8 classes; the maximum possible is 15.", 0.958)
    save(fig, out, "03_effective_rank_pooled.png", (0, 0, 1, 0.935))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    a = Path(args.run) / "analysis"
    out = a / "figures"
    out.mkdir(exist_ok=True)
    fig_alignment(a, out)
    fig_rank(a, out)


if __name__ == "__main__":
    main()
