"""Plot the tables from guidance_analysis.py into <run>/analysis/figures/.

usage: guidance_figures.py --run <dir> [--only 1 2 ...]
"""
import argparse, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

# Reference palette: categorical slots 1-3 (validated), blue sequential ramp, blue<->red diverging pair with a gray midpoint.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
W_STYLE = {4: dict(color=BLUE, linestyle="-"), 1: dict(color=ORANGE, linestyle="--")}
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#104281"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
W_LEGEND = [Line2D([], [], lw=2, label=f"w = {w}", **s) for w, s in W_STYLE.items()]
T_LABEL = "t  (0 = noise, 1 = data)"


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # A column is numeric only if every non-blank cell is. Mixed columns, like channel ("all", "0", ...), stay as text.
    table = {}
    for k in rows[0]:
        cells = [r[k] for r in rows]
        numeric = all(c == "" or _number(c) for c in cells)
        table[k] = np.array([np.nan if c == "" else float(c) for c in cells]) if numeric else np.array(cells)
    return table


def _number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def select(table, **conditions):
    mask = np.ones(len(next(iter(table.values()))), bool)
    for key, value in conditions.items():
        mask &= table[key] == (value if isinstance(table[key].flat[0], str) else float(value))
    return {k: v[mask] for k, v in table.items()}


def style(ax, ylabel=None, xlabel=None):
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
    fig.suptitle(title, x=0.06, y=0.985, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.text(0.06, top, subtitle, color=MUTED, fontsize=8.5, va="top")


def save(fig, out, name, rect):
    fig.tight_layout(rect=rect)
    fig.savefig(out / name, facecolor=SURFACE)
    plt.close(fig)


def w_lines(ax, table, y, low=None, high=None, drop_nan=True):
    for w, s in W_STYLE.items():
        d = select(table, w=w)
        ok = np.isfinite(d[y].astype(float))
        if low:
            ax.fill_between(d["t"][ok], d[low][ok].astype(float), d[high][ok].astype(float), color=s["color"], alpha=0.14, linewidth=0)
        ax.plot(d["t"][ok], d[y][ok].astype(float), lw=2, **s)


def fig_magnitude(a, out):
    d = read(a / "magnitude_direction.csv")
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.4), sharex=True, dpi=170, facecolor=SURFACE)
    for ax, key, label in zip(axes, ("norm", "ratio", "cos"),
                              (r"$\|\Delta_t\|_2$", r"$\|\Delta_t\|\,/\,\|f_{uncond}\|$", r"$\cos(\Delta_t, \Delta_{t-1})$")):
        w_lines(ax, d, f"{key}_mean", f"{key}_p10", f"{key}_p90")
        style(ax, label)
    axes[-1].set_xlabel(T_LABEL, color=MUTED, fontsize=9.5)
    axes[0].legend(handles=W_LEGEND, frameon=False, loc="upper right", fontsize=9)
    header(fig, "Guidance is largest early, and w = 4 turns it most sharply near t = 0.1",
           "Line: mean of 128 trajectories (16 seeds × 8 classes). Band: 10th–90th percentile.", 0.955)
    save(fig, out, "01_magnitude_direction.png", (0, 0, 1, 0.93))


def fig_alignment(a, out, scope, name, title, floor, floor_label, band_note):
    d = read(a / "alignment_shared.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, dpi=170, facecolor=SURFACE)
    panels = [(f"pairwise_cos_delta_{scope}", r"Mean pairwise $\cos(\Delta^{(i)}, \Delta^{(j)})$", 0),
              (f"S_delta_{scope}", r"$S(\Delta)$", floor), (f"S_f_cond_{scope}", r"$S(f_{cond})$", floor),
              (f"S_f_uncond_{scope}", r"$S(f_{uncond})$", floor)]
    for ax, (key, label, reference) in zip(axes.ravel(), panels):
        has_band = f"{key}_min" in d
        w_lines(ax, d, key, f"{key}_min" if has_band else None, f"{key}_max" if has_band else None)
        ax.axhline(reference, color=MUTED, lw=1, ls=":")
        style(ax, label)
        low, high = ax.get_ylim()
        ax.set_ylim(low - 0.06 * (high - low), high)  # room under the reference line for its label
        ax.text(0.99, reference, f"{floor_label if reference else 'independent = 0'} ", color=MUTED, fontsize=7.5, ha="right", va="top")
    for ax in axes[1]:
        ax.set_xlabel(T_LABEL, color=MUTED, fontsize=9.5)
    axes[0, 0].legend(handles=W_LEGEND, frameon=False, loc="upper right", fontsize=9)
    header(fig, title, band_note, 0.955)
    save(fig, out, name, (0, 0, 1, 0.925))


def heat(ax, values, t, cmap, norm=None, label=""):
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
    """(24 layers, T) grid from a pooled subspace table restricted to one branch."""
    steps = sorted(set(d["step"].astype(int)))
    g = np.full((24, len(steps)), np.nan)
    g[d["layer"].astype(int), d["step"].astype(int)] = d[key].astype(float)
    return g


def fig_rank(a, out):
    p = select(read(a / "subspace_pooled.csv"), branch="cond")
    t = np.unique(p["t"].astype(float))
    r1, r4 = grid(p, "eff_rank_w1"), grid(p, "eff_rank_w4")
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, dpi=170, facecolor=SURFACE)
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


def fig_subspace(a, out):
    p = select(read(a / "subspace_pooled.csv"), branch="cond")
    t = np.unique(p["t"].astype(float))
    fig = plt.figure(figsize=(11, 9.4), dpi=170, facecolor=SURFACE)
    spec = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    top_left, top_right = fig.add_subplot(spec[0, :2]), fig.add_subplot(spec[0, 2])
    ratio = np.log2(grid(p, "total_variance_ratio"))
    limit = np.nanmax(np.abs(ratio))
    im = heat(top_left, ratio, t, DIVERGING, TwoSlopeNorm(0, -limit, limit), "total variance, w=4 / w=1")
    ticks = [x for x in (0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.4) if abs(np.log2(x)) <= limit]
    im.colorbar.set_ticks(np.log2(ticks), labels=[str(x) for x in ticks])
    top_left.set_title("Red: guidance shrinks cross-seed variance. Blue: it grows.", loc="left", color=INK, fontsize=10)
    top_left.set_ylabel("block", color=MUTED, fontsize=9.5)
    top_left.set_xlabel(T_LABEL, color=MUTED, fontsize=9.5)

    tok = select(read(a / "subspace_tokens.csv"), branch="cond")
    for layer, color, ls in ((4, BLUE, "-"), (12, ORANGE, "--"), (22, AQUA, ":")):
        pooled = select(p, layer=layer)
        tokens = select(tok, layer=layer)
        top_right.plot(pooled["t"].astype(float), pooled["total_variance_ratio"].astype(float), color=color, ls=ls, lw=2)
        top_right.plot(tokens["t"].astype(float), tokens["total_variance_ratio"].astype(float), color=color, ls=ls, lw=1.2, marker="o", ms=4)
    top_right.axhline(1, color=MUTED, lw=1, ls=":")
    style(top_right, "total variance, w=4 / w=1", T_LABEL)
    top_right.set_title("Pooled (no marker) vs. tokens (dots)", loc="left", color=INK, fontsize=10)
    top_right.legend(handles=[Line2D([], [], color=c, ls=l, lw=2, label=f"block {b}") for b, c, l in
                              ((4, BLUE, "-"), (12, ORANGE, "--"), (22, AQUA, ":"))], frameon=False, fontsize=8.5)

    splits = read(a / "subspace_rotation_splits.csv")
    for i, layer in enumerate((4, 12, 22)):
        ax = fig.add_subplot(spec[1, i])
        d = select(splits, layer=layer)
        ax.fill_between(d["t"], d["cross_w"], d["same_w"], color=BLUE, alpha=0.14, linewidth=0)
        ax.plot(d["t"], d["same_w"], color=INK, lw=2, label="same w, different seeds (ceiling)")
        ax.plot(d["t"], d["cross_w"], color=BLUE, lw=2, label="w=1 vs w=4, different seeds")
        ax.plot(d["t"], d["paired"], color=ORANGE, lw=1.5, ls="--", label="w=1 vs w=4, same seeds")
        ax.set_ylim(0, 1.03)
        style(ax, "subspace overlap (mean cos²)" if i == 0 else None, T_LABEL)
        ax.set_title(f"Block {layer}", loc="left", color=INK, fontsize=10)
        if i == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    header(fig, "Pooled variation mostly shrinks in place; the later blocks also rotate a little",
           "Pooled conditional features, class-centered, top 5 directions, 8 seeds per estimate, mean of 20 random splits. Rotation = blue below black.", 0.957)
    save(fig, out, "04_subspace_rotation_vs_shrink.png", (0, 0, 1, 0.935))


def fig_shrink(a, out):
    p = select(read(a / "subspace_pooled.csv"), branch="cond")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), sharey=True, dpi=170, facecolor=SURFACE)
    colors = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4"]
    for ax, layer in zip(axes, (4, 12, 22)):
        d = select(p, layer=layer)
        t = d["t"].astype(float)
        for k, color in enumerate(colors, 1):
            y = d[f"shrink_pc{k}"].astype(float)
            # Smoothing over 5 steps: each point is a cross-fitted variance ratio from 64 rows, so single steps are noisy.
            ok = np.isfinite(y)
            kernel = np.ones(5) / 5
            ax.plot(t[ok][2:-2], np.convolve(y[ok], kernel, "valid"), color=color, lw=1.6, label=f"PC{k}")
        ax.axhline(1, color=MUTED, lw=1, ls=":")
        ax.set_title(f"Block {layer}", loc="left", color=INK, fontsize=10)
        style(ax, "variance along direction, w=4 / w=1" if layer == 4 else None, T_LABEL)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    header(fig, "Shrinkage is uneven: most w = 1 directions lose variance under guidance, a few do not",
           "Directions fit on half the seeds at w=1, variance measured on the other half at both w, 5-step moving average.", 0.905)
    save(fig, out, "05_principal_direction_shrinkage.png", (0, 0, 1, 0.86))


def fig_frequency(a, out):
    e = read(a / "spectrum_energy.csv")
    all_ch = {k: v for k, v in select(e, channel="all").items()}
    fig = plt.figure(figsize=(11, 8.6), dpi=170, facecolor=SURFACE)
    spec = fig.add_gridspec(2, 4)
    energy = fig.add_subplot(spec[0, :2])
    fraction = fig.add_subplot(spec[0, 2:])
    for w, s in W_STYLE.items():
        d = select(all_ch, w=w)
        t = d["t"].astype(float)
        energy.plot(t, d["low_8"].astype(float), lw=2, **s)
        energy.plot(t, d["high_8"].astype(float), lw=1.2, **s)
    energy.set_yscale("log")
    style(energy, r"energy in $\Delta_t$ (cutoff 8 cycles)", T_LABEL)
    energy.legend(handles=W_LEGEND + [Line2D([], [], color=MUTED, lw=2.4, label="below cutoff"), Line2D([], [], color=MUTED, lw=1, label="above cutoff")],
                  frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    for w, s in W_STYLE.items():
        d = select(all_ch, w=w)
        t = d["t"].astype(float)
        fraction.plot(t, d["high_fraction_8"].astype(float), lw=2, **s)
        fraction.plot(t, d["blur_high_fraction_s2"].astype(float), lw=1.2, alpha=0.75, **s)
    fraction.legend(handles=[Line2D([], [], color=MUTED, lw=2.4, label="FFT, cutoff 8"), Line2D([], [], color=MUTED, lw=1, label="Δ − blur(Δ), σ = 2")],
                    frameon=False, fontsize=8.5, loc="lower right")
    style(fraction, "high-frequency share of energy", T_LABEL)
    for ch in range(4):
        ax = fig.add_subplot(spec[1, ch])
        for w, s in W_STYLE.items():
            d = select(e, channel=str(ch), w=w)
            ax.plot(d["t"].astype(float), d["high_fraction_8"].astype(float), lw=1.8, **s)
        ax.set_ylim(0, 1)
        style(ax, "high-frequency share" if ch == 0 else None, T_LABEL)
        ax.set_title(f"Channel {ch}" + (" (not guided)" if ch == 3 else ""), loc="left", color=INK, fontsize=10)
    header(fig, "Guidance energy moves from low to high spatial frequencies as t rises",
           "Radial FFT of Δ_t. Cutoff is 8 cycles per 32-pixel latent, half of Nyquist. Means over 128 trajectories.", 0.955)
    save(fig, out, "06_frequency_content.png", (0, 0, 1, 0.925))


def fig_spread(a, out):
    d = read(a / "spread_over_time.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True, dpi=170, facecolor=SURFACE)
    for ax, key, label in zip(axes.ravel(), ("l2_latent", "cos_latent", "dino", "clip"),
                              ("L2 on latent x̂", "cosine distance on latent x̂", "DINOv2 cosine distance", "CLIP cosine distance")):
        s = select(d, metric=key)
        t = s["t"].astype(float)
        keep = s["step"].astype(int) < 49
        ax.fill_between(t[keep], s["ratio_class_min"][keep].astype(float), s["ratio_class_max"][keep].astype(float), color=BLUE, alpha=0.14, linewidth=0)
        ax.plot(t[keep], s["ratio_of_means"][keep].astype(float), color=BLUE, lw=2)
        ax.axhline(1, color=MUTED, lw=1, ls=":")
        style(ax, "spread(w=4) / spread(w=1)")
        ax.set_title(label, loc="left", color=INK, fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel(T_LABEL, color=MUTED, fontsize=9.5)
    header(fig, "Guidance more than halves cross-seed spread in feature space, starting around t = 0.2",
           "Mean pairwise distance across 16 seeds within each class. Line: ratio of class-averaged spreads. Band: range across 8 classes.", 0.958)
    save(fig, out, "07_spread_ratio.png", (0, 0, 1, 0.935))


def fig_final(a, out):
    d = read(a / "final_diversity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=170, facecolor=SURFACE)
    for ax, key, label in zip(axes, ("dino", "clip"), ("DINOv2", "CLIP")):
        s = select(d, metric=key)
        names = list(s["class_name"])
        order = np.argsort(s["spread_w1"].astype(float))
        for row, i in enumerate(order):
            x1, x4 = float(s["spread_w1"][i]), float(s["spread_w4"][i])
            ax.plot([x4, x1], [row, row], color=GRID, lw=2, zorder=1)
            ax.scatter([x1], [row], color=ORANGE, marker="s", s=44, zorder=2, edgecolors=SURFACE, linewidths=1.5)
            ax.scatter([x4], [row], color=BLUE, marker="o", s=52, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.set_yticks(range(len(names)), [names[i].replace(" (mean of classes)", " classes") for i in order])
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlabel(f"{label}: mean pairwise cosine distance across seeds", color=MUTED, fontsize=9.5)
    axes[1].legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="", label="w = 4"), Line2D([], [], color=ORANGE, marker="s", ls="", label="w = 1")],
                   frameon=False, fontsize=9, loc="lower right")
    header(fig, "Final images are less diverse at w = 4 in every class", "16 seeds per class, same noise at both w. ALL = mean of the 8 classes.", 0.92)
    save(fig, out, "08_final_image_diversity.png", (0, 0, 1, 0.88))


def fig_scale(a, out):
    pooled, tokens = read(a / "scale_free_pooled.csv"), read(a / "scale_free_tokens.csv")
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), sharex=True, dpi=170, facecolor=SURFACE)
    blocks = ((4, BLUE, "-"), (12, ORANGE, "--"), (22, AQUA, ":"))
    for layer, color, ls in blocks:
        p, k = select(pooled, layer=layer), select(tokens, layer=layer)
        axes[0, 0].plot(p["t"], p["norm_ratio"], color=color, ls=ls, lw=2)
        axes[1, 0].plot(k["t"], k["norm_ratio"], color=color, ls=ls, lw=1.6, marker="o", ms=4)
        axes[0, 1].plot(p["t"], p["raw_ratio"], color=color, ls=ls, lw=1.2, alpha=0.55)
        axes[0, 1].plot(p["t"], p["V_ratio"], color=color, ls=ls, lw=2.2)
        axes[1, 1].plot(k["t"], k["raw_ratio"], color=color, ls=ls, lw=1.2, alpha=0.55)
        axes[1, 1].plot(k["t"], k["V_ratio"], color=color, ls=ls, lw=2.2, marker="o", ms=4)
        axes[0, 2].plot(k["t"], k["pooled_share_w1"], color=color, ls=ls, lw=1.8, marker="o", ms=4)
        # V splits like the variance does, so V * (1 - pooled share) is the scale-free spatial part.
        spatial = k["V_w4"] * (1 - k["pooled_share_w4"]) / (k["V_w1"] * (1 - k["pooled_share_w1"]))
        axes[1, 2].plot(k["t"], spatial, color=color, ls=ls, lw=1.8, marker="o", ms=4)
    for ax, title, label in ((axes[0, 0], "Pooled activation norm", "norm, w=4 / w=1"), (axes[1, 0], "Per-token activation norm", "norm, w=4 / w=1"),
                             (axes[0, 1], "Pooled spread", "w=4 / w=1"), (axes[1, 1], "Token spread", "w=4 / w=1"),
                             (axes[0, 2], "Pooled share of token variance, w=1", "share"),
                             (axes[1, 2], "Spatial-pattern spread, scale-free", "w=4 / w=1")):
        style(ax, label)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        if ax is not axes[0, 2]:
            ax.axhline(1, color=MUTED, lw=1, ls=":")
    for ax in axes[1]:
        ax.set_xlabel(T_LABEL, color=MUTED, fontsize=9.5)
    axes[0, 0].legend(handles=[Line2D([], [], color=c, ls=l, lw=2, label=f"block {b}") for b, c, l in blocks], frameon=False, fontsize=8.5)
    axes[0, 1].legend(handles=[Line2D([], [], color=MUTED, lw=2.2, label="scale-free, V = 1 − S"), Line2D([], [], color=MUTED, lw=1.2, alpha=0.55, label="raw variance")],
                      frameon=False, fontsize=8.5)
    header(fig, "Pooled shrinkage is not a scale effect; token spread stays near 1 outside block 22",
           "Conditional branch, class-wise over 16 seeds. Token panels use the 10 logged timesteps.", 0.955)
    save(fig, out, "09_scale_free_spread.png", (0, 0, 1, 0.925))


COND_STYLE = {"w1": dict(color=ORANGE, ls="--", label="w = 1"), "w4": dict(color=BLUE, ls="-", label="w = 4"),
              "early": dict(color=AQUA, ls="-", label="w = 4 on [0, 0.3)"), "late": dict(color="#6250d6", ls=":", label="w = 4 on [0.3, 1]")}


def fig_window_curves(a, out):
    d = read(a / "window_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), dpi=170, facecolor=SURFACE)
    for ax, key, label in zip(axes, ("S_f_cond", "eff_rank", "V_pooled"), (r"$S(f_{cond})$", "effective rank, blocks 13–23", "V = 1 − S, pooled, blocks 13–23")):
        for name, s in COND_STYLE.items():
            c = select(d, condition=name)
            ax.plot(c["t"], c[key], lw=2, color=s["color"], ls=s["ls"], label=s["label"])
        ax.axvline(0.3, color=MUTED, lw=1, ls=":")
        style(ax, label, T_LABEL)
    axes[0].legend(frameon=False, fontsize=8.5)
    header(fig, WINDOW_TITLE, "Same 16 seeds and noise in every condition. Dotted vertical line: window boundary at t = 0.3.", 0.905)
    save(fig, out, "10_window_curves.png", (0, 0, 1, 0.86))


def fig_window_effects(a, out):
    d = read(a / "window_effects.csv")
    labels = {"S_f_cond": "S(f_cond), t ≥ 0.3", "S_f_uncond": "S(f_uncond), t ≥ 0.3", "eff_rank_late_blocks": "Effective rank, blocks 13–23",
              "V_pooled_late_blocks": "V pooled, blocks 13–23", "final_dino": "Final diversity, DINOv2", "final_clip": "Final diversity, CLIP",
              "final_latent_cos": "Final diversity, latent cosine", "V_final_low_band": "Final latent V, low band (≤ 4 cycles)",
              "V_final_high_band": "Final latent V, high band (> 8 cycles)"}
    keys = list(labels)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=170, facecolor=SURFACE)
    for row, key in enumerate(keys):
        i = list(d["metric"]).index(key)
        for name, shift, marker in (("early", 0.16, "o"), ("late", -0.16, "s")):
            color = COND_STYLE[name]["color"]
            ax.plot([d[f"{name}_ci_low"][i], d[f"{name}_ci_high"][i]], [row + shift] * 2, color=color, lw=2, solid_capstyle="round")
            ax.scatter([d[f"{name}_fraction"][i]], [row + shift], color=color, marker=marker, s=46, zorder=3, edgecolors=SURFACE, linewidths=1.2)
    ax.set_yticks(range(len(keys)), [labels[k] for k in keys])
    ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.axvline(1, color=MUTED, lw=1, ls=":")
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("share of the full w=4 effect reproduced  (0 = same as w=1, 1 = same as w=4)", color=MUTED, fontsize=9.5)
    ax.legend(handles=[Line2D([], [], color=COND_STYLE[n]["color"], marker=m, lw=2, label=COND_STYLE[n]["label"]) for n, m in (("early", "o"), ("late", "s"))],
              frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    header(fig, EFFECT_TITLE, "Points: all 16 seeds. Bars: 95% bootstrap interval over seeds, paired across conditions.", 0.925)
    save(fig, out, "11_window_effects.png", (0, 0, 1, 0.9))


WINDOW_TITLE = "S(f) and pooled spread follow the early window; rank mostly does"
EFFECT_TITLE = "Early guidance sets S(f) and layout, but either window cuts final diversity"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--only", type=int, nargs="+")
    args = p.parse_args()
    a = Path(args.run) / "analysis"
    out = a / "figures"
    out.mkdir(exist_ok=True)
    figures = {
        1: lambda: fig_magnitude(a, out),
        2: lambda: fig_alignment(a, out, "class", "02_alignment_within_class.png", "Seeds of one class share a guidance direction only early on",
                                 1 / 16, "no shared part ≈ 1/16", "i = 16 seeds within a class, averaged over classes. Band: range across the 8 classes."),
        3: lambda: fig_alignment(a, out, "all", "02b_alignment_all_trajectories.png", "Across all classes and seeds, little is shared in Δ",
                                 1 / 128, "no shared part ≈ 1/128", "i = all 128 trajectories treated as one population, so class differences count as variation."),
        4: lambda: fig_rank(a, out), 5: lambda: fig_subspace(a, out), 6: lambda: fig_shrink(a, out), 7: lambda: fig_frequency(a, out),
        8: lambda: fig_spread(a, out), 9: lambda: fig_final(a, out), 10: lambda: fig_scale(a, out),
        11: lambda: fig_window_curves(a, out), 12: lambda: fig_window_effects(a, out)}
    for key in args.only or figures:
        figures[key]()


if __name__ == "__main__":
    main()
