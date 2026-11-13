"""Aggregate a guidance_delta.py run: per-trajectory CSV, per-timestep summary CSV, and the plot.

Means, SDs and percentiles are over all seed x class trajectories at each timestep.
usage: guidance_plot.py --run <dir> [--title <text>]
"""
import argparse, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colors from the dataviz reference palette: series 1, then primary text, secondary text and a recessive grid.
SERIES, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e0"


def load(run):
    man = json.loads((run / "manifest.json").read_text())
    data = [np.load(run / f"seed_{s}.npz") for s in man["seeds"]]
    stack = lambda key: np.stack([d[key] for d in data], 1)  # (T, seeds, classes)
    return man, data[0]["t"], {k: stack(k) for k in ("norm", "cos", "norm_guided", "cos_guided")}


def write_tables(run, man, t, m):
    with open(run / "metrics_per_trajectory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("seed label label_name step t delta_norm cos_prev delta_norm_guided cos_prev_guided".split())
        for si, seed in enumerate(man["seeds"]):
            for ci, label in enumerate(man["labels"]):
                for k, tk in enumerate(t):
                    # cos row k-1 belongs to step k (see metrics in guidance_delta.py). Step 0 has no predecessor, so its cells stay blank.
                    c = (m["cos"][k - 1, si, ci], m["cos_guided"][k - 1, si, ci]) if k else ("", "")
                    w.writerow([seed, label, man["label_names"][ci], k, f"{tk:.6f}", m["norm"][k, si, ci], c[0],
                                m["norm_guided"][k, si, ci], c[1]])
    with open(run / "metrics_by_timestep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("step t n norm_mean norm_sd norm_p10 norm_p90 cos_mean cos_sd cos_min cos_p10 cos_p90".split())
        for k, tk in enumerate(t):
            n = m["norm"][k].ravel()
            c = m["cos"][k - 1].ravel() if k else None
            w.writerow([k, f"{tk:.6f}", n.size, n.mean(), n.std(ddof=1), *np.percentile(n, [10, 90])]
                       + ([c.mean(), c.std(ddof=1), c.min(), *np.percentile(c, [10, 90])] if k else [""] * 5))


def panel(ax, t, values, ylabel):
    mean = values.mean(1)
    # Percentiles rather than +/-1 SD: cosine is capped at 1 and skewed, and mean + SD goes past 1 near t = 0.08.
    low, high = np.percentile(values, [10, 90], axis=1)
    ax.fill_between(t, low, high, color=SERIES, alpha=0.18, linewidth=0)
    ax.plot(t, mean, color=SERIES, linewidth=2)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot(run, man, t, m, title):
    # Merge the seed and class axes, so each timestep has one column per trajectory (64 here).
    flat = lambda a: a.reshape(a.shape[0], -1)
    n = flat(m["norm"]).shape[1]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, dpi=180, facecolor="#fcfcfb")
    for ax in (top, bottom):
        ax.set_facecolor("#fcfcfb")
    panel(top, t, flat(m["norm"]), r"$\|\Delta_t\|_2$")
    # t[1:] because cos[k] belongs to t[k+1].
    panel(bottom, t[1:], flat(m["cos"]), r"$\cos(\Delta_t, \Delta_{t-1})$")
    bottom.set_xlabel("t  (0 = noise, 1 = data)", color=MUTED)
    bottom.set_xlim(0, 1)
    fig.suptitle(title, x=0.075, ha="left", color=INK, fontsize=12.5, fontweight="bold")
    fig.text(0.075, 0.895, f"{man['model']}, velocity parameterization, CFG {man['cfg_scale']:g}, Euler with "
             f"{man['model_calls']} model calls.\nLine: mean of {n} trajectories "
             f"({len(man['seeds'])} seeds × {len(man['labels'])} classes). Band: 10th–90th percentile.", color=MUTED, fontsize=8.5)
    # rect reserves the top 10% for the manually placed title and subtitle.
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(run / "guidance_delta.png", facecolor=fig.get_facecolor())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--title", default="Guidance vector norm and direction stability over sampling time")
    a = p.parse_args()
    run = Path(a.run)
    man, t, m = load(run)
    write_tables(run, man, t, m)
    plot(run, man, t, m, a.title)


if __name__ == "__main__":
    main()
