"""The diversity-fidelity frontier with realistic diversity (realistic_diversity.py) in place of mean pairwise DINO distance.

For every condition with seeds 16-47: precision (x) against realistic distinct images per 32 draws (Vendi over on-manifold samples,
y), both against the 1,268-image reference. "vs curve" is a condition's realistic Vendi minus the condition-damping curve
(w4, cond_0.5, cond_0.75, cond_1, piecewise linear in precision) at its own precision: positive = more realistic diversity than
condition damping at equal fidelity. Outside the curve's precision range the nearest end is used. Intervals: paired half-samples
(the same 16-seed halves for every condition), centered on the full-sample value and scaled by 1/sqrt(2).
usage: realistic_frontier.py   -> stdout and outputs/guidance_delta/figures/frontier_realistic.png
"""
import sys
sys.path.insert(0, "bench")
import numpy as np
import realistic_diversity as rd

CURVE = ["w4", "cond_0.5", "cond_0.75", "cond_1"]
P, V = rd.KEYS.index("prec"), rd.KEYS.index("rvendi")


def vs_curve(prec, rv):
    """prec, rv: {condition: value}. Realistic Vendi minus the condition-damping curve at the same precision."""
    xs = np.array([prec[c] for c in CURVE]); ys = np.array([rv[c] for c in CURVE]); o = np.argsort(xs)
    return {c: rv[c] - np.interp(prec[c], xs[o], ys[o]) for c in prec}


point = vs_curve({c: rd.full[c]["prec"] for c in rd.names}, {c: rd.full[c]["rvendi"] for c in rd.names})
halves = [vs_curve({c: rd.H[c][h, P] for c in rd.names}, {c: rd.H[c][h, V] for c in rd.names}) for h in range(len(rd.halves))]
print(f"{'condition':42} {'precision':>9} {'realistic Vendi/32':>18} {'vs curve [95%]':>24}")
for c in sorted(rd.names, key=lambda c: -point[c]):
    h = np.array([x[c] for x in halves]); dev = (h - h.mean()) / np.sqrt(2) + point[c]
    flag = "" if c not in CURVE else "  (curve)"
    print(f"{c:42} {rd.full[c]['prec']:9.2f} {rd.full[c]['rvendi']:18.2f}   {point[c]:+.2f} [{np.percentile(dev, 2.5):+.2f},{np.percentile(dev, 97.5):+.2f}]{flag}")

# Figure: one axis, precision vs realistic Vendi; families in fixed categorical order, direct labels on every point.
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FAM = [("condition damping", "#257de0", "o", CURVE),
       ("score space", "#f47230", "s", ["score_1", "spec_1", "swap_1"]),
       ("lower or windowed w", "#00b894", "^", ["wall_3", "late_1", "scalar_1", "gstart_0.06", "gstart_0.12", "gstart_0.2", "gstart_0.12+wall_5"]),
       ("inside the network", "#f2a600", "D", ["mcond_1.3333+mchunks_8", "mcond_1.3333+mchunks_9", "tfeat_1.3333", "tfeat_1.3333+score_1"])]
NAME = {"w4": "w = 4", "cond_0.5": "cond 0.5", "cond_0.75": "cond 0.75", "cond_1": "cond 1", "score_1": "score damping",
        "spec_1": "shared push only", "swap_1": "swapped seed part", "wall_3": "w = 3", "late_1": "guidance early only",
        "scalar_1": "guidance late only", "mcond_1.3333+mchunks_8": "shift_mlp damping", "mcond_1.3333+mchunks_9": "shift damping",
        "tfeat_1.3333": "tfeat", "tfeat_1.3333+score_1": "tfeat + score", "w1": "w = 1", "real sample": "real images",
        "gstart_0.06": "guidance from t = 0.06", "gstart_0.12": "guidance from t = 0.12", "gstart_0.2": "guidance from t = 0.2",
        "gstart_0.12+wall_5": "w = 5 from t = 0.12"}
INK, INK2, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.set_facecolor("white"); fig.patch.set_facecolor("white")
for side in ("top", "right"): ax.spines[side].set_visible(False)
for side in ("left", "bottom"): ax.spines[side].set_color(GRID)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2, length=0)
xy = lambda c: (rd.full[c]["prec"], rd.full[c]["rvendi"])
ax.plot(*zip(*sorted(xy(c) for c in CURVE)), color="#257de0", lw=2, zorder=2)
for j, (label, col, mk, cs) in enumerate(FAM):
    cs = [c for c in cs if c in rd.full]
    if not cs:
        continue  # a family with no condition on these seeds
    ax.scatter(*zip(*[xy(c) for c in cs]), s=64, color=col, marker=mk, edgecolor="white", linewidth=2, zorder=4 if mk == "^" else 3, label=label)
for c, mk in (("w1", "o"), ("real sample", "*")):
    if c not in rd.full:
        continue
    ax.scatter(*xy(c), s=150 if mk == "*" else 64, color=INK if c == "real sample" else "#9c9c9c", marker=mk, edgecolor="white", linewidth=2, zorder=3)
OFF = {"w4": (6, -12), "cond_1": (-8, -4), "cond_0.75": (6, 4), "cond_0.5": (6, -4), "score_1": (6, 4), "spec_1": (-8, -3),
       "swap_1": (-10, -12), "wall_3": (-4, 9), "late_1": (6, -4), "scalar_1": (-8, -4), "mcond_1.3333+mchunks_8": (6, -16),
       "mcond_1.3333+mchunks_9": (-4, -16), "tfeat_1.3333": (-8, 4), "tfeat_1.3333+score_1": (6, 4), "w1": (6, -4), "real sample": (-8, -4),
       "gstart_0.06": (6, 4), "gstart_0.12": (-6, -13), "gstart_0.2": (6, 4), "gstart_0.12+wall_5": (6, 4)}
# The tight cluster near precision 0.65 gets leader lines out to open space (data coordinates).
LEAD = {"mcond_1.3333+mchunks_9": (0.625, 3.22, "right"), "mcond_1.3333+mchunks_8": (0.66, 2.88, "right"), "wall_3": (0.69, 3.95, "left")}
for c, (dx, dy) in OFF.items():
    if c in rd.full and c not in LEAD:
        ax.annotate(NAME[c], xy(c), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK, ha="right" if dx < 0 else "left")
for c, (tx, ty, ha) in LEAD.items():
    if c in rd.full:
        ax.annotate(NAME[c], xy(c), xytext=(tx, ty), textcoords="data", fontsize=8, color=INK, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=2, shrinkB=5))
ax.set_xlabel("precision (share of samples inside the real-image manifold)", color=INK2)
ax.set_ylabel("realistic distinct images per 32 draws", color=INK2)
SEEDS = f"{rd.seeds.start}-{rd.seeds.stop - 1}"
ax.set_title(f"Realistic diversity vs fidelity, seeds {SEEDS}", color=INK, loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.32, 1.0), labelcolor=INK)
path = "outputs/guidance_delta/figures/frontier_realistic.png" if SEEDS == "16-47" else f"outputs/guidance_delta/figures/frontier_realistic_{SEEDS}.png"
fig.tight_layout(); fig.savefig(path, dpi=170)
print("figure:", path)
