"""Analyse switch_sweep.py: when each sample commits to its w = 4 outcome, for layout and for semantics.

For every seed, class and switch step k, the hybrid H's final image is placed on the segment from that seed's w = 1 image A
(0) to its w = 4 image B (1):  lambda = <H - A, B - A> / |B - A|^2, in two spaces:
  sem     DINOv2 embedding, unit length
  layout  low-band part of the final latents (band_split, as in the layout diversity measure)
plus off = |H - (A + lambda (B - A))| / |B - A|, how far H lies off the segment (0 = on it).
  w4to1_k  w = 4 before step k, w = 1 after: lambda rises from 0 (k = 0) to 1 as more of the trajectory is guided
  w1to4_k  w = 1 before step k, w = 4 after: lambda falls from 1 to 0
The "commitment" of a sample to its w = 4 outcome by step k is lambda(w4to1_k); 1 - lambda(w1to4_k) says how much guidance
before k is no longer needed. Population measures per condition: DINO mean pairwise distance and layout spread V (diversity),
and precision against the real images (REAL, default the 1,300-image training reference).
Medians over samples are reported for lambda (single samples can sit far off the segment when |B - A| is small); intervals
resample seeds. usage: switch_analysis.py <switch run> [--figure]   -> <run>/analysis/switch.csv (+ figure)
"""
import os, sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from common import LABELS
from guidance_analysis import pairwise_cos_distance, unit, write_csv
from guidance_window import band_split, spread_v
from real_fidelity import knn_radius

run = Path(sys.argv[1])
conds = sorted(p.name for p in (run / "embeddings").iterdir() if p.is_dir())
ks = sorted({int(c.split("_k")[1]) for c in conds if "_k" in c})
seeds = sorted(set.intersection(*[{int(p.stem.split("_")[1]) for p in (run / "embeddings" / c).glob("seed_*.npz")} for c in conds]))
load = lambda c, key: np.stack([np.load(run / ("embeddings/" if key == "dino" else "") / c / f"seed_{s}.npz")[key] for s in seeds]).astype(np.float64)
SEM = {c: unit(load(c, "dino")) for c in conds}  # (seeds, 8, 768)
LAY = {c: band_split(load(c, "final"))[0] for c in conds}  # (seeds, 8, D)
real = np.load(os.environ.get("REAL", "outputs/real_imagenet_train") + "/dino.npz")
R = {l: real["dino"][real["labels"] == l].astype(np.float64) for l in LABELS}; RAD = {l: knn_radius(R[l]) for l in LABELS}
RAW = {c: load(c, "dino") for c in conds}
INSIDE = {c: np.stack([(np.linalg.norm(RAW[c][:, i, None] - R[l][None], axis=-1) <= RAD[l][None]).any(1) for i, l in enumerate(LABELS)], 1) for c in conds}


def place(H, A, B):
    d = B - A
    lam = ((H - A) * d).sum(-1) / (d ** 2).sum(-1)
    off = np.linalg.norm(H - (A + lam[..., None] * d), axis=-1) / np.linalg.norm(d, axis=-1)
    return lam, off


rng = np.random.default_rng(0); n = len(seeds)
boots = [np.arange(n)] + [np.unique(rng.choice(n, n)) for _ in range(1000)]
ci = lambda v: (v[0], np.percentile(v[1:], 2.5), np.percentile(v[1:], 97.5))
rows = []
for name in ["w1", "w4"] + [f"{d}_k{k}" for d in ("w4to1", "w1to4") for k in ks]:
    k = int(name.split("_k")[1]) if "_k" in name else (0 if name == "w1" else 49)
    row = dict(condition=name, direction=name.split("_k")[0] if "_k" in name else "", k=k, s_t=k / 49)
    for space, X in (("sem", SEM), ("layout", LAY)):
        lam, off = place(X[name], X["w1"], X["w4"])
        row[f"{space}_lambda"], row[f"{space}_lambda_low"], row[f"{space}_lambda_high"] = ci([np.median(lam[b]) for b in boots])
        row[f"{space}_off"] = float(np.median(off))
    row["dino_div"], row["dino_div_low"], row["dino_div_high"] = ci([np.mean([pairwise_cos_distance(SEM[name][b, i]) for i in range(8)]) for b in boots])
    row["layout_V"], row["layout_V_low"], row["layout_V_high"] = ci([float(spread_v(np.moveaxis(LAY[name][b], 0, 1)).mean()) for b in boots])
    row["precision"], row["precision_low"], row["precision_high"] = ci([INSIDE[name][b].mean() for b in boots])
    rows.append(row)
    print(f"{name:12} s_t {row['s_t']:.2f}  sem lambda {row['sem_lambda']:.2f} [{row['sem_lambda_low']:.2f},{row['sem_lambda_high']:.2f}] "
          f"layout lambda {row['layout_lambda']:.2f} [{row['layout_lambda_low']:.2f},{row['layout_lambda_high']:.2f}]  off {row['sem_off']:.2f}/{row['layout_off']:.2f}  "
          f"dino div {row['dino_div']:.3f}  layout V {row['layout_V']:.3f}  prec {row['precision']:.2f}", flush=True)
(run / "analysis").mkdir(exist_ok=True)
write_csv(run / "analysis" / "switch.csv", rows)

# Per-sample commitment step: first k where lambda(w4to1_k) >= 0.5 (linear interpolation between grid points).
for space, X in (("sem", SEM), ("layout", LAY)):
    grid = [0] + ks + [49]
    lam = np.stack([np.zeros(X["w1"].shape[:2])] + [place(X[f"w4to1_k{k}"], X["w1"], X["w4"])[0] for k in ks] + [np.ones(X["w1"].shape[:2])])
    t50 = np.full(lam.shape[1:], np.nan)
    for idx in np.ndindex(*lam.shape[1:]):
        v = np.maximum.accumulate(lam[(slice(None),) + idx])  # first crossing of 0.5
        j = np.argmax(v >= 0.5)
        t50[idx] = (grid[j - 1] + (0.5 - v[j - 1]) / (v[j] - v[j - 1] + 1e-12) * (grid[j] - grid[j - 1])) / 49 if j > 0 else 0.0
    q = np.percentile(t50, [25, 50, 75])
    print(f"{space}: s_t at which a sample is half committed to its w = 4 outcome: median {q[1]:.2f} (IQR {q[0]:.2f}-{q[2]:.2f}); "
          f"per class medians " + ", ".join(f"{np.median(t50[:, i]):.2f}" for i in range(8)))
    np.save(run / "analysis" / f"t50_{space}.npy", t50)

if "--figure" in sys.argv:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for d, ls in (("w4to1", "-"), ("w1to4", "--")):
        sub = [r for r in rows if r["direction"] == d]
        t = [r["s_t"] for r in sub]
        for space, col in (("layout", "#2a6fdb"), ("sem", "#d9480f")):
            ax[0].plot(t, [r[f"{space}_lambda"] for r in sub], ls, color=col, marker="o", ms=3, label=f"{space}, {'w=4 then w=1' if d == 'w4to1' else 'w=1 then w=4'}")
            ax[0].fill_between(t, [r[f"{space}_lambda_low"] for r in sub], [r[f"{space}_lambda_high"] for r in sub], color=col, alpha=0.12)
        w1, w4 = rows[0], rows[1]
        ax[1].plot(t, [(r["dino_div"] - w4["dino_div"]) / (w1["dino_div"] - w4["dino_div"]) for r in sub], ls, color="#d9480f", marker="o", ms=3, label=f"DINO, {d}")
        ax[1].plot(t, [(r["layout_V"] - w4["layout_V"]) / (w1["layout_V"] - w4["layout_V"]) for r in sub], ls, color="#2a6fdb", marker="o", ms=3, label=f"layout, {d}")
        ax[2].plot(t, [r["precision"] for r in sub], ls, color="#333", marker="o", ms=3, label=d)
    ax[0].set(xlabel="switch time s_t", ylabel="position between w=1 (0) and w=4 (1) outcome", title="Per-sample commitment")
    ax[1].set(xlabel="switch time s_t", ylabel="diversity (0 = w=4, 1 = w=1)", title="Population diversity")
    ax[2].axhline(rows[0]["precision"], color="#999", lw=0.8); ax[2].axhline(rows[1]["precision"], color="#999", lw=0.8, ls=":")
    ax[2].set(xlabel="switch time s_t", ylabel="precision (1,300-image reference)", title="Fidelity (grey: w=1 solid, w=4 dotted)")
    for a in ax: a.axvline(0.3, color="#bbb", lw=0.8); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(run / "analysis" / "switch.png", dpi=150)
