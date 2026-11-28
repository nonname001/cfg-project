"""Analyse patch_sweep.py: how far each patch moves a sample from its own outcome (0) toward the donor run's outcome (1).

Per condition <direction>_<mode>, per sample (seed x class), as in switch_analysis.py:
  lambda = <H - A, B - A> / |B - A|^2, A = recipient's own final (w1 for r1, w4 for r4), B = donor's final (the other run),
  in DINOv2 space (sem, unit embeddings) and in the low band of the final latents (layout); off = distance from the A-B line.
Medians over samples with 95% intervals from resampling samples (seed x class pairs). "share" divides a write patch's median
lambda by the velocity transplant's at the same k: the fraction of what any K-step intervention at k can transfer that block b's
own computation carries. For r4 (undoing guidance) also population DINO diversity recovery (0 = w4, 1 = w1) and precision
against the real images (REAL, default the 1,300-image reference).
usage: patch_analysis.py <patch run>   -> <run>/analysis/patch.csv
"""
import os, re, sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from common import LABELS
from guidance_analysis import pairwise_cos_distance, unit, write_csv
from guidance_window import band_split
from real_fidelity import knn_radius

run = Path(sys.argv[1])
conds = sorted(p.name for p in (run / "embeddings").iterdir() if p.is_dir())
# Each condition is scored on its own seeds (the confirmation grid has fewer blocks than the screen on seeds 0-1), against the
# w1 / w4 references on the same seeds.
SEEDS = {c: sorted(int(p.stem.split("_")[1]) for p in (run / "embeddings" / c).glob("seed_*.npz") if ".tmp" not in p.name) for c in conds}
emb = lambda c: {s: np.load(run / "embeddings" / c / f"seed_{s}.npz")["dino"].astype(np.float64) for s in SEEDS[c]}
lat = lambda c: {s: band_split(np.load(run / c / f"seed_{s}.npz")["final"].astype(np.float64))[0] for s in SEEDS[c]}
RAW = {c: emb(c) for c in conds}
LAY = {c: lat(c) for c in conds}
get = lambda D, c, seeds: np.stack([D[c][s] for s in seeds])  # (seeds, 8, dim)
real = np.load(os.environ.get("REAL", "outputs/real_imagenet_train") + "/dino.npz")
R = {l: real["dino"][real["labels"] == l].astype(np.float64) for l in LABELS}; RAD = {l: knn_radius(R[l]) for l in LABELS}
inside = lambda X: np.stack([(np.linalg.norm(X[:, i, None] - R[l][None], axis=-1) <= RAD[l][None]).any(1) for i, l in enumerate(LABELS)], 1)


def place(H, A, B):
    d = B - A
    lam = ((H - A) * d).sum(-1) / (d ** 2).sum(-1)
    off = np.linalg.norm(H - (A + lam[..., None] * d), axis=-1) / np.linalg.norm(d, axis=-1)
    return lam.ravel(), off.ravel()


rng = np.random.default_rng(0)
div = lambda X: np.mean([pairwise_cos_distance(unit(X[:, i])) for i in range(8)])
rows = []
for c in conds:
    m = re.match(r"(r[14])_(write_b(\d+)|vel|band_(\w+?)|lowshared|lowspec)_k(\d+)$", c)
    if not m:
        continue
    d, k = m.group(1), int(m.group(5))
    A, B = ("w1", "w4") if d == "r1" else ("w4", "w1")
    seeds = [s for s in SEEDS[c] if s in SEEDS["w1"] and s in SEEDS["w4"]]
    n = len(seeds) * 8
    boots = [rng.choice(n, n) for _ in range(1000)]
    med = lambda v: (np.median(v), *np.percentile([np.median(v[b]) for b in boots], [2.5, 97.5]))
    row = dict(condition=c, direction=d, k=k, s_t=k / 49, kind="write" if m.group(3) else (m.group(2) if m.group(2) in ("vel", "lowshared", "lowspec") else "band"),
               block=int(m.group(3)) if m.group(3) else -1, band=m.group(4) or "", n_seeds=len(seeds))
    for space, D in (("sem", RAW), ("layout", LAY)):
        f = (lambda X: unit(X)) if space == "sem" else (lambda X: X)
        lam, off = place(f(get(D, c, seeds)), f(get(D, A, seeds)), f(get(D, B, seeds)))
        row[f"{space}"], row[f"{space}_low"], row[f"{space}_high"] = med(lam)
        row[f"{space}_off"] = float(np.median(off))
    if d == "r4":
        X, X1, X4 = (get(RAW, x, seeds) for x in (c, "w1", "w4"))
        row["dino_recovery"] = (div(X) - div(X4)) / (div(X1) - div(X4))
        row["precision"], row["precision_w4"], row["precision_w1"] = float(inside(X).mean()), float(inside(X4).mean()), float(inside(X1).mean())
    rows.append(row)
for r in rows:
    v = next((x for x in rows if x["direction"] == r["direction"] and x["kind"] == "vel" and x["k"] == r["k"]), None)
    r["sem_share"] = r["sem"] / v["sem"] if v and v["sem"] else float("nan")
(run / "analysis").mkdir(exist_ok=True)
keys = list(dict.fromkeys(k for r in rows for k in r))  # only r4 rows have dino_recovery / precision
write_csv(run / "analysis" / "patch.csv", [{k: r.get(k, float("nan")) for k in keys} for r in rows])
print(f"{run.name}: each condition on its own seeds (n = seeds; x 8 classes = samples)")
for d, title in (("r1", "recipient w = 1, donor w = 4: does this part carry the guided outcome?"),
                 ("r4", "recipient w = 4, donor w = 1: can this part undo it?")):
    print(f"\n== {title}")
    for k in sorted({r["k"] for r in rows if r["direction"] == d}):
        print(f"-- k = {k} (s_t {k / 49:.2f}): median lambda toward the donor's outcome [95%]; off = distance from the line")
        for r in sorted((r for r in rows if r["direction"] == d and r["k"] == k), key=lambda r: (r["kind"] != "vel", r["kind"], r["block"], r["band"])):
            name = {"write": f"write block {r['block']:2d}", "vel": "velocity transplant", "band": f"x_t band {r['band']}",
                    "lowshared": "low band, shared", "lowspec": "low band, seed-spec."}[r["kind"]]
            extra = f" | DINO recovery {r['dino_recovery']:+.2f}, precision {r['precision']:.2f} (w4 {r['precision_w4']:.2f})" if d == "r4" else ""
            share = f" (share {r['sem_share']:.2f})" if r["kind"] == "write" else ""
            print(f"   {name:20} sem {r['sem']:+.2f} [{r['sem_low']:+.2f},{r['sem_high']:+.2f}]{share}  layout {r['layout']:+.2f} "
                  f"[{r['layout_low']:+.2f},{r['layout_high']:+.2f}]  off {r['sem_off']:.2f}/{r['layout_off']:.2f}  n={r['n_seeds']}{extra}")
