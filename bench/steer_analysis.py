"""Analyse subcluster_steer.py: does steering send samples to their target subcluster, and at what cost in realism?

Per condition steer_<s> (steer_0 = w = 4 with the same targets, the paired baseline), over the seeds every condition has:
  hit        share of samples whose nearest real-subcluster centroid (DINOv2, cosine) is their target
  cos_tgt    mean cosine between a sample's DINOv2 embedding and its target centroid
  eff_sub    effective number of subclusters used per class, exp(entropy) of the assignment shares (pooled over seeds)
  tv         total variation between those shares and the real subcluster shares
  prec       share of samples inside the real manifold (k = 3, the 1,300 training images of the class)
  rvendi     Vendi (DINOv2, cosine) over on-manifold samples only, per class (realistic distinct images among n seeds)
  dino       mean pairwise cosine distance (raw semantic spread)
Intervals for hit, cos_tgt and prec: paired seed bootstrap of the difference from steer_0 (1000 resamples).
With --prefix ref (ref_steer.py runs) also "copy": mean cosine between a sample and its own reference image minus its mean cosine
to the other validation images of the same subcluster (0 = no closer to its reference than to the reference's neighbours).
usage: steer_analysis.py <run> [--k 8] [--prefix steer|ref]
"""
import sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from common import LABELS
from guidance_analysis import pairwise_cos_distance, unit, vendi
from real_fidelity import knn_radius

run = Path(sys.argv[1]); K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 8
PRE = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else "steer"
BASE = f"{PRE}_0"
tab = np.load(f"outputs/guidance_delta/analysis/subclusters_k{K}.npz")
C = {l: tab[f"centroids_{l}"] for l in LABELS}; SHARE = {l: tab[f"share_{l}"] for l in LABELS}
train = np.load("outputs/real_imagenet_train/dino.npz")
R = {l: train["dino"][train["labels"] == l].astype(np.float64) for l in LABELS}; RAD = {l: knn_radius(R[l]) for l in LABELS}
conds = sorted((p.name for p in (run / "embeddings").iterdir() if p.is_dir() and p.name.startswith(PRE + "_")), key=lambda c: (c != BASE, c))
seeds = sorted(set.intersection(*[{int(f.stem.split("_")[1]) for f in (run / "embeddings" / c).glob("seed_*.npz")} for c in conds]))
D = {c: np.stack([np.load(run / "embeddings" / c / f"seed_{s}.npz")["dino"] for s in seeds]).astype(np.float64) for c in conds}
TGT = np.stack([np.load(run / BASE / f"seed_{s}.npz")["target"] for s in seeds])  # (seeds, 8)
for c in conds:
    assert all((np.load(run / c / f"seed_{s}.npz")["target"] == TGT[j]).all() for j, s in enumerate(seeds)), f"targets differ in {c}"


def per_sample(c):
    """(seeds, 8) arrays: hit, cosine to target, inside the manifold; and the assigned subcluster."""
    hit, cos, ins, sub = (np.zeros(TGT.shape) for _ in range(4))
    for i, l in enumerate(LABELS):
        u = unit(D[c][:, i]); sims = u @ C[l].T
        sub[:, i] = sims.argmax(1); hit[:, i] = sub[:, i] == TGT[:, i]; cos[:, i] = sims[np.arange(len(seeds)), TGT[:, i]]
        ins[:, i] = (np.linalg.norm(D[c][:, i, None] - R[l][None], axis=-1) <= RAD[l][None]).any(1)
    return hit, cos, ins, sub.astype(int)


P = {c: per_sample(c) for c in conds}
rng = np.random.default_rng(0); n = len(seeds)
boots = [rng.choice(n, n) for _ in range(1000)]
print(f"{run.name}: seeds {seeds[0]}-{seeds[-1]} ({n}), K = {K}; real subcluster shares are the targets' distribution")
print(f"{'':10} {'hit':>22} {'cos to target':>24} {'precision':>22} {'eff. subclusters':>17} {'TV':>6} {'real. Vendi':>12} {'DINO':>6}")
for c in conds:
    hit, cos, ins, sub = P[c]
    eff, tv, rv = [], [], []
    for i, l in enumerate(LABELS):
        q = np.bincount(sub[:, i], minlength=K) / n
        eff.append(np.exp(-(q[q > 0] * np.log(q[q > 0])).sum())); tv.append(0.5 * np.abs(q - SHARE[l]).sum())
        x = D[c][ins[:, i].astype(bool), i]; rv.append(vendi(x) if len(x) > 1 else float(len(x)))
    def fmt(arr, j):
        base = P[BASE][j]
        diff = [arr[b].mean() - base[b].mean() for b in boots]
        return f"{arr.mean():.2f} ({arr.mean() - base.mean():+.2f} [{np.percentile(diff, 2.5):+.2f},{np.percentile(diff, 97.5):+.2f}])" if c != BASE else f"{arr.mean():.2f}"
    print(f"{c:10} {fmt(hit, 0):>22} {fmt(cos, 1):>24} {fmt(ins, 2):>22} {np.mean(eff):17.2f} {np.mean(tv):6.2f} {np.mean(rv):12.2f} "
          f"{np.mean([pairwise_cos_distance(unit(D[c][:, i])) for i in range(8)]):6.3f}")
if PRE == "ref":
    val = np.load("outputs/real_imagenet_val/dino.npz")
    V = {l: unit(val["dino"][val["labels"] == l].astype(np.float64)) for l in LABELS}
    REF = np.stack([np.load(run / BASE / f"seed_{s}.npz")["ref"] for s in seeds])
    print("copy (cosine to own reference minus to other validation images of the same subcluster):")
    for c in conds:
        cp = []
        for i, l in enumerate(LABELS):
            vs = tab[f"val_sub_{l}"]
            for j in range(n):
                r = REF[j, i]; others = np.flatnonzero((vs == vs[r]) & (np.arange(len(vs)) != r))
                if len(others):
                    u = unit(D[c][j, i]); cp.append(u @ V[l][r] - (V[l][others] @ u).mean())
        print(f"   {c:14} {np.mean(cp):+.3f}")
print(f"(chance hit rate if samples ignored their targets but kept w4's subcluster use: {np.mean([(np.bincount(P[BASE][3][:, i], minlength=K) / n * SHARE[l]).sum() for i, l in enumerate(LABELS)]):.2f})")
