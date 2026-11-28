"""Sub-clusters of each class's real images in DINOv2 space, and how generated samples spread over them.

1. Per class, cosine k-means (k-means++ init, 10 restarts) on the 1,300 real training embeddings for K = 2..12, with the mean
   silhouette (cosine) of each: is there sub-cluster structure, and at what granularity?
2. At a fixed K (--k, default 8): assign each generated sample (seeds 16-47, 32 per class) and a held-out real sample (32 training
   images per class, excluded from the fit) to its nearest centroid. Reported per condition, averaged over classes:
     eff_sub   effective number of subclusters used, exp(entropy) of the assignment counts (real sample = the target)
     tv        total variation distance between the assignment shares and the real subcluster shares (0 = same proportions)
     top       share of samples in the single most-used subcluster
     inside    the same eff_sub over on-manifold samples only (precision test against the 1,268 reference images)
3. Writes the centroids, real shares and each validation image's subcluster to <out>/subclusters_k<K>.npz, for steering.
usage: subclusters.py [--k 8] [--out outputs/guidance_delta/analysis]
"""
import argparse, sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from common import LABELS, LABEL_NAMES
from guidance_analysis import unit
from real_fidelity import knn_radius

p = argparse.ArgumentParser(); p.add_argument("--k", type=int, default=8); p.add_argument("--out", default="outputs/guidance_delta/analysis")
a = p.parse_args()
rng = np.random.default_rng(0)
train = np.load("outputs/real_imagenet_train/dino.npz"); val = np.load("outputs/real_imagenet_val/dino.npz")
run = Path("outputs/guidance_delta/SiT-L_2_night"); SEEDS = range(16, 48)
CONDS = ["w1", "w4", "score_1", "wall_3", "cond_0.5", "cond_0.75", "cond_1", "scalar_1"]


def kmeans(X, K, restarts=10, iters=60):
    """Cosine k-means on unit vectors. Returns (centroids (K, D) unit, labels, inertia = sum of cosine distances)."""
    best = None
    for _ in range(restarts):
        C = [X[rng.integers(len(X))]]
        for _ in range(K - 1):  # k-means++
            d = np.clip(1 - np.max(X @ np.array(C).T, 1), 0, None); C.append(X[rng.choice(len(X), p=d / d.sum())])
        C = np.array(C)
        for _ in range(iters):
            lab = np.argmax(X @ C.T, 1)
            C_new = np.stack([unit(X[lab == k].mean(0)) if (lab == k).any() else C[k] for k in range(K)])
            if np.allclose(C_new, C): break
            C = C_new
        inertia = float((1 - np.max(X @ C.T, 1)).sum())
        if best is None or inertia < best[2]:
            best = (C, np.argmax(X @ C.T, 1), inertia)
    return best


def silhouette(X, lab):
    D = 1 - X @ X.T
    s = []
    for i in range(len(X)):
        same = lab == lab[i]
        a_ = D[i, same].sum() / max(same.sum() - 1, 1)
        b_ = min(D[i, lab == k].mean() for k in np.unique(lab) if k != lab[i])
        s.append((b_ - a_) / max(a_, b_))
    return float(np.mean(s))


R = {l: unit(train["dino"][train["labels"] == l].astype(np.float64)) for l in LABELS}
RAW = {l: train["dino"][train["labels"] == l].astype(np.float64) for l in LABELS}
print("1. silhouette (cosine) of k-means on the 1,300 real images per class; 0 = no structure, higher = clearer subclusters")
print("   K:      " + " ".join(f"{k:5d}" for k in range(2, 13)))
for l, name in zip(LABELS, LABEL_NAMES):
    print(f"   {name:17}" + " ".join(f"{silhouette(R[l], kmeans(R[l], k, restarts=3)[1]):5.2f}" for k in range(2, 13)), flush=True)

K = a.k
G = {c: np.stack([np.load(run / "embeddings" / c / f"seed_{s}.npz")["dino"] for s in SEEDS]).astype(np.float64) for c in CONDS}
out = dict(k=K)
rows = {c: [] for c in ["real sample"] + CONDS}
for i, l in enumerate(LABELS):
    perm = rng.permutation(len(R[l])); hold, fit = perm[:32], perm[32:]
    C, lab, _ = kmeans(R[l][fit], K)
    share = np.bincount(lab, minlength=K) / len(lab)
    rad = knn_radius(RAW[l][fit])
    vl = unit(val["dino"][val["labels"] == l].astype(np.float64))
    out[f"centroids_{l}"], out[f"share_{l}"], out[f"val_sub_{l}"] = C, share, np.argmax(vl @ C.T, 1)
    samples = {"real sample": RAW[l][hold], **{c: G[c][:, i] for c in CONDS}}
    for c, Xr in samples.items():
        sub = np.argmax(unit(Xr) @ C.T, 1)
        inside = (np.linalg.norm(Xr[:, None] - RAW[l][fit][None], axis=-1) <= rad[None]).any(1)
        q = np.bincount(sub, minlength=K) / len(sub)
        qi = np.bincount(sub[inside], minlength=K) / max(inside.sum(), 1)
        H = lambda q: float(np.exp(-(q[q > 0] * np.log(q[q > 0])).sum())) if q.sum() else 0.0
        rows[c].append(dict(eff=H(q), tv=0.5 * np.abs(q - share).sum(), top=q.max(), inside=H(qi), n_in=int(inside.sum())))
Path(a.out).mkdir(parents=True, exist_ok=True)
np.savez(Path(a.out) / f"subclusters_k{K}.npz", **{k: v for k, v in out.items()})
real_eff = np.mean([float(np.exp(-(s[s > 0] * np.log(s[s > 0])).sum())) for s in (out[f"share_{l}"] for l in LABELS)])
print(f"\n2. K = {K}: effective number of subclusters in the real distribution itself: {real_eff:.2f} (mean over classes)")
print(f"   {'':12} {'eff. subclusters':>16} {'TV to real shares':>18} {'top share':>10} {'eff. (on-manifold only)':>24} {'n on-manifold':>14}")
for c, r in rows.items():
    m = lambda k: np.mean([x[k] for x in r])
    print(f"   {c:12} {m('eff'):16.2f} {m('tv'):18.2f} {m('top'):10.2f} {m('inside'):24.2f} {m('n_in'):14.1f}")
print(f"\n3. validation images per subcluster (steering references), class means: "
      f"{np.mean([np.bincount(out[f'val_sub_{l}'], minlength=K).min() for l in LABELS]):.1f} min, "
      f"{np.mean([np.bincount(out[f'val_sub_{l}'], minlength=K).max() for l in LABELS]):.1f} max per subcluster; "
      f"saved {Path(a.out) / f'subclusters_k{K}.npz'}")
