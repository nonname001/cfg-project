"""Diversity that counts only realistic images: each condition scored against the real distribution of its class.

Mean pairwise distance and Vendi reward spread anywhere in embedding space, including regions no real image of the class occupies
(w = 1 scores as more diverse than real photos). These measures count only what lies on the real manifold, or compare with the
real distribution directly. Per class, 32 random training images are held out as a "real sample" (the baseline a perfect
generator would match); every condition and that sample are scored against the remaining 1,268 (DINOv2 features, k = 3):
  prec     share of samples inside the real manifold (Kynkaenniemi et al.)
  cov      coverage (Naeem et al. 2020): share of real reference images whose k-NN ball holds a sample. Spread off the manifold
           earns nothing; for 32 samples against 1,268 images the values are small but comparable
  rvendi   realistic distinct images per 32 draws: Vendi score (cosine) over the samples inside the manifold only
  kid      kernel inception distance (x 1000) to the reference, cubic polynomial kernel, unbiased: penalises both off-manifold spread
           and over-concentration; the real sample's value is the floor
Averaged over the 8 classes. Intervals: 1000 paired half-samples of 16 seeds (coverage and rvendi depend on the whole sample set),
scaled by 1/sqrt(2); "vs w4" rows are paired differences.
usage: realistic_diversity.py [run] [seed_lo seed_hi]   (default SiT-L_2_night 16 47; REAL default the 1,300-image reference)
"""
import os, sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from common import LABELS
from guidance_analysis import vendi
from real_fidelity import knn_radius

args = sys.argv[1:] if __name__ == "__main__" else os.environ.get("RD_ARGS", "").split()  # RD_ARGS="<run> <lo> <hi>" when imported
run = Path(args[0]) if args else Path("outputs/guidance_delta/SiT-L_2_night")
seeds = range(int(args[1]), int(args[2]) + 1) if len(args) > 2 else range(16, 48)
real = np.load(os.environ.get("REAL", "outputs/real_imagenet_train") + "/dino.npz")
rng = np.random.default_rng(0)
NHOLD = max(32, len(seeds))  # the real sample needs one image per seed; 32 (the original split) for up to 32 seeds
REF, HOLD, RAD, KYY = {}, {}, {}, {}
for l in LABELS:
    X = real["dino"][real["labels"] == l].astype(np.float64)
    perm = rng.permutation(len(X))
    HOLD[l], REF[l] = X[perm[:NHOLD]], X[perm[NHOLD:]]
    RAD[l] = knn_radius(REF[l], int(os.environ.get("RD_K", 3)))  # RD_K: neighbours defining the real manifold (default 3)
d = REF[LABELS[0]].shape[1]
kern = lambda A, B: (A @ B.T / d + 1) ** 3
for l in LABELS:
    K = kern(REF[l], REF[l]); m = len(K)
    KYY[l] = (K.sum() - np.trace(K)) / (m * (m - 1))
    REF_SQ = None

conds = sorted(p.name for p in (run / "embeddings").iterdir() if p.is_dir()
               and all((run / "embeddings" / p.name / f"seed_{s}.npz").exists() for s in seeds))
G = {c: np.stack([np.load(run / "embeddings" / c / f"seed_{s}.npz")["dino"] for s in seeds]).astype(np.float64) for c in conds}
G["real sample"] = np.stack([HOLD[l] for l in LABELS], 1)  # (32, 8, D): plays the role of 32 "seeds"
names = ["real sample"] + conds
n = len(seeds)
D2 = {c: [np.sqrt(np.clip((G[c][:, i] ** 2).sum(1)[:, None] + (REF[l] ** 2).sum(1)[None] - 2 * G[c][:, i] @ REF[l].T, 0, None))
          for i, l in enumerate(LABELS)] for c in names}  # (n, 1268) per class
INB = {c: [D2[c][i] <= RAD[l][None] for i, l in enumerate(LABELS)] for c in names}  # (n, 1268): sample j inside ball of real i
KXY = {c: [kern(G[c][:, i], REF[l]) for i, l in enumerate(LABELS)] for c in names}
KXX = {c: [kern(G[c][:, i], G[c][:, i]) for i in range(8)] for c in names}


def metrics(c, idx):
    out = dict(prec=[], cov=[], rvendi=[], kid=[])
    for i, l in enumerate(LABELS):
        inb = INB[c][i][idx]
        inside = inb.any(1)
        out["prec"].append(inside.mean())
        out["cov"].append(inb.any(0).mean())
        x = G[c][idx, i][inside]
        out["rvendi"].append(vendi(x) * 32 / len(idx) if len(x) > 1 else float(len(x)) * 32 / len(idx))
        kxx = KXX[c][i][np.ix_(idx, idx)]; m = len(idx)
        out["kid"].append(1e3 * ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + KYY[l] - 2 * KXY[c][i][idx].mean()))
    return {k: float(np.mean(v)) for k, v in out.items()}


KEYS = ("prec", "cov", "rvendi", "kid")
full = {c: metrics(c, np.arange(n)) for c in names}
halves = [np.sort(rng.permutation(n)[: n // 2]) for _ in range(1000)]
H = {c: np.array([[metrics(c, h)[k] for k in KEYS] for h in halves]) for c in names}
ci = lambda c, j, ref=None: (lambda v: f"[{np.percentile(v, 2.5):+.3f},{np.percentile(v, 97.5):+.3f}]")(
    (H[c][:, j] - (H[ref][:, j] if ref else 0) - (H[c][:, j] - (H[ref][:, j] if ref else 0)).mean()) / np.sqrt(2)
    + full[c][KEYS[j]] - (full[ref][KEYS[j]] if ref else 0))
if __name__ == "__main__":
    print(f"{run.name}, seeds {seeds.start}-{seeds.stop - 1}; reference: {1300 - NHOLD} real images per class ({NHOLD} held out as the real sample)")
    print(f"{'':42} {'precision':>9} {'coverage':>9} {'real. Vendi/32':>15} {'KID x1e3':>10}")
    for c in sorted(names, key=lambda c: full[c]["kid"]):
        m = full[c]
        print(f"{c:42} {m['prec']:9.2f} {m['cov']:9.3f} {m['rvendi']:15.2f} {m['kid']:10.2f}")
    print("\npaired differences vs w4 (half-sample intervals): coverage, realistic Vendi, KID")
    for c in sorted(names, key=lambda c: full[c]["kid"]):
        if c != "w4":
            print(f"{c:42} cov {full[c]['cov'] - full['w4']['cov']:+.3f} {ci(c, 1, 'w4')}  rvendi {full[c]['rvendi'] - full['w4']['rvendi']:+.2f} "
                  f"{ci(c, 2, 'w4')}  kid {full[c]['kid'] - full['w4']['kid']:+.2f} {ci(c, 3, 'w4')}")
