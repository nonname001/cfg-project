"""Is the semantic diversity that late guidance removes real diversity, or a quality repair? (open direction 1)

Guidance only after s_t = 0.3 reproduces most of w = 4's DINO/CLIP diversity loss (RESULTS.md 9) but none of its layout loss.
This scores the two window conditions against the real ImageNet training images of each class (1,300 per class, DINOv2 space):

  early-only  late_1    w = 4 for s_t < 0.3, w = 1 after
  late-only   scalar_1  w = 1 for s_t < 0.3, w = 4 after

Per class, then averaged over the 8 classes:
  prec   Kynkaenniemi precision, k = 3: share of generated images inside some real image's k-NN ball (realism)
  rec    Kynkaenniemi recall, k = 3: share of real images inside some generated image's k-NN ball (among the class's samples)
  cov    Naeem et al. coverage, k = 3: share of real images whose own k-NN ball holds a generated image. Unlike recall it does
         not reward generated outliers, whose k-NN balls are large
  dino   mean pairwise cosine distance (the diversity measure used throughout)
  typ    cosine distance to the real class mean: how typical a sample is. Real images give the reference value
  nn     Euclidean distance to the nearest real image (lower = closer to some real photo)
"real" rows give the same measures for real images (pairwise distance and typicality over all 1,300; nn and the others do not
apply). Two contrasts isolate late guidance: w1 -> late-only (late guidance on top of early w = 1) and early-only -> w4 (on
top of early w = 4). For each, "dino among both-inside" is the diversity of only the samples that are inside the real manifold
under both conditions of the contrast: if late guidance still lowers it, it removes diversity among already-realistic images.
Intervals: 95% percentile over 1000 seed resamples, the same resample for every condition (recall and coverage: half-samples,
see below).
usage: window_fidelity.py <run> <seed_lo> <seed_hi>   (REAL=<dir with dino.npz>, default outputs/real_imagenet_train)
"""
import os, sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from common import LABELS
from guidance_analysis import pairwise_cos_distance, unit
from real_fidelity import knn_radius, K

run = Path(sys.argv[1]); seeds = range(int(sys.argv[2]), int(sys.argv[3]) + 1)
NAMES = ["w1", "late_1", "scalar_1", "w4"]
LABEL = dict(w1="w = 1", late_1="early-only", scalar_1="late-only", w4="w = 4")
real = np.load(os.environ.get("REAL", "outputs/real_imagenet_train") + "/dino.npz")
R = {l: real["dino"][real["labels"] == l].astype(np.float64) for l in LABELS}
RAD = {l: knn_radius(R[l]) for l in LABELS}  # (1300,) per class
MEAN = {l: unit(R[l]).mean(0) for l in LABELS}
G = {c: np.stack([np.load(run / "embeddings" / c / f"seed_{s}.npz")["dino"] for s in seeds]).astype(np.float64) for c in NAMES}
n = len(seeds)

# Per-sample quantities that do not depend on the other generated samples: computed once.
D = {c: [np.sqrt(np.clip((G[c][:, i] ** 2).sum(1)[:, None] + (R[l] ** 2).sum(1)[None] - 2 * G[c][:, i] @ R[l].T, 0, None))
         for i, l in enumerate(LABELS)] for c in NAMES}  # (seeds, 1300) per class
INSIDE = {c: np.stack([(D[c][i] <= RAD[l][None]).any(1) for i, l in enumerate(LABELS)], 1) for c in NAMES}  # (seeds, 8)
NN = {c: np.stack([D[c][i].min(1) for i in range(8)], 1) for c in NAMES}
TYP = {c: np.stack([1 - unit(G[c][:, i]) @ (MEAN[l] / np.linalg.norm(MEAN[l])) for i, l in enumerate(LABELS)], 1) for c in NAMES}


def metrics(c, idx):
    rec, cov = [], []
    for i, l in enumerate(LABELS):
        d = D[c][i][idx]  # (m, 1300)
        g = G[c][idx, i]
        rec.append((d <= knn_radius(g)[:, None]).any(0).mean())
        cov.append((d <= RAD[l][None]).any(0).mean())
    return dict(prec=INSIDE[c][idx].mean(), rec=np.mean(rec), cov=np.mean(cov),
                dino=np.mean([pairwise_cos_distance(G[c][idx, i]) for i in range(8)]),
                typ=TYP[c][idx].mean(), nn=NN[c][idx].mean())


def both_inside(a, b, idx):
    """Diversity of a's and b's samples, restricted per class to seeds inside the real manifold under both a and b."""
    out = []
    for i in range(8):
        keep = idx[INSIDE[a][idx, i] & INSIDE[b][idx, i]]
        if len(keep) >= 3:
            out.append((pairwise_cos_distance(G[a][keep, i]), pairwise_cos_distance(G[b][keep, i])))
    return np.array(out).mean(0) if out else np.array([np.nan, np.nan])


KEYS = ("prec", "rec", "cov", "dino", "typ", "nn")
CONTRASTS = [("w1", "scalar_1"), ("late_1", "w4")]
rng = np.random.default_rng(0)
boot = {c: [] for c in NAMES}; bi = {p: [] for p in CONTRASTS}
for b in range(1001):
    idx = np.arange(n) if b == 0 else np.unique(rng.choice(n, n))  # b = 0: the point estimate on all seeds
    for c in NAMES:
        boot[c].append([metrics(c, idx)[k] for k in KEYS])
    for p in CONTRASTS:
        bi[p].append(both_inside(*p, idx))
boot = {c: np.array(v) for c, v in boot.items()}; bi = {p: np.array(v) for p, v in bi.items()}
# Recall and coverage depend on the whole generated set, so a bootstrap resample (np.unique keeps ~20 of 32 seeds) biases them
# downward. Their intervals instead come from random half-samples (16 seeds, the same halves for every condition, so
# differences stay paired), centered on the full-sample value and scaled by 1/sqrt(2) to the full sample size.
halves = [rng.permutation(n)[: n // 2] for _ in range(1000)]
for c in NAMES:
    for j, k in enumerate(KEYS):
        if k in ("rec", "cov"):
            h = np.array([metrics(c, np.sort(idx))[k] for idx in halves])
            boot[c][1:, j] = boot[c][0, j] + (h - h.mean()) / np.sqrt(2)
ci = lambda v: f"{v[0]:.3f} [{np.nanpercentile(v[1:], 2.5):.3f},{np.nanpercentile(v[1:], 97.5):.3f}]"

print(f"{run.name}, seeds {seeds.start}-{seeds.stop - 1} ({n}), real reference: {sum(len(r) for r in R.values())} images")
print(f"{'':12}" + "".join(f"{k:>24}" for k in KEYS))
for c in NAMES:
    print(f"{LABEL[c]:12}" + "".join(f"{ci(boot[c][:, j]):>24}" for j in range(len(KEYS))))
rd = np.mean([pairwise_cos_distance(R[l]) for l in LABELS]); rt = np.mean([(1 - unit(R[l]) @ (MEAN[l] / np.linalg.norm(MEAN[l]))).mean() for l in LABELS])
print(f"{'real':12}{'':>24}{'':>24}{'':>24}{rd:>24.3f}{rt:>24.3f}")

print("\nshare of the w1 -> w4 change reproduced (0 = as w1, 1 = as w4)")
share = lambda c: (boot[c] - boot["w1"]) / (boot["w4"] - boot["w1"])
for c in ("late_1", "scalar_1"):
    s = share(c)
    print(f"{LABEL[c]:12}" + "".join(f"{ci(s[:, j]):>24}" for j in range(len(KEYS))))

print("\nlate guidance, paired differences (after - before); 'among both-inside' = dino of seeds inside under both")
for a, b in CONTRASTS:
    d = boot[b] - boot[a]
    print(f"{LABEL[a]} -> {LABEL[b]}: " + ", ".join(f"{k} {d[0, j]:+.3f} [{np.percentile(d[1:, j], 2.5):+.3f},{np.percentile(d[1:, j], 97.5):+.3f}]" for j, k in enumerate(KEYS)))
    v = bi[(a, b)]; dd = v[:, 1] - v[:, 0]
    print(f"    dino among both-inside: {v[0, 0]:.3f} -> {v[0, 1]:.3f}, change {dd[0]:+.3f} [{np.nanpercentile(dd[1:], 2.5):+.3f},{np.nanpercentile(dd[1:], 97.5):+.3f}]")
