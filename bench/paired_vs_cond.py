"""Paired test of interventions against the condition-damping curve, at the same real-image fidelity.

For every condition, per bootstrap resample of seeds (the same resample for all conditions):
  precision retention  (P - P_w1) / (P_w4 - P_w1), P = class-conditional k-NN precision against the 50 real ImageNet validation
                       images per class (real_fidelity.py), DINOv2 space
  recovery             (M - M_w4) / (M_w1 - M_w4) for layout (V_low of the final latents), DINO and CLIP mean pairwise distance,
                       and the Vendi score on DINO and CLIP features (vdino, vclip)
"vs cond curve" is a condition's recovery minus the curve through w4, cond_0.5, cond_0.75 and cond_1 (piecewise linear in
precision retention), read off at the condition's own precision retention: positive means more diversity than condition
damping at the same fidelity. Outside the curve's range the nearest end point is used, so values for conditions with precision
retention below cond_1's are not meaningful. "a - b" rows are paired differences, with precision retention last.
Intervals are 95% percentile intervals over 1000 resamples. Needs guidance_intervene_eval.py --embed for every condition.
usage: paired_vs_cond.py <run> <seed_lo> <seed_hi> cond ... [-- a:b ...]     (from the project root)
"""
import os, sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from common import LABELS
from real_fidelity import knn_radius
from guidance_analysis import pairwise_cos_distance, vendi
from guidance_window import band_split, spread_v

run = Path(sys.argv[1]); seeds = range(int(sys.argv[2]), int(sys.argv[3]) + 1)
rest = sys.argv[4:]
conds = rest[:rest.index("--")] if "--" in rest else rest
pairs = [p.split(":") for p in rest[rest.index("--") + 1:]] if "--" in rest else []
C = ("w4", "cond_0.5", "cond_0.75", "cond_1")  # cond_0 is w4: the curve starts at (precision 1, recovery 0)
names = sorted({"w1", "w4", *C, *conds, *[x for p in pairs for x in p]})
real = np.load(os.environ.get("REAL", "outputs/real_imagenet_val") + "/dino.npz"); R = {c: real["dino"][real["labels"] == c].astype(np.float64) for c in LABELS}
E = {}
for c in names:
    E[c] = {k: np.stack([np.load(run / "embeddings" / c / f"seed_{s}.npz")[k] for s in seeds]).astype(np.float64) for k in ("dino", "clip")}
    E[c]["low"] = band_split(np.stack([np.load(run / c / f"seed_{s}.npz")["final"] for s in seeds]).astype(np.float64))[0]


RADIUS = {l: knn_radius(R[l]) for l in LABELS}  # real-image k-NN radii; precision needs no generated-side radius


def precision(label, gen):
    d = np.linalg.norm(gen[:, None] - R[label][None], axis=-1)
    return float((d <= RADIUS[label][None]).any(1).mean())


# A sample's precision depends only on the real images, not on the other generated samples, so whether each sample lies inside
# its class's real manifold is computed once; the bootstrap then only averages these flags.
INSIDE = {c: np.array([[precision(l, E[c]["dino"][s:s + 1, i]) for i, l in enumerate(LABELS)] for s in range(len(seeds))]) for c in names}


def m(c, idx):
    e = E[c]
    return dict(prec=INSIDE[c][idx].mean(),
                dino=np.mean([pairwise_cos_distance(e["dino"][idx, i]) for i in range(8)]),
                clip=np.mean([pairwise_cos_distance(e["clip"][idx, i]) for i in range(8)]),
                vdino=np.mean([vendi(e["dino"][idx, i]) for i in range(8)]),
                vclip=np.mean([vendi(e["clip"][idx, i]) for i in range(8)]),
                lay=float(spread_v(np.moveaxis(e["low"][idx], 0, 1)).mean()))


K = ("lay", "dino", "clip", "vdino", "vclip")  # vdino / vclip: Vendi score on DINOv2 / CLIP features
rng = np.random.default_rng(0)
vs_curve = {c: [] for c in conds}; vs_pair = {f"{a} - {b}": [] for a, b in pairs}
for _ in range(1000):
    idx = np.unique(rng.choice(len(seeds), len(seeds))); M = {c: m(c, idx) for c in names}
    ret = lambda c: (M[c]["prec"] - M["w1"]["prec"]) / (M["w4"]["prec"] - M["w1"]["prec"])
    rec = lambda c, k: (M[c][k] - M["w4"][k]) / (M["w1"][k] - M["w4"][k])
    xs = [ret(x) for x in C][::-1]
    for c in conds:
        vs_curve[c].append([rec(c, k) - np.interp(ret(c), xs, [rec(x, k) for x in C][::-1]) for k in K])
    for a, b in pairs:
        vs_pair[f"{a} - {b}"].append([rec(a, k) - rec(b, k) for k in K] + [ret(a) - ret(b)])
fmt = lambda v: f"{v.mean():+.3f} [{np.percentile(v, 2.5):+.3f},{np.percentile(v, 97.5):+.3f}]"
for c, v in vs_curve.items():
    v = np.array(v); print(f"{c:30} vs cond curve:", ", ".join(f"{k} {fmt(v[:, i])}" for i, k in enumerate(K)), flush=True)
for p, v in vs_pair.items():
    v = np.array(v); print(f"{p:45}", ", ".join(f"{k} {fmt(v[:, i])}" for i, k in enumerate(K + ("precRet",))), flush=True)
