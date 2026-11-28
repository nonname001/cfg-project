"""Progress summary for the SDE sampler test: each finished SDE condition next to the ODE run of the same condition, same seeds.

Per condition, seeds 16-31: precision against the real images (REAL, default the 1,300-image reference), DINO mean pairwise
distance, Vendi (DINO), layout spread V_low. Once both samplers have w1 and w4: DINO recovery (0 = w4, 1 = w1) and precision
retention (1 = w4, 0 = w1), each relative to that sampler's own w1 and w4. Point values only; the paired test with intervals is
paired_vs_cond.py once every condition is done.
usage: sde_update.py   (from the project root)
"""
import os, sys
sys.path.insert(0, "bench")
import numpy as np
from pathlib import Path
from common import LABELS
from guidance_analysis import pairwise_cos_distance, vendi
from guidance_window import band_split, spread_v
from real_fidelity import knn_radius

O = Path("outputs/guidance_delta"); SEEDS = range(16, 32)
RUNS = dict(ode=O / "SiT-L_2_night", sde=O / "SiT-L_2_sde")
CONDS = ["w1", "w4", "score_1", "cond_0.5", "cond_0.75", "cond_1"]
real = np.load(os.environ.get("REAL", "outputs/real_imagenet_train") + "/dino.npz")
R = {l: real["dino"][real["labels"] == l].astype(np.float64) for l in LABELS}; RAD = {l: knn_radius(R[l]) for l in LABELS}


def stats(run, c):
    files = [run / "embeddings" / c / f"seed_{s}.npz" for s in SEEDS]
    if not all(f.exists() for f in files):
        return None
    D = np.stack([np.load(f)["dino"] for f in files]).astype(np.float64)
    low = band_split(np.stack([np.load(run / c / f"seed_{s}.npz")["final"] for s in SEEDS]).astype(np.float64))[0]
    prec = np.mean([(np.linalg.norm(D[:, i, None] - R[l][None], axis=-1) <= RAD[l][None]).any(1).mean() for i, l in enumerate(LABELS)])
    return dict(prec=prec, dino=np.mean([pairwise_cos_distance(D[:, i]) for i in range(8)]),
                vendi=np.mean([vendi(D[:, i]) for i in range(8)]), layout=float(spread_v(np.moveaxis(low, 0, 1)).mean()))


S = {k: {c: stats(r, c) for c in CONDS} for k, r in RUNS.items()}
print(f"seeds 16-31; precision vs {sum(len(r) for r in R.values())} real images")
print(f"{'':10}  {'ODE: prec  DINO  Vendi layout':32}  {'SDE: prec  DINO  Vendi layout':32}")
for c in CONDS:
    if S["sde"][c] is None:
        continue
    f = lambda m: f"{m['prec']:.2f}  {m['dino']:.3f}  {m['vendi']:4.2f}  {m['layout']:.3f}" if m else "-"
    print(f"{c:10}  {f(S['ode'][c]):32}  {f(S['sde'][c]):32}")
if all(S[k][w] for k in S for w in ("w1", "w4")):
    print("\nrelative to each sampler's own w1 / w4:      ODE: prec. ret.  DINO rec.   SDE: prec. ret.  DINO rec.")
    for c in CONDS[2:]:
        if S["sde"][c] is None:
            continue
        rel = lambda k: ((S[k][c]["prec"] - S[k]["w1"]["prec"]) / (S[k]["w4"]["prec"] - S[k]["w1"]["prec"]),
                         (S[k][c]["dino"] - S[k]["w4"]["dino"]) / (S[k]["w1"]["dino"] - S[k]["w4"]["dino"]))
        (po, do), (ps, ds) = rel("ode"), rel("sde")
        print(f"{c:10}                                        {po:5.2f}       {do:5.2f}          {ps:5.2f}       {ds:5.2f}")
