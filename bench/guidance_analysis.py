"""Numerical analysis of a guidance_capture.py run. Writes CSV tables to <run>/analysis/ for guidance_figures.py to plot.

Conventions:
  * i indexes seeds within a class (N=16). Cross-seed quantities are computed per class, then averaged over the 8 classes.
    The "all" scope instead treats all 128 trajectories as one population, which also counts differences between classes.
  * Features are the conditional branch unless a table's branch column says otherwise.
  * Subspace comparisons remove each class's mean and stack the classes, giving 128 rows and 120 degrees of freedom. N=16 per
    class is too few to estimate a 5-dimensional subspace reliably, especially after splitting seeds in half.

usage: guidance_analysis.py --run <dir> [--stages stats spectrum subspace rotation spread]
"""
import argparse, csv, itertools, json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

WS = (4, 1)
SEEDS = list(range(16))
K = 5  # principal directions compared
HALF_A, HALF_B = SEEDS[:8], SEEDS[8:]
CUTOFFS = (4, 8)  # cycles per image; 32-pixel latents have Nyquist 16
BLUR_SIGMAS = (1.0, 2.0)  # latent pixels
N_SPLITS = 20


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_trajectories(run, w, keys):
    """Returns {key: array} with axes (T, seeds, classes, ...)."""
    files = [np.load(run / f"w{w}" / f"seed_{s}" / "trajectory.npz") for s in SEEDS]
    return {k: np.stack([f[k] for f in files], 1) for k in keys}


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def mean_pairwise_cos(v):
    """v: (N, D) in float64. Mean cosine over distinct pairs, using ||sum of units||^2 = N + sum over ordered pairs."""
    n = len(v)
    total = unit(v).sum(0)
    return float((total @ total - n) / (n * (n - 1)))


def shared_proportion(v):
    """S = ||mean_i v_i||^2 / mean_i ||v_i||^2. Equals 1 when all v_i match, and about 1/N for independent zero-mean v_i."""
    return float(np.sum(v.mean(0) ** 2) / np.mean(np.sum(v ** 2, axis=1)))


def stats_stage(run, out):
    """Magnitude, direction stability, cross-seed alignment and shared proportions."""
    per_step, alignment = [], []
    for w in WS:
        d = load_trajectories(run, w, ("s_t", "delta", "f_cond", "f_uncond", "norm", "cos"))
        t = d["s_t"][:, 0]
        T = len(t)
        flat = {k: d[k].reshape(T, 16, 8, -1).astype(np.float64) for k in ("delta", "f_cond", "f_uncond")}
        ratio = np.linalg.norm(flat["delta"], axis=-1) / np.linalg.norm(flat["f_uncond"], axis=-1)
        for k in range(T):
            n, r = d["norm"][k].ravel(), ratio[k].ravel()
            row = dict(w=w, step=k, t=t[k], norm_mean=n.mean(), norm_p10=np.percentile(n, 10), norm_p90=np.percentile(n, 90),
                       ratio_mean=r.mean(), ratio_p10=np.percentile(r, 10), ratio_p90=np.percentile(r, 90))
            c = d["cos"][k - 1].ravel() if k else None
            row.update(cos_mean=c.mean() if k else "", cos_p10=np.percentile(c, 10) if k else "",
                       cos_p90=np.percentile(c, 90) if k else "")
            per_step.append(row)

            row = dict(w=w, step=k, t=t[k])
            for scope, groups in (("class", [flat_k[:, c] for flat_k in [flat["delta"][k]] for c in range(8)]),
                                  ("all", [flat["delta"][k].reshape(128, -1)])):
                row[f"pairwise_cos_delta_{scope}"] = np.mean([mean_pairwise_cos(g) for g in groups])
                vals = [mean_pairwise_cos(g) for g in groups]
                row[f"pairwise_cos_delta_{scope}_min"], row[f"pairwise_cos_delta_{scope}_max"] = min(vals), max(vals)
            for name in ("delta", "f_cond", "f_uncond"):
                per_class = [shared_proportion(flat[name][k][:, c]) for c in range(8)]
                row[f"S_{name}_class"] = np.mean(per_class)
                row[f"S_{name}_class_min"], row[f"S_{name}_class_max"] = min(per_class), max(per_class)
                row[f"S_{name}_all"] = shared_proportion(flat[name][k].reshape(128, -1))
            alignment.append(row)
    write_csv(out / "magnitude_direction.csv", per_step)
    write_csv(out / "alignment_shared.csv", alignment)


def spectrum_stage(run, out):
    """Radial power spectrum of Delta per channel, and the Gaussian low/high split."""
    fy, fx = np.meshgrid(np.fft.fftfreq(32, 1 / 32), np.fft.fftfreq(32, 1 / 32), indexing="ij")
    radius = np.hypot(fy, fx)  # cycles per 32-pixel image
    bins = np.rint(radius).astype(int)
    n_bins = bins.max() + 1
    energy_rows, radial_rows = [], []
    for w in WS:
        d = load_trajectories(run, w, ("s_t", "delta"))
        t, delta = d["s_t"][:, 0], d["delta"].astype(np.float64)  # (T, 16, 8, 4, 32, 32)
        # With norm="ortho", Parseval makes the sum of |F|^2 equal the sum of squares, so the bins partition ||Delta||^2 exactly.
        power = np.abs(np.fft.fft2(delta, axes=(-2, -1), norm="ortho")) ** 2
        total = (delta ** 2).sum((-2, -1))
        assert np.allclose(power.sum((-2, -1)), total, rtol=1e-9)
        for k in range(len(t)):
            for ch in ["all", 0, 1, 2, 3]:
                sel = slice(None) if ch == "all" else slice(ch, ch + 1)
                p, e = power[k][:, :, sel].sum(2), total[k][:, :, sel].sum(2)  # sum channels: (16, 8, 32, 32) and (16, 8)
                row = dict(w=w, step=k, t=t[k], channel=ch, energy=e.mean())
                for cutoff in CUTOFFS:
                    low = p[..., radius <= cutoff].sum(-1)
                    row[f"low_{cutoff}"] = low.mean()
                    row[f"high_{cutoff}"] = (e - low).mean()
                    row[f"high_fraction_{cutoff}"] = ((e - low) / e).mean()
                if ch == "all":
                    for sigma in BLUR_SIGMAS:
                        # mode="wrap" treats the latent as periodic, matching the FFT above.
                        blurred = gaussian_filter(delta[k], sigma=(0, 0, 0, sigma, sigma), mode="wrap")
                        low_e, high_e = (blurred ** 2).sum((2, 3, 4)), ((delta[k] - blurred) ** 2).sum((2, 3, 4))
                        all_e = (delta[k] ** 2).sum((2, 3, 4))
                        # The two parts are not orthogonal, so these fractions need not sum to 1.
                        row[f"blur_low_fraction_s{sigma:g}"] = (low_e / all_e).mean()
                        row[f"blur_high_fraction_s{sigma:g}"] = (high_e / all_e).mean()
                    radial = np.bincount(bins.ravel(), weights=p.mean((0, 1)).ravel(), minlength=n_bins)
                    radial_rows += [dict(w=w, step=k, t=t[k], radius=r, power=radial[r]) for r in range(n_bins)]
                energy_rows.append(row)
    write_csv(out / "spectrum_energy.csv", energy_rows)
    write_csv(out / "spectrum_radial.csv", radial_rows)


def svd_basis(x):
    """x: (rows, D) already centered. Returns singular values and right singular vectors, using the small Gram matrix."""
    x = x.astype(np.float64)
    vals, vecs = np.linalg.eigh(x @ x.T)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]
    keep = vals > vals[0] * 1e-12
    sigma = np.sqrt(vals[keep])
    return sigma, (vecs[:, keep] / sigma).T @ x


def effective_rank(sigma):
    if len(sigma) == 0:
        return 0.0
    p = sigma ** 2 / np.sum(sigma ** 2)
    return float(np.exp(-np.sum(p * np.log(p))))


def overlap(a, b):
    """a, b: (k, D) orthonormal rows. Returns the cosines of the principal angles, largest first."""
    return np.clip(np.linalg.svd(a @ b.T, compute_uv=False), 0, 1)


def class_centered(x):
    """x: (seeds, 8, D). Removes each class's mean over the given seeds and stacks to (seeds*8, D)."""
    return (x - x.mean(0, keepdims=True)).reshape(-1, x.shape[-1])


def compare_feature(x1, x4):
    """x1, x4: (16, 8, D) features at w=1 and w=4, paired by seed. Returns one dict of subspace statistics."""
    row = {}
    # User-specified version: N=16 seeds per class, averaged over classes.
    ranks = {1: [], 4: []}
    variance = {1: [], 4: []}
    angles_class = []
    for c in range(8):
        basis = {}
        for w, x in ((1, x1), (4, x4)):
            m = x[:, c] - x[:, c].mean(0)
            sigma, v = svd_basis(m)
            ranks[w].append(effective_rank(sigma))
            variance[w].append(np.sum(sigma ** 2) / 15)
            basis[w] = v
        k = min(K, len(basis[1]), len(basis[4]))
        if k == 0:
            continue
        angles_class.append(overlap(basis[1][:k], basis[4][:k]))
    row.update(eff_rank_w1=np.mean(ranks[1]), eff_rank_w4=np.mean(ranks[4]),
               variance_ratio=np.mean(np.array(variance[4]) / np.maximum(np.array(variance[1]), 1e-30)),
               class_mean_cos_angle=float(np.mean(angles_class)) if angles_class else 1.0,
               class_max_angle_deg=float(np.degrees(np.arccos(np.mean(angles_class, axis=0)[-1]))) if angles_class else 0.0)

    # Class-centered population: 128 rows.
    sigma1, v1 = svd_basis(class_centered(x1))
    sigma4, v4 = svd_basis(class_centered(x4))
    row.update(pooled_eff_rank_w1=effective_rank(sigma1), pooled_eff_rank_w4=effective_rank(sigma4))
    if min(len(v1), len(v4)) < K:  # step 0 at low rank, or identical features
        return row

    cosines = overlap(v1[:K], v4[:K])
    row.update(pooled_mean_cos=float(cosines.mean()), pooled_largest_angle_deg=float(np.degrees(np.arccos(cosines[-1]))))

    # Rotation test. Each subspace is estimated from 8 seeds, so even an unrotated subspace overlaps imperfectly.
    # same_w gives that ceiling: same w, different noise. cross_w is the same comparison across w, so cross_w ~ same_w means
    # guidance did not rotate the subspace. paired shares noise as well as seeds, so it is inflated by construction.
    bases = {(w, h): svd_basis(class_centered(x[idx]))[1][:K]
             for w, x in ((1, x1), (4, x4)) for h, idx in (("A", HALF_A), ("B", HALF_B))}
    energy = lambda a, b: float(np.mean(overlap(bases[a], bases[b]) ** 2))
    row.update(same_w=np.mean([energy((1, "A"), (1, "B")), energy((4, "A"), (4, "B"))]),
               cross_w=np.mean([energy((1, "A"), (4, "B")), energy((1, "B"), (4, "A"))]),
               paired=np.mean([energy((1, "A"), (4, "A")), energy((1, "B"), (4, "B"))]))

    # Shrinkage test, cross-fitted. Directions chosen on w=1 half A maximize that half's variance, so measuring them on the same
    # data would overstate the shrinkage. Measure them on the held-out half at both w, then swap halves and average.
    ratios = []
    for fit, held in ((HALF_A, HALF_B), (HALF_B, HALF_A)):
        v = svd_basis(class_centered(x1[fit]))[1][:K]
        var1 = ((class_centered(x1[held]) @ v.T) ** 2).sum(0)
        var4 = ((class_centered(x4[held]) @ v.T) ** 2).sum(0)
        ratios.append(var4 / np.maximum(var1, 1e-30))
    ratios = np.mean(ratios, axis=0)
    row.update({f"shrink_pc{i + 1}": ratios[i] for i in range(K)})
    row["total_variance_ratio"] = float(np.sum(class_centered(x4) ** 2) / np.sum(class_centered(x1) ** 2))
    return row


def subspace_stage(run, out):
    manifest = json.loads((run / "manifest.json").read_text())
    pooled = {w: [np.load(run / f"w{w}" / f"seed_{s}" / "pooled.npy", mmap_mode="r") for s in SEEDS] for w in WS}
    t = np.load(run / "w4" / "seed_0" / "trajectory.npz")["s_t"]

    rows = []
    for branch, name in ((0, "cond"), (1, "uncond")):
        for layer in range(24):
            series = {w: np.stack([np.asarray(p[:, layer, branch]) for p in pooled[w]], 0) for w in WS}  # (16, T, 8, 1024)
            for k in range(len(t)):
                rows.append(dict(kind="pooled", branch=name, layer=layer, step=k, t=t[k],
                                 **compare_feature(series[1][:, k], series[4][:, k])))
        print(f"pooled {name} done", flush=True)
    write_csv(out / "subspace_pooled.csv", [fill(r, rows) for r in rows])

    tokens = {w: [np.load(run / f"w{w}" / f"seed_{s}" / "tokens.npy", mmap_mode="r") for s in SEEDS] for w in WS}
    rows = []
    for slot, step in enumerate(manifest["token_steps"]):
        for j, layer in enumerate(manifest["token_layers_zero_indexed"]):
            for branch, name in ((0, "cond"), (1, "uncond")):
                x = {w: np.stack([np.asarray(p[slot, j, branch]).reshape(8, -1) for p in tokens[w]], 0) for w in WS}
                rows.append(dict(kind="tokens", branch=name, layer=layer, step=step, t=t[step], **compare_feature(x[1], x[4])))
        print(f"tokens step {step} done", flush=True)
    write_csv(out / "subspace_tokens.csv", [fill(r, rows) for r in rows])


def rotation_stage(run, out):
    """Repeat the rotation test over random half-splits of the seeds, and save the leading w=1 directions.

    compare_feature uses one fixed split. The same_w - cross_w gap is small, so it needs checking across splits. The splits reuse
    the same 16 seeds, so their SD shows sensitivity to the split rather than a full sampling error.
    """
    manifest = json.loads((run / "manifest.json").read_text())
    pooled = [{w: np.load(run / f"w{w}" / f"seed_{s}" / "pooled.npy", mmap_mode="r") for w in WS} for s in SEEDS]
    t = np.load(run / "w4" / "seed_0" / "trajectory.npz")["s_t"]
    rng = np.random.default_rng(0)
    splits = [rng.permutation(16) for _ in range(N_SPLITS)]
    rows, directions = [], {}
    for layer in range(24):
        series = {w: np.stack([np.asarray(p[w][:, layer, 0]) for p in pooled], 0) for w in WS}  # (16, T, 8, 1024)
        for k in range(len(t)):
            x = {w: series[w][:, k] for w in WS}
            same, cross, paired = [], [], []
            for order in splits:
                a, b = order[:8], order[8:]
                basis = {(w, h): svd_basis(class_centered(x[w][idx]))[1][:K] for w in WS for h, idx in (("A", a), ("B", b))}
                energy = lambda p_, q: float(np.mean(overlap(basis[p_], basis[q]) ** 2))
                same.append(np.mean([energy((1, "A"), (1, "B")), energy((4, "A"), (4, "B"))]))
                cross.append(np.mean([energy((1, "A"), (4, "B")), energy((1, "B"), (4, "A"))]))
                paired.append(np.mean([energy((1, "A"), (4, "A")), energy((1, "B"), (4, "B"))]))
            gap = np.array(same) - np.array(cross)
            rows.append(dict(layer=layer, step=k, t=t[k], same_w=np.mean(same), cross_w=np.mean(cross), paired=np.mean(paired),
                             gap=gap.mean(), gap_sd=gap.std(ddof=1), gap_positive_share=float(np.mean(gap > 0))))
            if layer in manifest["token_layers_zero_indexed"] and k in manifest["token_steps"]:
                sigma, v = svd_basis(class_centered(x[1]))
                directions[f"block{layer}_step{k}"] = v[:K].astype(np.float32)
        print(f"rotation block {layer} done", flush=True)
    write_csv(out / "subspace_rotation_splits.csv", rows)
    np.savez(out / "principal_directions_w1.npz", **directions)


def fill(row, rows):
    """Rows that stopped early (identical features at step 0) lack some columns. Blank them so every row has the same fields."""
    fields = max(rows, key=len).keys()
    return {k: row.get(k, "") for k in fields}


def pairwise_distances(x, metric):
    """x: (N, D). Mean over distinct pairs of L2 or cosine distance."""
    i, j = np.triu_indices(len(x), 1)
    if metric == "l2":
        return float(np.linalg.norm(x[i] - x[j], axis=1).mean())
    u = unit(x)
    return float((1 - (u[i] * u[j]).sum(1)).mean())


def spread_stage(run, out):
    """Mean pairwise distance across seeds, per class, at every step and for the final samples."""
    latents = {w: load_trajectories(run, w, ("s_t", "x_hat", "final")) for w in WS}
    emb = {w: [np.load(run / "analysis" / "embeddings" / f"w{w}_seed_{s}.npz") for s in SEEDS] for w in WS}
    emb = {w: {k: np.stack([f[k] for f in files], 1) for k in ("dino", "clip")} for w, files in emb.items()}  # (50, 16, 8, D)
    t = latents[4]["s_t"][:, 0]

    def per_class(x, metric):
        return np.array([pairwise_distances(x[:, c].reshape(16, -1).astype(np.float64), metric) for c in range(8)])

    rows = []
    for k in range(len(t)):
        spreads = {("l2_latent", w): per_class(latents[w]["x_hat"][k], "l2") for w in WS}
        # L2 grows with latent scale, and guidance changes the scale, so also measure spread with scale removed.
        spreads.update({("cos_latent", w): per_class(latents[w]["x_hat"][k], "cos") for w in WS})
        spreads.update({(name, w): per_class(emb[w][name][k], "cos") for name in ("dino", "clip") for w in WS})
        for name in ("l2_latent", "cos_latent", "dino", "clip"):
            ratio = spreads[(name, 4)] / spreads[(name, 1)]
            rows.append(dict(metric=name, step=k, t=t[k], latent_rms_w1=np.sqrt((latents[1]["x_hat"][k].astype(np.float64) ** 2).mean()),
                             latent_rms_w4=np.sqrt((latents[4]["x_hat"][k].astype(np.float64) ** 2).mean()),
                             spread_w1=spreads[(name, 1)].mean(), spread_w4=spreads[(name, 4)].mean(),
                             ratio_of_means=spreads[(name, 4)].mean() / spreads[(name, 1)].mean(), ratio_class_min=ratio.min(),
                             ratio_class_max=ratio.max()))
    write_csv(out / "spread_over_time.csv", rows)

    names = json.loads((run / "manifest.json").read_text())["class_names"]
    rows = []
    rng = np.random.default_rng(0)
    for name in ("dino", "clip"):
        final = {w: emb[w][name][-1].astype(np.float64) for w in WS}  # (16, 8, D)
        per = {w: np.array([pairwise_distances(final[w][:, c], "cos") for c in range(8)]) for w in WS}
        for c, label in enumerate(names):
            rows.append(dict(metric=name, class_name=label, spread_w1=per[1][c], spread_w4=per[4][c], ratio=per[4][c] / per[1][c]))
        # Bootstrap over seeds, resampling the same seeds at both w to keep the pairing. Pairs of a resampled seed with itself are
        # dropped, because their zero distance would bias the spread down.
        boots = []
        for _ in range(2000):
            idx = rng.choice(16, 16)
            pick = np.unique(idx)
            r = [pairwise_distances(final[4][pick, c], "cos") / pairwise_distances(final[1][pick, c], "cos") for c in range(8)]
            boots.append(np.mean(r))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(metric=name, class_name="ALL (mean of classes)", spread_w1=per[1].mean(), spread_w4=per[4].mean(),
                         ratio=np.mean(per[4] / per[1]), ci_low=lo, ci_high=hi))
    write_csv(out / "final_diversity.csv", [fill(r, rows) for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--stages", nargs="+", default=["stats", "spectrum", "subspace", "rotation", "spread"])
    a = p.parse_args()
    run = Path(a.run)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    stages = dict(stats=stats_stage, spectrum=spectrum_stage, subspace=subspace_stage, rotation=rotation_stage, spread=spread_stage)
    for name in a.stages:
        stages[name](run, out)
        print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
