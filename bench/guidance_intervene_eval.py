"""Evaluate guidance_intervene.py outputs (report Table 3): diversity recovery and fidelity retention per condition.

Diversity: scale-free low-band latent spread V (layout), and DINOv2 and CLIP mean pairwise cosine distance across seeds within a
class (semantics). Fidelity: an ImageNet classifier's probability for the intended class. This is a proxy: FID and
precision/recall need real ImageNet images, which are not used here.

recovery  = (M - M_w4) / (M_w1 - M_w4): how much of the diversity lost at w=4 a condition gets back.
retention = (P - P_w1) / (P_w4 - P_w1): how much of the fidelity gained at w=4 it keeps.
Intervals resample seeds 1000 times, identically across conditions.

An ideal intervention has recovery = 1 (as diverse as w = 1) and retention = 1 (as recognizable as w = 4).
usage: guidance_intervene_eval.py --run <dir> [--embed] [--stats]
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from common import LABELS
from guidance_analysis import pairwise_cos_distance, write_csv
from guidance_window import band_split, spread_v

CLASSIFIER = "vit_base_patch16_224.augreg2_in21k_ft_in1k"  # a timm ViT-B/16 ImageNet-1k classifier, independent of SiT
N_BOOT = 1000


def embed(run, device="mps"):
    """Decode every final latent, save a sample grid, and store DINOv2/CLIP embeddings and the classifier's target probability.

    The sample grids in <run>/samples/ are the images the report's sample figures were assembled from.
    """
    # Imported here so --stats runs without these heavy packages loaded.
    from torchvision.utils import save_image
    from guidance_embed import decode_and_embed, encode, encoder, load_models
    torch.set_grad_enabled(False)
    # The same VAE, DINOv2 and CLIP setup as guidance_embed.py, so diversity here is measured exactly as in Table 2.
    vae, dino, clip = load_models(device)
    classifier = encoder(CLASSIFIER, device)
    labels = torch.tensor(LABELS, device=device)
    for path in sorted(run.glob("*/seed_*.npz")):
        if path.name.endswith(".tmp.npz"):  # an unfinished write from a stopped run
            continue
        target = run / "embeddings" / path.parent.name / path.name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        z = torch.from_numpy(np.load(path)["final"]).to(device)
        image, small, dino_emb, clip_emb = decode_and_embed(vae, dino, clip, z)
        # The fidelity classifier sees the same 224px images as the embedding models.
        prob = encode(classifier, small).softmax(-1)
        rows = torch.arange(len(LABELS), device=device)
        (run / "samples").mkdir(exist_ok=True)
        save_image(image.cpu(), run / "samples" / f"{path.parent.name}_{path.stem}.png", nrow=4)  # 2 x 4 grid of the 8 classes
        tmp = target.with_suffix(".tmp.npz")
        # p_target: the classifier's probability for each sample's intended class, prob[i, LABELS[i]].
        np.savez(tmp, dino=dino_emb, clip=clip_emb, p_target=prob[rows, labels].cpu().numpy())
        tmp.rename(target)
        print("embedded", path.parent.name, path.stem, flush=True)


def load(run):
    """Load every condition, keeping only seeds finished in all of them, so every comparison uses the same seeds."""
    data = {}
    for folder in sorted(run.iterdir()):
        if not folder.is_dir() or folder.name in ("embeddings", "samples", "figures"):
            continue
        seeds = sorted(int(p.stem.split("_")[1]) for p in folder.glob("seed_*.npz") if not p.name.endswith(".tmp.npz"))
        data[folder.name] = seeds
    shared = sorted(set.intersection(*[set(v) for v in data.values()]))
    out = {}
    for condition in data:
        emb = [np.load(run / "embeddings" / condition / f"seed_{s}.npz") for s in shared]
        # Seeds on axis 0, classes on axis 1.
        out[condition] = dict(final=np.stack([np.load(run / condition / f"seed_{s}.npz")["final"] for s in shared], 0).astype(np.float64),
                              dino=np.stack([e["dino"] for e in emb], 0).astype(np.float64), clip=np.stack([e["clip"] for e in emb], 0).astype(np.float64),
                              p_target=np.stack([e["p_target"] for e in emb], 0))
    return out, shared


def metrics(d, idx):
    """Diversity and fidelity of one condition on the seeds idx (positions into the shared seed list)."""
    low, _ = band_split(d["final"][idx])
    # V_low: spread of the coarse (layout) part of the final latents. dino, clip: semantic spread. p_target: fidelity.
    return dict(V_low=float(spread_v(np.moveaxis(low, 0, 1)).mean()),
                dino=float(np.mean([pairwise_cos_distance(d["dino"][idx, c]) for c in range(8)])),
                clip=float(np.mean([pairwise_cos_distance(d["clip"][idx, c]) for c in range(8)])),
                p_target=float(d["p_target"][idx].mean()))


def stats(run):
    """Write intervention_results.csv: every metric, recovery and retention, each with a 95% bootstrap interval."""
    data, seeds = load(run)
    every = np.arange(len(seeds))
    point = {c: metrics(d, every) for c, d in data.items()}
    rng = np.random.default_rng(0)  # fixed seed, so the intervals are reproducible
    boots = []
    for _ in range(N_BOOT):
        # Same resampled seeds for every condition (paired), each drawn seed kept once (no zero-distance self-pairs).
        idx = np.unique(rng.choice(len(seeds), len(seeds)))
        boots.append({c: metrics(d, idx) for c, d in data.items()})
    rows = []
    for c, m in point.items():
        row = dict(condition=c, n_seeds=len(seeds), **m)
        for key in ("V_low", "dino", "clip"):
            # recovery is computed inside each resample too, so its interval includes the noise in the w1 and w4 baselines.
            f = lambda b: (b[c][key] - b["w4"][key]) / (b["w1"][key] - b["w4"][key])
            row[f"recovery_{key}"] = f(point)
            row[f"recovery_{key}_low"], row[f"recovery_{key}_high"] = np.percentile([f(b) for b in boots], [2.5, 97.5])
        f = lambda b: (b[c]["p_target"] - b["w1"]["p_target"]) / (b["w4"]["p_target"] - b["w1"]["p_target"])
        row["retention"] = f(point)
        row["retention_low"], row["retention_high"] = np.percentile([f(b) for b in boots], [2.5, 97.5])
        rows.append(row)
    write_csv(run / "intervention_results.csv", rows)
    return rows


def main():
    """--embed must have run before --stats."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--embed", action="store_true")
    p.add_argument("--stats", action="store_true")
    a = p.parse_args()
    run = Path(a.run)
    if a.embed:
        embed(run)
    if a.stats:
        for row in stats(run):
            print(row["condition"].ljust(13), {k: round(row[k], 3) for k in ("recovery_V_low", "recovery_dino", "recovery_clip", "retention")}, flush=True)


if __name__ == "__main__":
    main()
