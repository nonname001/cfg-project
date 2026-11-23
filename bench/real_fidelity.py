"""Fidelity against real ImageNet images: class-conditional precision and recall, and FD_DINOv2, for every condition of a run.

Real images are the 50 ImageNet-1k validation images of each of the 8 classes, from the gated Hugging Face dataset
ILSVRC/imagenet-1k (accept its terms and run `hf auth login` first). They are preprocessed the way SiT's training images were
(ADM's center_crop_arr: resize the short side to 256, center-crop 256) and then embedded exactly as generated images are
(guidance_embed.py: 256 -> 224 bicubic, DINOv2 CLS embedding), so both sides go through the same path.

Metrics, all in DINOv2 space:
  precision  Kynkaenniemi et al. (2019), k = 3, per class: share of generated images that fall within the k-NN radius of at least
             one real image of their class. The fidelity measure: it does not reward diversity.
  recall     the converse, per class: share of real images within the k-NN radius (among that class's generated images) of a
             generated image. With 16 generated images per class it is coarse; reported as a coverage check.
  fd         Frechet distance between all generated (seeds x 8 classes) and all 400 real embeddings. Mixes fidelity and
             diversity, and is biased upward at this sample size, so compare it only between conditions with the same seeds.
Each metric also gets a retention, (M - M_w1) / (M_w4 - M_w1) (1 = as at w = 4, 0 = as at w = 1), and 95% intervals from
resampling seeds, the same resample for every condition.

usage: real_fidelity.py --fetch --real <dir>               download the 14 validation shards one at a time, keep the 400 images
       real_fidelity.py --embed --real <dir>               DINOv2 embeddings of the kept images -> <dir>/dino.npz
       real_fidelity.py --run <run> --real <dir>           -> <run>/real_fidelity.csv (needs guidance_intervene_eval.py --embed)
"""
import argparse, io, sys, tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "bench")
from common import LABELS
from guidance_analysis import write_csv

K = 3
N_BOOT = 1000
REPO = "ILSVRC/imagenet-1k"


def fetch(real):
    """Download each validation shard to a temporary folder, keep the images of LABELS, then delete the shard.

    Memory stays near one row group's matching images: each row group's label column is read first, and the image column only
    for the rows that match. Finished shards are listed in <real>/done_shards.txt, so a stopped fetch resumes where it left off.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("data/validation-"))
    real.mkdir(parents=True, exist_ok=True)
    log = real / "done_shards.txt"
    done = set(log.read_text().split()) if log.exists() else set()
    wanted = pa.array(LABELS)
    for shard in shards:
        if shard in done:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            parquet = pq.ParquetFile(hf_hub_download(REPO, shard, repo_type="dataset", local_dir=tmp))
            for group in range(parquet.metadata.num_row_groups):
                labels = parquet.read_row_group(group, columns=["label"]).column("label")
                keep = pc.is_in(labels, value_set=wanted)
                if not pc.any(keep).as_py():
                    continue
                images = parquet.read_row_group(group, columns=["image"]).column("image").filter(keep)
                for image, label in zip(images.to_pylist(), labels.filter(keep).to_pylist()):
                    folder = real / str(label)
                    folder.mkdir(exist_ok=True)
                    (folder / Path(image["path"]).name).write_bytes(image["bytes"])
                del images
        with log.open("a") as f:
            f.write(shard + "\n")
        print("done", shard, flush=True)
    counts = {c: len(list((real / str(c)).glob("*"))) for c in LABELS}
    print(counts)
    assert all(n == 50 for n in counts.values()), "ImageNet validation has 50 images per class"


def center_crop_arr(image, size=256):
    """ADM's preprocessing, which SiT's training used: halve with a box filter while the short side is >= 2 * size, resize the short
    side to size with bicubic, then center-crop size x size."""
    from PIL import Image
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.BICUBIC)
    arr = np.array(image)
    top, left = (arr.shape[0] - size) // 2, (arr.shape[1] - size) // 2
    return arr[top: top + size, left: left + size]


def embed(real, device="mps"):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from guidance_embed import DINO, SIZE, encode, encoder
    torch.set_grad_enabled(False)
    dino = encoder(DINO, device, num_classes=0, img_size=SIZE)  # as in guidance_embed.load_models
    feats, labels = [], []
    for c in LABELS:
        files = sorted((real / str(c)).glob("*"))
        arr = np.stack([center_crop_arr(Image.open(io.BytesIO(f.read_bytes())).convert("RGB")) for f in files])
        image = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255).to(device)
        # Same 256 -> 224 resize as guidance_embed.decode applies to generated images.
        small = F.interpolate(image, size=SIZE, mode="bicubic", antialias=True).clamp(0, 1)
        feats.append(encode(dino, small).cpu().numpy())
        labels += [c] * len(files)
    np.savez(real / "dino.npz", dino=np.concatenate(feats).astype(np.float32), labels=np.array(labels))


def knn_radius(x, k=K):
    """Distance from each row of x to its k-th nearest other row."""
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    return np.sort(d, axis=1)[:, k]  # column 0 is the point itself


def precision_recall(real, gen):
    """real: (R, D), gen: (G, D), one class. Kynkaenniemi precision and recall with k-NN radii."""
    d = np.linalg.norm(gen[:, None] - real[None], axis=-1)  # (G, R)
    precision = float((d <= knn_radius(real)[None]).any(1).mean())
    recall = float((d <= knn_radius(gen)[:, None]).any(0).mean())
    return precision, recall


def frechet(real_mu, real_cov, gen):
    """Frechet distance between a Gaussian fit to gen and the real one. tr sqrt(C_r C_g) is computed from the small n x n matrix
    X C_r X^T / (n - 1), X = centered gen, which has the same nonzero eigenvalues; this keeps it fast and exact with n << D."""
    mu = gen.mean(0)
    x = gen - mu
    n = len(gen)
    cov_trace = (x ** 2).sum() / (n - 1)
    eig = np.linalg.eigvalsh(x @ real_cov @ x.T / (n - 1))
    return float(((mu - real_mu) ** 2).sum() + np.trace(real_cov) + cov_trace - 2 * np.sqrt(np.clip(eig, 0, None)).sum())


def evaluate(run, real):
    saved = np.load(real / "dino.npz")
    real_feats = {c: saved["dino"][saved["labels"] == c].astype(np.float64) for c in LABELS}
    all_real = saved["dino"].astype(np.float64)
    real_mu, real_cov = all_real.mean(0), np.cov(all_real, rowvar=False)
    # Same loading rule as guidance_intervene_eval.load: only seeds finished (and embedded) in every condition.
    conditions = sorted(p.name for p in (run / "embeddings").iterdir() if p.is_dir())
    seed_sets = [{int(p.stem.split("_")[1]) for p in (run / "embeddings" / c).glob("seed_*.npz")} for c in conditions]
    seeds = sorted(set.intersection(*seed_sets))
    gen = {c: np.stack([np.load(run / "embeddings" / c / f"seed_{s}.npz")["dino"] for s in seeds]).astype(np.float64) for c in conditions}

    def metrics(feats, idx):  # feats: (seeds, 8 classes, D)
        pr = [precision_recall(real_feats[c], feats[idx, i]) for i, c in enumerate(LABELS)]
        return dict(precision=float(np.mean([p for p, _ in pr])), recall=float(np.mean([r for _, r in pr])),
                    fd=frechet(real_mu, real_cov, feats[idx].reshape(-1, feats.shape[-1])))

    every = np.arange(len(seeds))
    point = {c: metrics(f, every) for c, f in gen.items()}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        idx = np.unique(rng.choice(len(seeds), len(seeds)))  # as in guidance_intervene_eval.stats
        boots.append({c: metrics(f, idx) for c, f in gen.items()})
    rows = []
    for c in conditions:
        row = dict(condition=c, n_seeds=len(seeds))
        for key in ("precision", "recall", "fd"):
            row[key] = point[c][key]
            row[f"{key}_low"], row[f"{key}_high"] = np.percentile([b[c][key] for b in boots], [2.5, 97.5])
            ret = lambda m: (m[c][key] - m["w1"][key]) / (m["w4"][key] - m["w1"][key])
            row[f"{key}_retention"] = ret(point)
            row[f"{key}_retention_low"], row[f"{key}_retention_high"] = np.percentile([ret(b) for b in boots], [2.5, 97.5])
        rows.append(row)
    write_csv(run / "real_fidelity.csv", rows)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--embed", action="store_true")
    p.add_argument("--run")
    a = p.parse_args()
    real = Path(a.real)
    if a.fetch:
        fetch(real)
    if a.embed:
        embed(real)
    if a.run:
        for row in evaluate(Path(a.run), real):
            print(row["condition"].ljust(22), {k: round(row[k], 3) for k in ("precision", "precision_retention", "recall", "fd")}, flush=True)


if __name__ == "__main__":
    main()
