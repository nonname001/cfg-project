"""Decode x_hat at every step, plus the final latents, and store DINOv2 and CLIP global embeddings for each image.

Writes <run>/analysis/embeddings/w<w>_seed_<n>.npz with dino (50, 8, 768) and clip (50, 8, 512): rows 0-48 are x_hat at each
model call and row 49 is the final sample. Decoded finals go to <run>/analysis/samples/ as grids. Finished files are skipped.
usage: guidance_embed.py --run <dir> [--seeds ...] [--ws 4 1] [--device mps]

Why embeddings: distances between raw latents mostly reflect pixel-level differences. DINOv2 (self-supervised) and CLIP
(trained on image-text pairs) map each image to a vector where nearby means "looks alike" or "means the same thing". Spread
across seeds in those spaces is therefore a measure of semantic diversity. Decoding x_hat at every step shows when during
sampling the image's content is decided.
"""
import argparse, json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

DINO = "vit_base_patch14_dinov2.lvd142m"  # timm model names; weights download from Hugging Face on first use
CLIP = "vit_base_patch16_clip_224.openai"
VAE = "stabilityai/sd-vae-ft-mse"  # the autoencoder SiT was trained with; it turns 4x32x32 latents into 3x256x256 images
LATENT_SCALE = 0.18215  # same constant as sample.py: SiT works on VAE latents multiplied by this, so decoding divides by it
SIZE = 224  # both encoders expect 224 x 224 inputs
BATCH = 8  # one batch is the 8 classes of one seed (not referenced below; kept for documentation)


def encoder(name, device, **kwargs):
    """Load a pretrained timm model. Returns (model, mean, std): the per-channel normalization that model was trained with."""
    model = timm.create_model(name, pretrained=True, **kwargs).to(device).eval()
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    return model, mean, std


def embed(enc, images):
    """images: (B, 3, 224, 224) in [0, 1]. Returns the model's global embedding."""
    model, mean, std = enc
    return model((images - mean) / std).float().cpu().numpy()


def main():
    """For each (seed, w): decode every latent, save the final grid, and write that batch's embeddings."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    p.add_argument("--ws", type=float, nargs="+", default=[4.0, 1.0])
    p.add_argument("--device", default="mps")
    p.add_argument("--final-only", action="store_true", help="embed only the final samples; arrays then have one row")
    a = p.parse_args()
    torch.set_grad_enabled(False)

    run = Path(a.run)
    out, samples = run / "analysis" / "embeddings", run / "analysis" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    samples.mkdir(exist_ok=True)
    vae = AutoencoderKL.from_pretrained(VAE).to(a.device).eval()
    # img_size=224 resamples DINOv2's position embeddings from its native 518, giving 16x16 patches of size 14.
    dino = encoder(DINO, a.device, num_classes=0, img_size=SIZE)
    clip = encoder(CLIP, a.device)  # the default head is CLIP's learned projection, so this is the 512-d image embedding
    # Record exactly which models produced the embeddings, so later analyses can be traced back to them.
    info = dict(vae=VAE, dino=DINO, clip=CLIP, image_size=SIZE, timm=timm.__version__, rows="0-48 x_hat per model call, 49 final",
                dino_hub=dino[0].pretrained_cfg.get("hf_hub_id"), clip_hub=clip[0].pretrained_cfg.get("hf_hub_id"))
    (out / "embeddings.json").write_text(json.dumps(info, indent=2))

    for seed in a.seeds:
        for w in a.ws:
            path = out / f"w{w:g}_seed_{seed}.npz"
            if path.exists():
                continue
            d = np.load(run / f"w{w:g}" / f"seed_{seed}" / "trajectory.npz")
            latents = d["final"][None] if a.final_only else np.concatenate([d["x_hat"], d["final"][None]], 0)  # (rows, 8, 4, 32, 32)
            rows = dict(dino=[], clip=[])
            for k in range(len(latents)):  # one row at a time: the 8 images of one step
                z = torch.from_numpy(latents[k]).to(a.device)
                # The VAE decodes to pixel values in [-1, 1]; (x + 1) / 2 maps them to [0, 1], and clamp trims any overshoot.
                image = ((vae.decode(z / LATENT_SCALE).sample + 1) / 2).clamp(0, 1)
                if k == len(latents) - 1:  # the last row is the final sample: also save it as a 2 x 4 image grid
                    save_image(image.cpu(), samples / f"w{w:g}_seed_{seed}.png", nrow=4)
                # Resize 256 -> 224 for the encoders. antialias avoids aliasing when downsampling.
                small = F.interpolate(image, size=SIZE, mode="bicubic", antialias=True).clamp(0, 1)
                rows["dino"].append(embed(dino, small))
                rows["clip"].append(embed(clip, small))
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **{k: np.stack(v) for k, v in rows.items()})
            tmp.rename(path)  # a file at the final path is always complete
            print(f"w={w:g} seed={seed} done", flush=True)


if __name__ == "__main__":
    main()
