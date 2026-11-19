"""Decode each batch's final latents and store their DINOv2 and CLIP global embeddings (inputs to report Table 2).

Writes <run>/analysis/embeddings/w<w>_seed_<n>.npz with dino (1, 8, 768) and clip (1, 8, 512). The leading axis of length 1 is
the final sample; readers take row [-1]. Finished files are skipped.
usage: guidance_embed.py --run <dir> [--seeds ...] [--ws 4 1] [--device mps]

Why embeddings: distances between raw latents mostly reflect pixel-level differences. DINOv2 (self-supervised) and CLIP
(trained on image-text pairs) map each image to a vector where nearby means "looks alike" or "means the same thing", so spread
across seeds in those spaces measures semantic diversity.
"""
import argparse, json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

DINO = "vit_base_patch14_dinov2.lvd142m"  # timm model names; weights download from Hugging Face on first use
CLIP = "vit_base_patch16_clip_224.openai"
VAE = "stabilityai/sd-vae-ft-mse"  # the autoencoder SiT was trained with; it turns 4x32x32 latents into 3x256x256 images
LATENT_SCALE = 0.18215  # same constant as sample.py: SiT works on VAE latents multiplied by this, so decoding divides by it
SIZE = 224  # both encoders expect 224 x 224 inputs


def encoder(name, device, **kwargs):
    """Load a pretrained timm model. Returns (model, mean, std): the per-channel normalization that model was trained with."""
    model = timm.create_model(name, pretrained=True, **kwargs).to(device).eval()
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    return model, mean, std


def load_models(device):
    """The VAE and both embedding models, set up identically for every script that measures diversity."""
    vae = AutoencoderKL.from_pretrained(VAE).to(device).eval()
    # img_size=224 resamples DINOv2's position embeddings from its native 518, giving 16x16 patches of size 14.
    dino = encoder(DINO, device, num_classes=0, img_size=SIZE)
    clip = encoder(CLIP, device)  # the default head is CLIP's learned projection, so this is the 512-d image embedding
    return vae, dino, clip


def decode(vae, z):
    """z: (B, 4, 32, 32) SiT latents on the device. Returns the decoded images at 256 x 256 and resized to 224 x 224, in [0, 1]."""
    # SiT's latents are the VAE's latents times LATENT_SCALE, so divide before decoding. The VAE decodes to pixel values in
    # [-1, 1]; (x + 1) / 2 maps them to [0, 1], and clamp trims any overshoot.
    image = ((vae.decode(z / LATENT_SCALE).sample + 1) / 2).clamp(0, 1)
    # Resize 256 -> 224 for the encoders. antialias avoids aliasing when downsampling.
    small = F.interpolate(image, size=SIZE, mode="bicubic", antialias=True).clamp(0, 1)
    return image, small


def encode(enc, images):
    """images: (B, 3, 224, 224) in [0, 1]. Normalize the way the model was trained, run it, and return its output on the device."""
    model, mean, std = enc
    return model((images - mean) / std).float()


def decode_and_embed(vae, dino, clip, z):
    """Decode latents z and embed the images.

    Returns (image at 256px, image at 224px, DINOv2 embeddings (B, 768), CLIP embeddings (B, 512)); the images stay torch tensors on
    the device, the embeddings are numpy arrays. Both diversity analyses (Table 2 via this script, Table 3 via guidance_intervene_eval.py) go through this one function,
    so semantic diversity is measured the same way in both.
    """
    image, small = decode(vae, z)
    return image, small, encode(dino, small).cpu().numpy(), encode(clip, small).cpu().numpy()


def main():
    """For each (seed, w): decode the 8 final latents and write that batch's embeddings."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(16)))
    p.add_argument("--ws", type=float, nargs="+", default=[4.0, 1.0])
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    torch.set_grad_enabled(False)

    run = Path(a.run)
    out = run / "analysis" / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    vae, dino, clip = load_models(a.device)
    # Record exactly which models produced the embeddings, so later analyses can be traced back to them.
    info = dict(vae=VAE, dino=DINO, clip=CLIP, image_size=SIZE, timm=timm.__version__, rows="0 = final sample",
                dino_hub=dino[0].pretrained_cfg.get("hf_hub_id"), clip_hub=clip[0].pretrained_cfg.get("hf_hub_id"))
    (out / "embeddings.json").write_text(json.dumps(info, indent=2))

    for seed in a.seeds:
        for w in a.ws:
            path = out / f"w{w:g}_seed_{seed}.npz"
            if path.exists():
                continue
            z = torch.from_numpy(np.load(run / f"w{w:g}" / f"seed_{seed}" / "trajectory.npz")["final"]).to(a.device)  # (8, 4, 32, 32)
            _, _, dino_emb, clip_emb = decode_and_embed(vae, dino, clip, z)
            tmp = path.with_suffix(".tmp.npz")
            # [None] adds the leading axis of length 1 (the final sample) that guidance_window.py reads as row [-1].
            np.savez(tmp, dino=dino_emb[None], clip=clip_emb[None])
            tmp.rename(path)  # a file at the final path is always complete
            print(f"w={w:g} seed={seed} done", flush=True)


if __name__ == "__main__":
    main()
