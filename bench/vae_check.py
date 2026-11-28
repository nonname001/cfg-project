"""Could part of the diversity loss come from the VAE decoder rather than SiT?

(a) Real round trip: the 50 ImageNet validation images of each class, preprocessed as for SiT's training, embedded directly and
    after VAE encode (posterior mean) + decode. If the VAE compressed semantic diversity, the reconstructions would be less
    diverse than the originals.
(b) Latent scale: guided latents are larger (RESULTS.md 6.4). The w = 4 final latents rescaled by one global factor to w = 1's RMS,
    and w = 1's scaled to w = 4's, then decoded. If diversity tracks the scale, part of the measured loss is how the decoder
    responds to larger latents rather than a change in content.
Diversity: mean pairwise cosine distance and Vendi (DINOv2 and CLIP), per class, averaged over the 8 classes; images go through
guidance_embed.decode_and_embed exactly like every generated sample.
usage: vae_check.py   (from the project root; seeds 16-47 of SiT-L_2_night for (b))
"""
import io, sys
sys.path.insert(0, "bench")
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from common import LABELS
from guidance_analysis import pairwise_cos_distance, vendi
from guidance_embed import LATENT_SCALE, SIZE, decode_and_embed, encode, load_models
from real_fidelity import center_crop_arr

torch.set_grad_enabled(False)
dev = "mps"
vae, dino, clip = load_models(dev)
div = lambda E: (np.mean([pairwise_cos_distance(e) for e in E]), np.mean([vendi(e) for e in E]))  # E: list of (N, D) per class


def report(name, D, C):
    d, v = div(D); c, vc = div(C)
    print(f"{name:34} DINO {d:.3f}  Vendi {v:5.2f}   CLIP {c:.3f}  Vendi {vc:5.2f}", flush=True)


# (a) real round trip
real = Path("outputs/real_imagenet_val")
orig_d, orig_c, rec_d, rec_c, sim, rms = [], [], [], [], [], []
for c in LABELS:
    files = sorted((real / str(c)).glob("*"))
    arr = np.stack([center_crop_arr(Image.open(io.BytesIO(f.read_bytes())).convert("RGB")) for f in files])
    od, oc, rd, rc = [], [], [], []
    for i in range(0, len(arr), 10):
        image = torch.from_numpy(arr[i:i + 10]).permute(0, 3, 1, 2).float().div(255).to(dev)
        small = F.interpolate(image, size=SIZE, mode="bicubic", antialias=True).clamp(0, 1)
        od.append(encode(dino, small).cpu().numpy()); oc.append(encode(clip, small).cpu().numpy())
        z = vae.encode(image * 2 - 1).latent_dist.mean * LATENT_SCALE
        rms.append(float(z.pow(2).mean().sqrt()))
        _, _, d, cc = decode_and_embed(vae, dino, clip, z)
        rd.append(d); rc.append(cc)
    od, oc, rd, rc = (np.concatenate(x).astype(np.float64) for x in (od, oc, rd, rc))
    orig_d.append(od); orig_c.append(oc); rec_d.append(rd); rec_c.append(rc)
    u = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    sim.append((u(od) * u(rd)).sum(1).mean())
print("(a) real ImageNet validation images, 50 per class")
report("original", orig_d, orig_c)
report("VAE encode + decode", rec_d, rec_c)
print(f"    DINO cosine between each image and its reconstruction: {np.mean(sim):.3f}; real latent RMS {np.mean(rms):.3f}")

# (b) latent scale
run = Path("outputs/guidance_delta/SiT-L_2_night")
Z = {w: np.stack([np.load(run / w / f"seed_{s}.npz")["final"] for s in range(16, 48)]).astype(np.float32) for w in ("w1", "w4")}  # (32, 8, 4, 32, 32)
R = {w: float(np.sqrt((Z[w] ** 2).mean())) for w in Z}
print(f"\n(b) generated final latents, seeds 16-47: RMS w1 {R['w1']:.3f}, w4 {R['w4']:.3f}")


def decoded(z):
    D, C = [], []
    for s in range(len(z)):  # one seed (8 classes) per decode
        _, _, d, c = decode_and_embed(vae, dino, clip, torch.from_numpy(z[s]).to(dev))
        D.append(d); C.append(c)
    D, C = np.stack(D).astype(np.float64), np.stack(C).astype(np.float64)  # (seeds, 8, dim)
    return [D[:, i] for i in range(8)], [C[:, i] for i in range(8)]


for name, z in (("w1", Z["w1"]), ("w1 scaled up to w4's RMS", Z["w1"] * R["w4"] / R["w1"]),
                ("w4", Z["w4"]), ("w4 scaled down to w1's RMS", Z["w4"] * R["w1"] / R["w4"])):
    report(name, *decoded(z))
