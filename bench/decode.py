"""VAE-decode final latents to a PNG grid (same step for all engines). Prints a JSON line.
usage: decode.py <latents.npy> <out.png> <label>

Also times the decode and measures its peak memory, for the benchmark.
"""
import json, sys, time
sys.path.insert(0, "bench")
import numpy as np, torch, memwatch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
torch.set_grad_enabled(False)
lat, out, label = sys.argv[1:4]
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("mps").eval()
z = torch.from_numpy(np.load(lat)).to("mps")
w = memwatch.Watch().start()
# 0.18215 is the SD VAE latent scale: SiT's latents are VAE latents times it, so divide before decoding.
vae.decode(z[:1] / 0.18215).sample; torch.mps.synchronize()  # warm-up on one latent, not timed
a = time.perf_counter(); img = vae.decode(z / 0.18215).sample; torch.mps.synchronize(); dt = time.perf_counter() - a
peak = w.stop()
save_image(img.float().cpu(), out, nrow=4, normalize=True, value_range=(-1, 1))  # map the VAE's [-1, 1] output to [0, 1]
print("RESULT " + json.dumps(dict(model=label, decode_s=round(dt, 3), decode_peak_gb=round(peak / 1024 ** 3, 3),
                                   finite=bool(torch.isfinite(img).all()))), flush=True)
