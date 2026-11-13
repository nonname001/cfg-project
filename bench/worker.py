"""Sample one config on one engine in a fresh process. Prints a JSON line; saves final latents.
usage: worker.py <mlx|mps> <model> <prefix> <latents_out> <steps> [compile]"""
import json, sys, time
sys.path.insert(0, "bench")
import os
import numpy as np
import memwatch
eng, name, prefix, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
use_compile = len(sys.argv) > 6 and sys.argv[6] == "compile"
LABELS, CFG, GB = [207, 360, 387, 974, 88, 979, 417, 279], 4.0, 1024 ** 3
z = np.load(prefix + ".noise.npy")
t0 = time.perf_counter()

if eng == "mlx":
    import mlx.core as mx
    from sit_mlx import SiTMLX
    if os.environ.get("SIT_MLX_CPU"):
        mx.set_default_device(mx.cpu)
    model = SiTMLX(mx.load(prefix + ".safetensors"), name, dtype=getattr(mx, os.environ.get("SIT_MLX_DTYPE", "float16")))
    mx.clear_cache()  # drop the fp32 load buffers so they don't count as sampling memory
    y = mx.array(LABELS + [1000] * 8)
    fwd = lambda x, tt: model.forward_with_cfg(x, tt, y, CFG)
    if use_compile:
        fwd = mx.compile(fwd)
    grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    def sample():
        # Euler over the same grid torchdiffeq uses: `steps` points = steps-1 model calls
        x = mx.array(np.concatenate([z, z], 0))
        for a, b in zip(grid[:-1], grid[1:]):
            x = x + float(b - a) * fwd(x, mx.full((16,), float(a)))
            mx.eval(x)
        return np.array(x[:8])
    def warm():
        mx.eval(fwd(mx.array(np.concatenate([z, z], 0)), mx.full((16,), 0.5)))
    mx.reset_peak_memory()
else:
    import torch
    sys.path.insert(0, "sit")
    from safetensors.torch import load_file
    from models import SiT_models
    from transport import create_transport, Sampler
    torch.set_grad_enabled(False)
    model = SiT_models[name](input_size=32, num_classes=1000, learn_sigma=True).to("mps")
    model.load_state_dict(load_file(prefix + ".safetensors", device="mps")); model.eval()
    fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=steps, atol=1e-6, rtol=1e-3, reverse=False)
    y = torch.tensor(LABELS + [1000] * 8, device="mps")
    zz = torch.from_numpy(np.concatenate([z, z], 0)).to("mps")
    kw = dict(y=y, cfg_scale=CFG)
    def sample():
        r = fn(zz, model.forward_with_cfg, **kw)[-1][:8].cpu().numpy(); torch.mps.synchronize(); return r
    def warm():
        model.forward_with_cfg(zz, torch.full((16,), .5, device="mps"), **kw); torch.mps.synchronize()

load_s = time.perf_counter() - t0
base = memwatch.footprint()
w = memwatch.Watch().start()
warm()
times = []
for i in range(3):
    a = time.perf_counter(); lat = sample(); times.append(time.perf_counter() - a)
    if times[0] > 30:   # expensive configs: single timed run
        break
peak = w.stop()
np.save(out, lat)
res = dict(engine=eng, model=name, steps=steps, nfe=steps - 1, compile=use_compile, load_s=round(load_s, 2),
           sample_s=round(float(np.median(times)), 3), runs=[round(t, 3) for t in times],
           s_per_nfe=round(float(np.median(times)) / (steps - 1), 4), loaded_gb=round(base / GB, 3),
           peak_gb=round(peak / GB, 3), lifetime_peak_gb=round(memwatch.lifetime_peak() / GB, 3),
           finite=bool(np.isfinite(lat).all()), lat_std=round(float(lat.std()), 4))
if eng == "mlx":
    res["mlx_peak_gb"] = round(mx.get_peak_memory() / GB, 3)
else:
    res["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / GB, 3)
print("RESULT " + json.dumps(res), flush=True)
