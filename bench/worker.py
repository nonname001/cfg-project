"""Sample one config on one engine in a fresh process. Prints a JSON line; saves final latents.
usage: worker.py <mlx|mps> <model> <prefix> <latents_out> <steps> [compile]

Part of the older speed benchmark, not the guidance experiments. Engines:
  mps   PyTorch on the Apple GPU, through SiT's own Sampler (fixed-step Euler)
  mlx   the MLX port in sit_mlx.py, with a hand-written Euler loop over the same time grid; `compile` traces it with mx.compile
A fresh process per run keeps memory measurements and caches from leaking between engines. run_all.py parses the RESULT line.
Environment: SIT_MLX_DTYPE picks the MLX precision (default float16); SIT_MLX_CPU forces MLX onto the CPU.
"""
import json, sys, time
sys.path.insert(0, "bench")
import os
import numpy as np
import memwatch
eng, name, prefix, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
use_compile = len(sys.argv) > 6 and sys.argv[6] == "compile"
LABELS, CFG, GB = [207, 360, 387, 974, 88, 979, 417, 279], 4.0, 1024 ** 3  # same 8 classes and w as the guidance experiments
z = np.load(prefix + ".noise.npy")  # shared noise written by prep.py, so both engines start from identical inputs
t0 = time.perf_counter()

# Each branch defines sample() (one full sampling run) and warm() (one forward pass, to trigger compilation and allocation first).
if eng == "mlx":
    import mlx.core as mx
    from sit_mlx import SiTMLX
    if os.environ.get("SIT_MLX_CPU"):
        mx.set_default_device(mx.cpu)
    model = SiTMLX(mx.load(prefix + ".safetensors"), name, dtype=getattr(mx, os.environ.get("SIT_MLX_DTYPE", "float16")))
    mx.clear_cache()  # drop the fp32 load buffers so they don't count as sampling memory
    y = mx.array(LABELS + [1000] * 8)  # 8 class labels, then 8 null labels for the unconditional half
    fwd = lambda x, tt: model.forward_with_cfg(x, tt, y, CFG)
    if use_compile:
        fwd = mx.compile(fwd)
    grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    def sample():
        # Euler over the same grid torchdiffeq uses: `steps` points = steps-1 model calls
        x = mx.array(np.concatenate([z, z], 0))
        for a, b in zip(grid[:-1], grid[1:]):
            x = x + float(b - a) * fwd(x, mx.full((16,), float(a)))
            mx.eval(x)  # MLX is lazy; evaluate each step so the graph does not grow across all 49 steps
        return np.array(x[:8])  # conditional rows only: the real samples
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
        # synchronize waits for queued GPU work, so the timer measures the computation rather than just its submission.
        r = fn(zz, model.forward_with_cfg, **kw)[-1][:8].cpu().numpy(); torch.mps.synchronize(); return r
    def warm():
        model.forward_with_cfg(zz, torch.full((16,), .5, device="mps"), **kw); torch.mps.synchronize()

load_s = time.perf_counter() - t0
base = memwatch.footprint()  # memory after loading, before sampling
w = memwatch.Watch().start()  # samples memory on a background thread to find the peak during sampling
warm()
times = []
for i in range(3):  # up to 3 timed runs; the median is reported
    a = time.perf_counter(); lat = sample(); times.append(time.perf_counter() - a)
    if times[0] > 30:   # expensive configs: single timed run
        break
peak = w.stop()
np.save(out, lat)
# nfe = number of function evaluations (model calls); s_per_nfe makes runs with different step counts comparable.
res = dict(engine=eng, model=name, steps=steps, nfe=steps - 1, compile=use_compile, load_s=round(load_s, 2),
           sample_s=round(float(np.median(times)), 3), runs=[round(t, 3) for t in times],
           s_per_nfe=round(float(np.median(times)) / (steps - 1), 4), loaded_gb=round(base / GB, 3),
           peak_gb=round(peak / GB, 3), lifetime_peak_gb=round(memwatch.lifetime_peak() / GB, 3),
           finite=bool(np.isfinite(lat).all()), lat_std=round(float(lat.std()), 4))
# Each framework's own memory counter, as a cross-check on the process-level footprint.
if eng == "mlx":
    res["mlx_peak_gb"] = round(mx.get_peak_memory() / GB, 3)
else:
    res["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / GB, 3)
print("RESULT " + json.dumps(res), flush=True)
