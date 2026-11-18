"""Numerical agreement between engines: is the MLX port as accurate as PyTorch?

1. A single large matrix multiply at a real SiT shape, on each backend, against a float64 reference. This separates the
   hardware's arithmetic precision from any bug in the port.
2. The full SiT-XL/2 forward pass with the official weights: the MLX port on CPU and GPU, at fp32, fp16 and bf16, against
   PyTorch on CPU in fp32.
usage: parity.py   (no arguments; needs pretrained_models/SiT-XL-2-256x256.pt)
"""
import sys
sys.path.insert(0, "sit"); sys.path.insert(0, "bench")
import numpy as np, torch, mlx.core as mx
from models import SiT_models
from sit_mlx import SiTMLX
torch.set_grad_enabled(False); torch.manual_seed(0)

# 1. raw matmul accuracy vs float64, real SiT shape
# (4096, 1152) @ (1152, 4608): 16 images x 256 tokens, XL width 1152, times the 4x-wide MLP layer.
a = np.random.default_rng(0).standard_normal((4096, 1152)).astype(np.float32)
b = np.random.default_rng(1).standard_normal((1152, 4608)).astype(np.float32)
truth = a.astype(np.float64) @ b.astype(np.float64)
def rel(o): return np.abs(o - truth).max() / np.abs(truth).max()  # largest error relative to the largest value
print(f"matmul relerr  mlx-gpu fp32 {rel(np.array(mx.matmul(mx.array(a), mx.array(b), stream=mx.gpu))):.2e}"
      f" | mlx-cpu fp32 {rel(np.array(mx.matmul(mx.array(a), mx.array(b), stream=mx.cpu))):.2e}"
      f" | torch-mps fp32 {rel((torch.from_numpy(a).to('mps') @ torch.from_numpy(b).to('mps')).cpu().numpy()):.2e}"
      f" | torch-cpu fp32 {rel((torch.from_numpy(a) @ torch.from_numpy(b)).numpy()):.2e}")

# 2. real XL/2 weights, torch CPU fp32 reference vs port on each MLX backend
sd = torch.load("pretrained_models/SiT-XL-2-256x256.pt", map_location="cpu", weights_only=True)
m = SiT_models["SiT-XL/2"](input_size=32, num_classes=1000, learn_sigma=True).eval(); m.load_state_dict(sd)
# 4 inputs at different times and labels, including the null class 1000.
x = torch.randn(4, 4, 32, 32); t = torch.tensor([.05, .3, .6, .95]); y = torch.tensor([207, 360, 1000, 88])
# SiT.forward already drops the sigma channels, so the else branch is the one taken; the check is defensive.
ref = m(x, t, y)[:, :4].numpy() if m(x, t, y).shape[1] > 4 else m(x, t, y).numpy()
print("ref shape", ref.shape, "absmax", np.abs(ref).max(), "rms", ref.std())
xm, tm, ym = mx.array(x.numpy()), mx.array(t.numpy()), mx.array(y.numpy())
for dev, dt in ((mx.cpu, mx.float32), (mx.gpu, mx.float32), (mx.gpu, mx.float16), (mx.gpu, mx.bfloat16)):
    mx.set_default_device(dev)
    mm = SiTMLX(sd, "SiT-XL/2", dtype=dt)
    o = np.array(mm(mx.array(x.numpy()), mx.array(t.numpy()), mx.array(y.numpy())))
    d = o - ref
    # rel-rmse: RMS error divided by the output's RMS, a scale-free accuracy figure comparable across precisions.
    print(f"{dev} {dt}: maxerr {np.abs(d).max():.2e} rmse {d.std():.2e} rel-rmse {d.std()/ref.std():.2e}", flush=True)
