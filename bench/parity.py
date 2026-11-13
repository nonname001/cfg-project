import sys
sys.path.insert(0, "sit"); sys.path.insert(0, "bench")
import numpy as np, torch, mlx.core as mx
from models import SiT_models
from sit_mlx import SiTMLX
torch.set_grad_enabled(False); torch.manual_seed(0)

# 1. raw matmul accuracy vs float64, real SiT shape
a = np.random.default_rng(0).standard_normal((4096, 1152)).astype(np.float32)
b = np.random.default_rng(1).standard_normal((1152, 4608)).astype(np.float32)
truth = a.astype(np.float64) @ b.astype(np.float64)
def rel(o): return np.abs(o - truth).max() / np.abs(truth).max()
print(f"matmul relerr  mlx-gpu fp32 {rel(np.array(mx.matmul(mx.array(a), mx.array(b), stream=mx.gpu))):.2e}"
      f" | mlx-cpu fp32 {rel(np.array(mx.matmul(mx.array(a), mx.array(b), stream=mx.cpu))):.2e}"
      f" | torch-mps fp32 {rel((torch.from_numpy(a).to('mps') @ torch.from_numpy(b).to('mps')).cpu().numpy()):.2e}"
      f" | torch-cpu fp32 {rel((torch.from_numpy(a) @ torch.from_numpy(b)).numpy()):.2e}")

# 2. real XL/2 weights, torch CPU fp32 reference vs port on each MLX backend
sd = torch.load("pretrained_models/SiT-XL-2-256x256.pt", map_location="cpu", weights_only=True)
m = SiT_models["SiT-XL/2"](input_size=32, num_classes=1000, learn_sigma=True).eval(); m.load_state_dict(sd)
x = torch.randn(4, 4, 32, 32); t = torch.tensor([.05, .3, .6, .95]); y = torch.tensor([207, 360, 1000, 88])
ref = m(x, t, y)[:, :4].numpy() if m(x, t, y).shape[1] > 4 else m(x, t, y).numpy()
print("ref shape", ref.shape, "absmax", np.abs(ref).max(), "rms", ref.std())
xm, tm, ym = mx.array(x.numpy()), mx.array(t.numpy()), mx.array(y.numpy())
for dev, dt in ((mx.cpu, mx.float32), (mx.gpu, mx.float32), (mx.gpu, mx.float16), (mx.gpu, mx.bfloat16)):
    mx.set_default_device(dev)
    mm = SiTMLX(sd, "SiT-XL/2", dtype=dt)
    o = np.array(mm(mx.array(x.numpy()), mx.array(t.numpy()), mx.array(y.numpy())))
    d = o - ref
    print(f"{dev} {dt}: maxerr {np.abs(d).max():.2e} rmse {d.std():.2e} rel-rmse {d.std()/ref.std():.2e}", flush=True)
