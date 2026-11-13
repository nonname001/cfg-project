"""One-forward timing of a SiT config at sample.py's effective batch (16 = 8 + CFG) for ONE engine, with parity vs CPU fp32.
usage: engine_probe.py <engine: cpu|mps|mlx> <dtype: float32|float16|bfloat16> [model]"""
import sys, time, statistics as st
sys.path.insert(0, "sit"); sys.path.insert(0, "bench")
import numpy as np, torch
from models import SiT_models

eng, dts = sys.argv[1], sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else "SiT-XL/2"
torch.manual_seed(0); torch.set_grad_enabled(False)
m = SiT_models[name](input_size=32, num_classes=1000, learn_sigma=True).eval()
for k, v in m.named_parameters():  # un-zero adaLN/final so parity exercises every block
    if "adaLN_modulation" in k or "final_layer.linear" in k:
        torch.nn.init.normal_(v, std=0.02)
sd = {k: v.clone() for k, v in m.state_dict().items()}
x = torch.randn(16, 4, 32, 32); t = torch.rand(16); y = torch.randint(0, 1000, (16,))
ref = m(x, t, y).numpy()

def bench(fn, sync, n):
    fn(); sync(); ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - a)
    return st.median(ts) * 1e3

if eng == "cpu":
    ms, out = bench(lambda: m(x, t, y), lambda: None, 3), ref
elif eng == "mps":
    dt = getattr(torch, dts); md = m.to("mps", dt)
    xd, td, yd = x.to("mps", dt), t.to("mps", dt), y.to("mps")
    ms = bench(lambda: md(xd, td, yd), torch.mps.synchronize, 5)
    out = md(xd, td, yd).float().cpu().numpy()
else:
    import mlx.core as mx
    from sit_mlx import SiTMLX
    mm = SiTMLX(sd, name, dtype=getattr(mx, dts))
    xm, tm, ym = mx.array(x.numpy()), mx.array(t.numpy()), mx.array(y.numpy())
    def f():
        o = mm(xm, tm, ym); mx.eval(o); return o
    ms = bench(f, lambda: None, 5); out = np.array(f())
print(f"{eng} {dts:9s} {ms:9.1f} ms/fwd  maxerr {np.abs(out-ref).max():.2e}  (ref absmax {np.abs(ref).max():.2f})")
