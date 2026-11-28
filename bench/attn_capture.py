"""Does guidance change where SiT attends? Attention summaries at every step, w = 1 and w = 4 (open direction 6).

SiT conditions on the class only through adaLN modulation, so the class can change attention only indirectly: through the
modulated input to each attention layer (same x_t, conditional vs unconditional row) and through the trajectory (different x_t
at w = 4 than at w = 1). A forward pre-hook on every block's attention recomputes its weights from its input (the module itself
uses fused attention and never materializes them) and keeps, per block, head and step:
  class_l1   mean over queries of the L1 distance between the conditional and unconditional rows' attention distributions
             (0 = the class does not change where this head looks; 2 = disjoint)
  entropy    attention entropy / log(256), conditional rows (1 = uniform)
  distance   mean distance in patches from each query to what it attends to, conditional rows
  received   attention each of the 256 positions receives (column mean), conditional rows: a saliency map per head, saved at
             every RECEIVED_EVERY-th step in float16 for the cross-seed analysis in attn_analysis.py
The hooks do not change the computation; --check compares the final latents with the saved run's.
usage: attn_capture.py --run <16-seed run> --weights <file> --out <dir> --seeds 0 ... [--ws 1 4]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from common import LABELS, NULL_CLASS, initial_noise, load_model
from transport import Sampler, create_transport

RECEIVED_EVERY = 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--weights", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True); p.add_argument("--ws", type=float, nargs="+", default=[1.0, 4.0])
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    dev, n = a.device, len(LABELS)
    model = load_model("SiT-L/2", a.weights, dev)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=50, atol=1e-6, rtol=1e-3, reverse=False)
    nb, H = len(model.blocks), model.blocks[0].attn.num_heads
    side = int(model.x_embedder.num_patches ** 0.5)
    yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([yy.flatten(), xx.flatten()], -1).float()
    dist = torch.cdist(pos, pos).to(dev)  # (256, 256) patch distances
    out = Path(a.out)
    for w in a.ws:
        for seed in a.seeds:
            target = out / f"w{w:g}" / f"seed_{seed}.npz"
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            state = dict(k=0)
            logs = {k: np.zeros((49, nb, n, H), np.float32) for k in ("class_l1", "entropy", "distance")}
            received = np.zeros((len(range(0, 49, RECEIVED_EVERY)), nb, n, H, side * side), np.float16)

            def t_hook(_m, inputs):
                state["k"] = int(round(float(inputs[0][0]) * 49))

            def attn_hook(b):
                def hook(m, inputs):
                    x = inputs[0]
                    B, N, C = x.shape
                    qkv = m.qkv(x).reshape(B, N, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
                    q, kk = m.q_norm(qkv[0]), m.k_norm(qkv[1])
                    att = ((q * m.scale) @ kk.transpose(-2, -1)).float().softmax(-1)  # (2n, H, N, N)
                    k = state["k"]
                    c, u = att[:n], att[n:]
                    logs["class_l1"][k, b] = (c - u).abs().sum(-1).mean(-1).cpu().numpy()
                    logs["entropy"][k, b] = (-(c * (c + 1e-12).log()).sum(-1).mean(-1) / np.log(N)).cpu().numpy()
                    logs["distance"][k, b] = (c * dist).sum(-1).mean(-1).cpu().numpy()
                    if k % RECEIVED_EVERY == 0:
                        received[k // RECEIVED_EVERY, b] = c.mean(-2).half().cpu().numpy()
                return hook

            handles = [model.t_embedder.register_forward_pre_hook(t_hook)]
            handles += [blk.attn.register_forward_pre_hook(attn_hook(b)) for b, blk in enumerate(model.blocks)]
            start = time.perf_counter()
            try:
                z = initial_noise(seed)
                x = torch.cat([z, z], 0).to(dev)
                y = torch.tensor(LABELS + [NULL_CLASS] * n, device=dev)
                final = sample_fn(x, model.forward_with_cfg, y=y, cfg_scale=w)[-1][:n].float().cpu().numpy()
            finally:
                for h in handles:
                    h.remove()
            saved = np.load(Path(a.run) / f"w{w:g}" / f"seed_{seed}" / "trajectory.npz")["final"]
            tmp = target.with_suffix(".tmp.npz")
            np.savez(tmp, seed=seed, w=w, final=final, received=received, received_steps=np.arange(0, 49, RECEIVED_EVERY), **logs)
            tmp.rename(target)
            print(f"w{w:g} seed={seed}: {time.perf_counter() - start:.1f}s, max |final - saved| = {np.abs(final - saved).max():.1e}", flush=True)
            assert np.abs(final - saved).max() < 1e-4, "the attention hooks changed the samples"


if __name__ == "__main__":
    main()
