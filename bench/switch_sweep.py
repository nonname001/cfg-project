"""When does each sample commit to its w = 4 (or w = 1) outcome? Switch the guidance scale at step k (open direction 2a).

From the saved trajectories of the 16-seed run, take the state x_t at step k of one run and finish sampling at the other scale:
  w4to1_k<k>   steps 0..k-1 at w = 4 (the saved run), steps k..48 at w = 1   (k = 15 is the early-only window of RESULTS.md 9)
  w1to4_k<k>   steps 0..k-1 at w = 1, steps k..48 at w = 4                    (k = 15 is the late-only window)
Restarting uses SiT's own fixed-grid Euler integrator with its time grid cut to t[k:], so the arithmetic is the same as a
full run's; --check confirms it reproduces the saved w = 4 final latents from k = 0 and from k = 20.
Also writes w1/ and w4/ (the saved runs' final latents) so every folder has the same layout as guidance_intervene.py's output
(<out>/<condition>/seed_<n>.npz with "final"), for guidance_intervene_eval.py --embed and switch_analysis.py.
usage: switch_sweep.py --run <16-seed run> --weights <file> --out <dir> --seeds 0 1 ... --ks 3 6 ... [--check]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from common import LABELS, NULL_CLASS, array_sha256, load_model
from transport import Sampler, create_transport


def restart(model, sample_fn, x_k, k, w, device):
    """Integrate from state x_k (8, 4, 32, 32) at grid index k to the end at guidance scale w. Returns the final latents."""
    n = len(LABELS)
    solver = sample_fn.__self__  # SiT's ode object; its grid self.t is linspace(t0, t1, num_steps)
    full = solver.t
    solver.t = full[k:]
    try:
        x = torch.from_numpy(np.concatenate([x_k, x_k], 0)).to(device)
        y = torch.tensor(LABELS + [NULL_CLASS] * n, device=device)
        return sample_fn(x, model.forward_with_cfg, y=y, cfg_scale=w)[-1][:n].float().cpu().numpy()
    finally:
        solver.t = full


def save(target, **arrays):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--ks", type=int, nargs="+", required=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    run, out = Path(a.run), Path(a.out)
    model = load_model("SiT-L/2", a.weights, a.device)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=50, atol=1e-6, rtol=1e-3, reverse=False)
    if a.check:
        t = np.load(run / "w4" / f"seed_{a.seeds[0]}" / "trajectory.npz")
        worst = 0.0
        for k in (0, 20):
            final = restart(model, sample_fn, t["x_t"][k], k, 4.0, a.device)
            worst = max(worst, np.abs(final - t["final"]).max())
            print(f"check k={k}: max |restart - saved final| = {np.abs(final - t['final']).max():.2e}", flush=True)
        sys.exit(0 if worst < 1e-4 else 1)
    for seed in a.seeds:
        traj = {w: np.load(run / w / f"seed_{seed}" / "trajectory.npz") for w in ("w1", "w4")}
        for w in ("w1", "w4"):
            target = out / w / f"seed_{seed}.npz"
            if not target.exists():
                save(target, seed=seed, final=traj[w]["final"], final_sha256=array_sha256(traj[w]["final"]))
        for k in a.ks:
            for name, base, w_after in ((f"w4to1_k{k}", "w4", 1.0), (f"w1to4_k{k}", "w1", 4.0)):
                target = out / name / f"seed_{seed}.npz"
                if target.exists():
                    continue
                start = time.perf_counter()
                final = restart(model, sample_fn, traj[base]["x_t"][k], k, w_after, a.device)
                assert np.isfinite(final).all()
                save(target, seed=seed, k=k, s_t=float(traj[base]["s_t"][k]), final=final, final_sha256=array_sha256(final),
                     seconds=time.perf_counter() - start)
                print(f"{name} seed={seed}: {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
