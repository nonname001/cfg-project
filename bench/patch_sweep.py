"""Where inside SiT is the guided semantic outcome carried? Patching between the w = 1 and w = 4 runs of one seed.

Why not plain activation patching: at a given step the w = 1 and w = 4 runs evaluate the same network with the same time and
class embeddings; guidance only mixes the two branch outputs afterwards. The runs differ only through x_t. Replacing block b's
full output (the residual stream) with the donor's therefore makes every later block, and the output, equal the donor's,
whatever b is. So this patches parts instead:

  write_b<b>_k<k>   for steps k .. k+K-1, block b's own contribution to the residual stream (its output minus its input: the
                    gated attention + MLP writes) is replaced by the donor's, computed by the donor run on its own x_t. Every other
                    block computes on the recipient's state. Both branches are patched, each with the donor's same branch.
  vel_k<k>          for the same steps, the whole network output is the donor's (equivalent to patching any full residual
                    stream): the upper bound for what a K-step intervention at k can transfer
  band_<b>_k<k>     at step k, the <b> = low (<= 4 cycles) / mid (4-8) / high (> 8) frequency band of x_t is replaced by the
                    donor's; sampling then continues unpatched: does the decision at time k sit in coarse layout or finer detail?
  lowshared_k<k>    at step k, only the shared part of the low band is swapped: the class mean over the other seeds of the
                    16-seed run (leave-one-out), i.e. x_low += mean_donor - mean_recipient
  lowspec_k<k>      only the seed-specific part: x_low = mean_recipient + (x_low_donor - mean_donor). Both runs of a seed start
                    from the same noise, so this transfers only the seed-specific part of what guidance changed
Directions: r1 = recipient w = 1 (sampled at w = 1), donor w = 4 ("does this part carry the guided outcome?"); r4 = recipient
w = 4, donor w = 1 ("can this part undo it?"). Condition folder names are <direction>_<mode>. The recipient restarts from its
saved state at step k (switch_sweep.restart, exact); donor states come from the saved 16-seed run.
Also writes w1/ and w4/ (the saved final latents) for the analysis. Read with patch_analysis.py.
usage: patch_sweep.py --run <16-seed run> --weights <file> --out <dir> --seeds 0 1 [--ks 5 12 20] [--blocks 4 10 ...]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from common import LABELS, NULL_CLASS, array_sha256, load_model
from transport import Sampler, create_transport
from switch_sweep import save

K = 5  # steps patched per window (~0.1 of the trajectory)
BANDS = dict(low=(-1, 4), mid=(4, 8), high=(8, 99))  # radius in cycles per 32-pixel latent: lo < r <= hi


def band_mask(name, device):
    f = torch.fft.fftfreq(32, 1 / 32)
    r = torch.hypot(*torch.meshgrid(f, f, indexing="ij"))
    lo, hi = BANDS[name]
    return ((r > lo) & (r <= hi)).to(device)


def band_part(x, mask):
    return torch.fft.ifft2(torch.fft.fft2(x) * mask).real


def run(model, sample_fn, base, donor, k, w, mode, device, means=None):
    """Restart the recipient from its saved state at step k under guidance w, with the given patch. Returns final latents."""
    n = len(LABELS)
    y = torch.tensor(LABELS + [NULL_CLASS] * n, device=device)
    x_k = torch.from_numpy(base["x_t"][k]).to(device)
    if mode.startswith("band_"):
        m = band_mask(mode.split("_")[1], device)
        x_k = x_k - band_part(x_k, m) + band_part(torch.from_numpy(donor["x_t"][k]).to(device), m)
    if mode.startswith("lowshared_") or mode.startswith("lowspec_"):
        m = band_mask("low", device)
        mb, md = (torch.from_numpy(v).to(device) for v in means)  # leave-one-out low-band class means: recipient, donor
        xd = band_part(torch.from_numpy(donor["x_t"][k]).to(device), m)
        x_k = x_k + (md - mb) if mode.startswith("lowshared_") else x_k - band_part(x_k, m) + mb + (xd - md)
    state = dict(write=None, out=None)
    patch_block = int(mode.split("_b")[1].split("_")[0]) if mode.startswith("write_") else None
    window = range(k, k + K) if mode.startswith(("write_", "vel_")) else range(0)
    rec = {}

    def rec_hook(_m, inputs, out):
        rec["write"] = out - inputs[0]

    def rec_final(_m, _i, out):
        rec["out"] = out

    def patch_hook(_m, inputs, out):
        return inputs[0] + state["write"] if state["write"] is not None else None

    def patch_final(_m, _i, out):
        return state["out"] if state["out"] is not None else None

    def fn(xb, tb, y, cfg_scale):
        j = int(round(float(tb[0]) * 49))
        state["write"] = state["out"] = None
        if j in window:
            # Donor pass on the donor's own saved state at this step (same grid time, same labels), recording the patch.
            xd = torch.from_numpy(np.concatenate([donor["x_t"][j]] * 2)).to(device)
            h = (model.blocks[patch_block].register_forward_hook(rec_hook) if patch_block is not None
                 else model.final_layer.register_forward_hook(rec_final))
            try:
                model.forward(xd, tb, y)
            finally:
                h.remove()
            state.update(rec)
        h = (model.blocks[patch_block].register_forward_hook(patch_hook) if patch_block is not None
             else model.final_layer.register_forward_hook(patch_final))
        try:
            return model.forward_with_cfg(xb, tb, y, cfg_scale)
        finally:
            h.remove()

    solver = sample_fn.__self__
    full = solver.t
    solver.t = full[k:]
    try:
        x = torch.cat([x_k, x_k], 0)
        return sample_fn(x, fn, y=y, cfg_scale=w)[-1][:n].float().cpu().numpy()
    finally:
        solver.t = full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--weights", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 12, 20])
    p.add_argument("--blocks", type=int, nargs="+", default=[4, 10, 13, 15, 17, 20, 23])
    p.add_argument("--directions", nargs="+", default=["r1", "r4"])
    p.add_argument("--split", action="store_true", help="also the lowshared / lowspec modes")
    p.add_argument("--split-only", action="store_true", help="only the lowshared / lowspec modes")
    p.add_argument("--check", action="store_true", help="vel and write patches with donor = recipient must reproduce the recipient")
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    torch.set_grad_enabled(False)
    run_dir, out = Path(a.run), Path(a.out)
    model = load_model("SiT-L/2", a.weights, a.device)
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=50, atol=1e-6, rtol=1e-3, reverse=False)
    if a.check:
        t = np.load(run_dir / "w4" / f"seed_{a.seeds[0]}" / "trajectory.npz")
        worst = 0.0
        for mode in ("vel_k12", "write_b15_k12", "band_low_k12"):
            final = run(model, sample_fn, t, t, 12, 4.0, mode, a.device)
            d = np.abs(final - t["final"]).max(); worst = max(worst, d)
            print(f"check {mode} with donor = recipient: max |final - saved| = {d:.2e}", flush=True)
        sys.exit(0 if worst < 1e-4 else 1)
    for seed in a.seeds:
        traj = {w: np.load(run_dir / w / f"seed_{seed}" / "trajectory.npz") for w in ("w1", "w4")}
        for w in ("w1", "w4"):
            target = out / w / f"seed_{seed}.npz"
            if not target.exists():
                save(target, seed=seed, final=traj[w]["final"], final_sha256=array_sha256(traj[w]["final"]))
        for k in a.ks:
            modes = [] if a.split_only else [f"write_b{b}_k{k}" for b in a.blocks] + [f"vel_k{k}"] + [f"band_{b}_k{k}" for b in BANDS]
            if a.split or a.split_only:
                modes += [f"lowshared_k{k}", f"lowspec_k{k}"]
                low = band_mask("low", "cpu")
                others = [s for s in range(16) if s != seed]
                lm = {w: band_part(torch.from_numpy(np.stack([np.load(run_dir / w / f"seed_{s}" / "trajectory.npz")["x_t"][k] for s in others])), low).mean(0).numpy()
                      for w in ("w1", "w4")}
            for d in a.directions:
                base, donor, w = (traj["w1"], traj["w4"], 1.0) if d == "r1" else (traj["w4"], traj["w1"], 4.0)
                for mode in modes:
                    target = out / f"{d}_{mode}" / f"seed_{seed}.npz"
                    if target.exists():
                        continue
                    start = time.perf_counter()
                    means = None
                    if mode.startswith("low"):
                        means = (lm["w1"], lm["w4"]) if d == "r1" else (lm["w4"], lm["w1"])
                    final = run(model, sample_fn, base, donor, k, w, mode, a.device, means)
                    assert np.isfinite(final).all()
                    save(target, seed=seed, final=final, final_sha256=array_sha256(final), seconds=time.perf_counter() - start)
                    print(f"{d}_{mode} seed={seed}: {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
