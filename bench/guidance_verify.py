"""Check a guidance_delta.py run. Each check prints one line; the JSON report lands next to the data.

1. rerun     resample one seed with no recording wrapper: tests determinism and that logging did not perturb sampling
2. replay    rebuild every Euler step from the logged Delta and v_uncond: proves the logged Delta drove the sampler
3. cpu       recompute Delta on CPU fp32 at a few steps: bounds MPS numerical error on a difference of two close outputs
4. reference compare velocity and Delta to a second model at the same (x_t, t, y), e.g. official XL/2 vs. unverified L/2

usage: guidance_verify.py --run <dir> [--seed 0] [--reference-model SiT-XL/2 --reference-weights <file>]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "bench")
from guidance_delta import GUIDED_CHANNELS, LABELS, NULL_CLASS, load_model, run_seed
from transport import Sampler, create_transport


def cos_rows(a, b):
    # Cosine per sample: row i of a against row i of b, each flattened to one vector.
    a, b = a.reshape(len(a), -1).astype(np.float64), b.reshape(len(b), -1).astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def branches(model, x, t, device):
    """Conditional and unconditional velocity at x (B, 4, 32, 32) and scalar t."""
    n = len(LABELS)
    # Same layout as run_seed: conditional rows first, then null-class rows, both at the same x.
    xb = torch.from_numpy(np.concatenate([x, x], 0)).to(device)
    y = torch.tensor(LABELS + [NULL_CLASS] * n, device=device)
    out = model.forward(xb, torch.full((2 * n,), float(t), device=device), y).float().cpu().numpy()
    return out[:n], out[n:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference-model")
    p.add_argument("--reference-weights")
    a = p.parse_args()
    torch.set_grad_enabled(False)

    man = json.loads((Path(a.run) / "manifest.json").read_text())
    d = np.load(Path(a.run) / f"seed_{a.seed}.npz")
    t, x, delta, v_u, final = d["t"], d["x"], d["delta"], d["v_uncond"], d["final"]
    report = {}

    model = load_model(man["model"], man["weights"], man["device"])
    sample_fn = Sampler(create_transport("Linear", "velocity", None, None, None)).sample_ode(
        sampling_method="euler", num_steps=man["grid_points"], atol=1e-6, rtol=1e-3, reverse=False)
    # Check 1. record=False leaves model.forward alone, so a bit-identical result shows both that sampling is deterministic here
    # and that the recording wrapper did not change it.
    _, again, _ = run_seed(model, sample_fn, a.seed, man["cfg_scale"], man["device"], record=False)
    report["rerun"] = dict(bit_identical=bool(np.array_equal(again, final)), max_abs_diff=float(np.abs(again - final).max()))
    print("rerun", report["rerun"], flush=True)

    # Check 2. Rebuild the sampler's drift by hand from the logged arrays, mirroring forward_with_cfg:
    # guided channels use v_uncond + s * Delta, and v_uncond + Delta is v_cond.
    drift = v_u + delta  # conditional velocity; the unguided 4th channel keeps this value
    drift[:, :, :GUIDED_CHANNELS] = v_u[:, :, :GUIDED_CHANNELS] + man["cfg_scale"] * delta[:, :, :GUIDED_CHANNELS]
    # float32 to match the th.linspace grid that the ode class builds.
    grid = np.linspace(0, 1, man["grid_points"], dtype=np.float32)
    # x holds the state before each model call, so appending final gives all T+1 states, and each Euler step must satisfy
    # x[k+1] = x[k] + dt * drift[k]. The error should be float32 rounding relative to state_abs_max, nothing more.
    states = np.concatenate([x, final[None]], 0)
    step_err = np.abs(states[:-1] + (grid[1:] - grid[:-1])[:, None, None, None, None] * drift - states[1:]).max()
    report["replay"] = dict(max_abs_step_error=float(step_err), state_abs_max=float(np.abs(states).max()))
    print("replay", report["replay"], flush=True)

    # Check 3. Delta is a difference of two close outputs, so it is the quantity most exposed to MPS rounding. Recompute it on
    # CPU at the first, middle and last steps from the exact logged x_t.
    steps = [0, len(t) // 2, len(t) - 1]
    cpu = load_model(man["model"], man["weights"], "cpu")
    rows = []
    for k in steps:
        c, u = branches(cpu, x[k], t[k], "cpu")
        ref = c - u
        rows.append(dict(step=k, t=float(t[k]), min_cos=float(cos_rows(ref, delta[k]).min()),
                         rel_err=float(np.linalg.norm(ref - delta[k]) / np.linalg.norm(ref))))
    report["cpu"] = rows
    print("cpu", rows, flush=True)
    # Free the CPU copy before a second model is loaded; this machine has 16 GB of unified memory.
    del cpu

    # Check 4. Evaluate the reference at this run's x_t, every 8th step. Those states come from this run's trajectory rather than
    # the reference's own, so agreement may be understated. High cos_v_cond means both models learned the same velocity field.
    # cos_delta is a much harder test, since Delta is a small residual of that field.
    if a.reference_model:
        ref_model = load_model(a.reference_model, a.reference_weights, man["device"])
        rows = []
        for k in range(0, len(t), 8):
            c, u = branches(ref_model, x[k], t[k], man["device"])
            # v_u + delta rebuilds this run's v_cond. norm_ratio is this run's Delta norm over the reference's, across the whole batch.
            rows.append(dict(step=k, t=round(float(t[k]), 3), cos_v_cond=float(cos_rows(c, v_u[k] + delta[k]).mean()),
                             cos_delta=float(cos_rows(c - u, delta[k]).mean()),
                             norm_ratio=float(np.linalg.norm(delta[k]) / np.linalg.norm(c - u))))
            print("reference", rows[-1], flush=True)
        report["reference"] = dict(model=a.reference_model, rows=rows)

    (Path(a.run) / f"verify_seed_{a.seed}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
