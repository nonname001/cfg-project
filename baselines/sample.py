"""Sample every (method, seed) batch and save final latents plus x_hat at N_RECORD steps.

usage: python sample.py --methods cfg_w1 cfg_w4 interval cads apg fdg score cond --seeds 100-131 [--out runs/main]
Output: <out>/<method spec>/seed_<n>.npz, one batch = the 8 classes in common.LABELS. Finished batches are skipped,
and each file is written to a temporary name first, so a stopped run never leaves a half-written file.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import methods
from common import DEFAULT_WEIGHTS, LABELS, STEPS, load_model, parse_seeds, sample

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True, help="method specs, e.g. cfg_w4 interval:lo=0.4")
    p.add_argument("--seeds", nargs="+", default=["100-131"], help="e.g. 100-131 or 100 101 102")
    p.add_argument("--out", default=str(HERE / "runs" / "main"))
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--model", default="SiT-XL/2")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    torch.set_grad_enabled(False)
    out, seeds = Path(a.out), parse_seeds(a.seeds)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.methods:
        methods.build(spec)  # fail fast on a bad spec before loading the model
    (out / "manifest.json").write_text(json.dumps(dict(
        weights=Path(a.weights).name, model=a.model, labels=LABELS, steps=STEPS, sampler="euler, own loop (see check.py)",
        torch=torch.__version__, platform=platform.platform()), indent=2))

    model = load_model(a.weights, a.model, a.device)
    for seed in seeds:  # seeds outermost: a stopped run leaves complete method sets for the finished seeds
        for spec in a.methods:
            target = out / methods.folder_name(spec) / f"seed_{seed}.npz"
            if target.exists():
                continue
            target.parent.mkdir(exist_ok=True)
            start = time.perf_counter()
            result = sample(model, methods.build(spec), seed)
            assert np.isfinite(result["final"]).all(), f"non-finite latents: {spec} seed {seed}"
            tmp = target.with_name(target.stem + ".tmp.npz")
            np.savez(tmp, **result, seed=seed, labels=np.array(LABELS))
            tmp.rename(target)
            print(f"{spec:24s} seed {seed}: {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
