"""Sweep all 12 configs. Each (config, engine) gets a fresh process. usage: run_all.py <workdir> <steps>

For each SiT size (S, B, L, XL) and patch size (8, 4, 2): write weights and noise (prep.py), sample with MLX, compiled MLX and
PyTorch MPS (worker.py), decode the MLX result to a PNG (decode.py), and record how closely the MLX and MPS latents agree.
Every result is appended as one JSON line to bench/results.jsonl.
"""
import json, os, subprocess, sys, time
import numpy as np
W, STEPS = sys.argv[1], sys.argv[2]
PY = ".venv/bin/python"
MODELS = [f"SiT-{s}/{p}" for s in ("S", "B", "L", "XL") for p in (8, 4, 2)]
log = open("bench/results.jsonl", "a")

def call(cmd):
    """Run a worker and return the JSON from its last RESULT line, or an error record if it failed or printed none."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    rows = [l[7:] for l in r.stdout.splitlines() if l.startswith("RESULT ")]  # l[7:] strips the "RESULT " prefix
    if r.returncode or not rows:
        return dict(error=(r.stderr or r.stdout)[-400:], returncode=r.returncode, cmd=cmd[2:4])
    return json.loads(rows[-1])

def emit(d):
    """Timestamp a result, append it to the log file (flushed, so a crash loses nothing) and echo it."""
    d["ts"] = time.strftime("%H:%M:%S"); log.write(json.dumps(d) + "\n"); log.flush(); print(json.dumps(d), flush=True)

for name in MODELS:
    tag = name.replace("/", "_"); pre = f"{W}/{tag}"
    subprocess.run([PY, "bench/prep.py", name, pre, "0"], capture_output=True, check=True)
    lat = {}
    for eng, extra, key in (("mlx", [], "mlx"), ("mlx", ["compile"], "mlxc"), ("mps", [], "mps")):
        lat[key] = f"{pre}.{key}.npy"
        emit(call([PY, "bench/worker.py", eng, name, pre, lat[key], STEPS] + extra))
        time.sleep(5)  # let the machine cool slightly between runs
    emit(call([PY, "bench/decode.py", lat["mlx"], f"outputs/{tag}.png", name]))
    if all(os.path.exists(p) for p in lat.values()):
        # Relative RMS difference between the MLX and PyTorch final latents: 0 would mean identical samples.
        a, b = np.load(lat["mlx"]), np.load(lat["mps"])
        emit(dict(model=name, parity_rel_rmse=float((a - b).std() / b.std())))
    os.remove(pre + ".safetensors")  # weights are large; delete each config's copy once it is done
