# Classifier-free guidance in SiT: measurement code

This project studies **classifier-free guidance (CFG)** in SiT, a class-conditional image generator, by recording
what guidance does at every sampling step and testing ways to change it. The findings are in
[RESULTS.md](RESULTS.md). This file explains the code.

All commands run from the project root with the project's virtual environment, `.venv/bin/python`.

## What this repository contains

This repository holds the code, both documents, every figure, the CSV result tables, and each run's manifest and
per-batch metadata. Clone it with the SiT submodule:

```sh
git clone --recurse-submodules https://github.com/nonname001/cfg-project.git
```

Not included, because of size or licensing (see `.gitignore`):

- **Raw experiment arrays** (`.npz`, `.npy`, about 130 GB). Regenerate them with the commands below. The
  per-batch `metadata.json` files record the SHA-256 of each batch's noise and final latents, so a rerun can be
  checked against them.
- **Model weights.** Download them as described in [Weights](#weights).
- **Decoded sample images and DINOv2/CLIP embeddings**, which are derived from the raw arrays.

The one array that is included is `outputs/guidance_delta/SiT-L_2_16seeds/analysis/intervention_tables.npz`, the
direction tables the interventions need. It is small and lets the intervention experiments run without first
regenerating the 16-seed run.

## Layout

| Path | What it is |
|---|---|
| `sit/` | The upstream SiT repository at commit `cbde832`, **unmodified**. Every script imports its model and sampler from here. |
| `bench/` | All of this project's code (described below). |
| `pretrained_models/` | Model weights (see [Weights](#weights)). |
| `outputs/guidance_delta/` | Every experiment's raw data, tables and figures (see [Outputs](#outputs)). |
| `outputs/*.png` | Sample grids from an earlier speed benchmark (see [Older benchmark scripts](#older-benchmark-scripts)). |

## Weights

| File | Source | SHA-256 |
|---|---|---|
| `pretrained_models/SiT-XL-2-256x256.pt` | Official SiT-XL/2 release | `e645b5cd…ab55aca` |
| `pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors` | Community upload, [BiliSakura/SiT-diffusers](https://huggingface.co/BiliSakura/SiT-diffusers), pinned to revision `2f02393a3079dfa6210c10302cc927f59d2cb142`. The SiT authors only released XL/2. | `fd3c19ec1a7675f5648f5200acf4f1a94ae6a44ff5ff81a1011de8242d12edd6` |

All experiments use SiT-L/2. To fetch it again:

```sh
mkdir -p pretrained_models/SiT-L-2-256_BiliSakura_2f02393
curl -fL -o pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors \
  "https://huggingface.co/BiliSakura/SiT-diffusers/resolve/2f02393a3079dfa6210c10302cc927f59d2cb142/SiT-L-2-256/transformer/diffusion_pytorch_model.safetensors"
shasum -a 256 pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors
```

Only the `.safetensors` weights are used; the repo's own pipeline code (`trust_remote_code`) is never run. RESULTS.md
describes how the weights were checked.

## How the code hooks into SiT without editing it

SiT samples by integrating a velocity field from noise (t = 0) to data (t = 1). With CFG, `model.forward_with_cfg`
runs the network on a doubled batch (8 conditional rows, 8 rows with the "null" class) and mixes the two outputs.
The scripts observe and change this in three ways, none of which edit `sit/`:

1. **Replace `model.forward` on the instance.** `forward_with_cfg` calls `self.forward(...)` directly, so ordinary
   PyTorch forward hooks never see it. A wrapper assigned to `model.forward` records the conditional output
   `f_cond`, the unconditional output `f_uncond` and the input `x_t` at every step, then deletes itself afterwards.
2. **Forward hooks on `model.blocks[i]`** record each transformer block's output (activations), and on
   `model.y_embedder` change the class embedding for the condition-space intervention.
3. **Wrap the function the sampler integrates**, to adjust the guided velocity after the model has run (the
   score-space intervention) or to change w over time (windowed guidance).

Sampling always uses SiT's own `Sampler.sample_ode` with a **fixed-step Euler** solver, 50 grid points, so 49
model calls. SiT's default solver (dopri5) is adaptive, which would make "step k" ill-defined.

## Pipeline

```
guidance_delta.py ── first run: Δ only, 8 seeds, w = 4
        │
guidance_capture.py ── full logging: f_cond, f_uncond, x_t, x̂, activations; any w, window or intervention
        ├── guidance_capture_verify.py  checks every batch
        ├── guidance_embed.py           decodes images, DINOv2 + CLIP embeddings
        ├── guidance_analysis.py        statistics (alignment, rank, subspace, frequency, spread)
        ├── guidance_window.py          scale-free measures, windowed-guidance comparison
        ├── guidance_mech.py            mechanistic analysis of interventions
        └── guidance_figures.py, guidance_mech_figures.py   plots
guidance_intervene.py ── interventions, final images only
        └── guidance_intervene_eval.py  diversity vs fidelity
```

## Scripts

### Sampling

| Script | What it does |
|---|---|
| `guidance_delta.py` | The first experiment. Samples 8 classes per seed at w = 4 and stores Δ = f_cond − f_uncond, x_t, f_uncond and per-step metrics in one `.npz` per seed. Also defines shared helpers (class labels, noise, model loading). |
| `guidance_capture.py` | The main logger. Per batch it stores `trajectory.npz` (s_t, x_t, x̂, f_cond, f_uncond, f_guided, Δ and metrics), `pooled.npy` (every block's output averaged over the 256 image tokens, both branches, every step) and `tokens.npy` (full 256 × 1024 token activations at blocks 4, 12 and 22 at 10 timesteps). Options: several w values (`--ws`), a guidance window (`--window lo hi`), or named conditions including interventions (`--conditions`). Writes each batch to a temporary folder and renames it when complete, so a stopped run never leaves half-written data; rerunning skips finished batches. |
| `guidance_intervene.py` | The intervention experiments, saving final latents only. `--build-tables` fits the per-class directions the score and feature methods need. The `Intervention` class here is also what `guidance_capture.py` uses. Its docstring defines every method. |

### Checking

| Script | What it checks |
|---|---|
| `guidance_verify.py` | For the first run: rerun reproducibility, rebuilding each Euler step from logged values, CPU-vs-GPU agreement, and agreement with a reference model (official XL/2). |
| `guidance_capture_verify.py` | For any capture run: same noise across conditions, Euler replay, x̂ and CFG identities, pooled = token mean, and with `--deep` an independent hand-written forward pass on CPU. Exits non-zero on any failure. |

### Analysis and plotting

| Script | Produces |
|---|---|
| `guidance_embed.py` | Decodes x̂ at every step (or only the final image, `--final-only`) with the SD VAE and embeds it with DINOv2 ViT-B/14 and CLIP ViT-B/16. |
| `guidance_analysis.py` | Stages `stats`, `spectrum`, `subspace`, `rotation`, `spread`: the 16-seed analyses in RESULTS.md parts 6 and 7. |
| `guidance_window.py` | Stages `scale`, `windows`, `paired`: scale-free measures and the windowed-guidance tests (parts 8 and 9). |
| `guidance_intervene_eval.py` | `--embed` (images, embeddings, ImageNet classifier), `--stats` (diversity, fidelity, bootstrap intervals), `--figures` (part 10). |
| `guidance_mech.py` | Stages `curves`, `blocks`, `patch`, `finals`: the mechanistic study (part 11). `patch` reruns the model to split changes into direct and through-x_t parts. |
| `guidance_plot.py`, `guidance_figures.py`, `guidance_mech_figures.py` | Figures. `guidance_figures.py --only N ...` renders a subset; without `--only` it renders all. |

## Reproducing each experiment

`W` is the SiT-L/2 weights path and `SRC` the source string from [Weights](#weights). Run times are on an M-series Mac
on AC power; keep it awake for long runs (`caffeinate -i ...`).

```sh
W=pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors
O=outputs/guidance_delta
```

**First run (8 seeds, about 20 minutes)**

```sh
.venv/bin/python bench/guidance_delta.py --model SiT-L/2 --weights $W --out $O/SiT-L_2 --seeds 0 1 2 3 4 5 6 7 --steps 50 --cfg 4.0
.venv/bin/python bench/guidance_verify.py --run $O/SiT-L_2 --seed 0 --reference-model SiT-XL/2 --reference-weights pretrained_models/SiT-XL-2-256x256.pt
.venv/bin/python bench/guidance_plot.py --run $O/SiT-L_2
```

**w = 4 vs w = 1, 16 seeds (about 1 hour, then about 1.5 hours of embedding)**

```sh
.venv/bin/python bench/guidance_capture.py --weights $W --weights-source "$SRC" --out $O/SiT-L_2_16seeds --seeds $(seq 0 15)
.venv/bin/python bench/guidance_capture_verify.py --run $O/SiT-L_2_16seeds --first-run $O/SiT-L_2 --deep 4 13
.venv/bin/python bench/guidance_embed.py --run $O/SiT-L_2_16seeds
.venv/bin/python bench/guidance_analysis.py --run $O/SiT-L_2_16seeds
.venv/bin/python bench/guidance_figures.py --run $O/SiT-L_2_16seeds --only 1 2 3 4 5 6 7 8 9
```

**Windowed guidance and scale-free measures**

```sh
.venv/bin/python bench/guidance_capture.py --weights $W --weights-source "$SRC" --out $O/SiT-L_2_window_early --seeds $(seq 0 15) --ws 4 --window 0 0.3
.venv/bin/python bench/guidance_capture.py --weights $W --weights-source "$SRC" --out $O/SiT-L_2_window_late  --seeds $(seq 0 15) --ws 4 --window 0.3 1.01
.venv/bin/python bench/guidance_embed.py --run $O/SiT-L_2_window_early --ws 4 --final-only
.venv/bin/python bench/guidance_embed.py --run $O/SiT-L_2_window_late  --ws 4 --final-only
.venv/bin/python bench/guidance_window.py --base $O
.venv/bin/python bench/guidance_figures.py --run $O/SiT-L_2_16seeds --only 10 11 12
```

The late window's upper bound is 1.01 so that the last step (t = 0.98) falls inside it.

**Interventions (about 5 hours)**

```sh
T=$O/SiT-L_2_16seeds/analysis/intervention_tables.npz
.venv/bin/python bench/guidance_intervene.py --build-tables --calibration $O/SiT-L_2_16seeds --tables $T
.venv/bin/python bench/guidance_intervene.py --tables $T --weights $W --out $O/SiT-L_2_interventions --seeds $(seq 16 31)
.venv/bin/python bench/guidance_intervene_eval.py --run $O/SiT-L_2_interventions --embed --stats --figures
```

The tables are fit on seeds 0–15, and evaluation uses seeds 16–31. The script refuses overlapping seeds.

**Mechanistic study (about 5 hours, 92 GB)**

```sh
.venv/bin/python bench/guidance_capture.py --weights $W --weights-source "$SRC" --out $O/SiT-L_2_mech --tables $T --seeds $(seq 16 31) \
  --conditions w1 w4 score_0.5 score_1 score_2 cond_0.25 cond_0.5 cond_1 score_1+cond_0.25 score_1+cond_0.5
.venv/bin/python bench/guidance_capture_verify.py --run $O/SiT-L_2_mech --deep cond_1 23
.venv/bin/python bench/guidance_mech.py --run $O/SiT-L_2_mech
.venv/bin/python bench/guidance_intervene_eval.py --run $O/SiT-L_2_mech/analysis/finals --embed --stats
.venv/bin/python bench/guidance_mech_figures.py --run $O/SiT-L_2_mech
```

## Outputs

Sizes are for a full local run; the repository keeps only the small files (see
[What this repository contains](#what-this-repository-contains)).

| Folder | Contents | Size |
|---|---|---|
| `SiT-L_2/` | First run: `seed_N.npz`, `manifest.json`, metrics CSVs, `guidance_delta.png`, verification reports, decoded samples | 158 MB |
| `SiT-L_2_16seeds/` | `w4/` and `w1/` (16 seeds each, full logging), `manifest.json`, `index.csv`, and `analysis/` (CSV tables, embeddings, sample grids, figures 01–12, intervention tables) | 19 GB |
| `SiT-L_2_window_early/`, `SiT-L_2_window_late/` | Windowed-guidance runs, same layout | 9.2 GB each |
| `SiT-L_2_interventions/` | One folder per condition with `seed_N.npz` (final latents), embeddings, samples, result CSVs, figures 13–14 | 251 MB |
| `SiT-L_2_mech/` | One folder per condition, full logging, and `analysis/` (mech_*.csv, finals, figures m1–m4) | 92 GB |
| `SiT-L_2_interventions_calibration/` | Three batches used only to check the intervention code reproduces earlier runs bit for bit | small |
| `SiT-L_2_seeds8-15/` | A run that was stopped after two seeds; not used, and not in the repository | 37 MB |

**Reproducibility record.** Every run folder has a `manifest.json` with the command, model, weights hash, class
labels, guidance settings, sampler, device, `sit/` commit and package versions. Every batch has a `metadata.json`
with its seed, condition and the SHA-256 of its initial noise and final latents. Noise comes from
`torch.Generator().manual_seed(seed)` on CPU, so it does not depend on the device.

**Array conventions.** Time `s_t` runs from 0 (noise) to 1 (data). In `trajectory.npz`, arrays have axes
`(step, class, channel, 32, 32)`. In `pooled.npy` and `tokens.npy`, the branch axis is 0 = conditional,
1 = unconditional. `cos[k]` compares step k+1 with step k.

**Not reproducible by a script:** the contact sheets in `SiT-L_2/samples/` and figure 12 were assembled with one-off
snippets from the saved sample grids.

## Older benchmark scripts

These came before the guidance work and compare inference engines and speed. They are not needed for the guidance
experiments.

| Script | What it does |
|---|---|
| `run_all.py`, `rerun.sh` | Sweep all 12 SiT sizes across engines, one process per run, logging to `results*.jsonl`. |
| `worker.py` | Samples one model on one engine (`mps` = PyTorch on the Apple GPU, `mlx` = Apple's MLX) and prints timing and memory. MLX precision comes from `SIT_MLX_DTYPE`. |
| `prep.py` | Writes weights and shared noise for a benchmark run. Every size except XL/2 uses **random** weights. |
| `sit_mlx.py` | An MLX port of the SiT forward pass. |
| `engine_probe.py`, `parity.py` | Single-forward timing and numerical agreement between engines. |
| `decode.py` | Decodes latents to a PNG grid with the SD VAE. |
| `memwatch.py` | Peak process memory on macOS, including GPU memory. |
| `summary.csv`, `therm.log` | Benchmark summary and thermal log. |

## Requirements

Python 3.12 with torch 2.15, torchvision 0.29, torchdiffeq 0.2.5, diffusers 0.42, timm 1.0.31, numpy 2.5,
safetensors 0.8, scipy 1.18, pillow and matplotlib 3.11. The VAE (`stabilityai/sd-vae-ft-mse`), DINOv2, CLIP and the
ImageNet classifier (`vit_base_patch16_224.augreg2_in21k_ft_in1k`) download from Hugging Face on first use.
