# Classifier-free guidance in SiT: code for the report

This project studies **classifier-free guidance (CFG)** in SiT, a class-conditional image generator: where in sampling
guidance removes within-class diversity, and whether damping it early can restore diversity while keeping fidelity.
The report is `paper/main.tex`. [RESULTS.md](RESULTS.md) is the longer lab notebook of every experiment run along the way.

This repository contains only the code that produces the report's results. All commands run from the project root with the
project's virtual environment, `.venv/bin/python`.

## What produces what

| Report item | Data | Script that computes it | Script that plots it |
|---|---|---|---|
| Figure 1: shared guidance over time | 16-seed run | `guidance_analysis.py` (stats) → `alignment_shared.csv` | `guidance_figures.py` → `02_alignment_within_class.png` |
| Figure 2: effective rank, w = 4 vs w = 1 | 16-seed run | `guidance_analysis.py` (subspace) → `subspace_pooled.csv` | `guidance_figures.py` → `03_effective_rank_pooled.png` |
| Table 2: early vs late guidance | 16-seed run + both window runs | `guidance_window.py` → `window_effects.csv` | |
| Table 3: comparison of interventions | intervention run | `guidance_intervene_eval.py` → `intervention_results.csv` | |
| Figure 3: effective rank under interventions | mechanistic run | `guidance_mech.py` → `mech_blocks.csv` | `guidance_mech_figures.py` → `m3b_block_rank.png` |
| Figures 5–6: sample images (w = 1, w = 4, score damping rows) | intervention run | `guidance_intervene_eval.py --embed` → `samples/*.png` | assembled by hand |

**Not produced by code in this repository.** The guidance-interval sweep for τ = 0.1, 0.2, 0.4 and 0.5 (Table 4; its
τ = 0.3 row is `scalar_1` from Table 3), the shuffled push (Section 2.6, Table 5, the shuffled-push rows of Figures 5
and 6), and Figure 4 (recovery vs fidelity) were made with code that is not in this repository.

## Layout

| Path | What it is |
|---|---|
| `sit/` | The upstream SiT repository at commit `cbde832`, **unmodified**. Every script imports its model and sampler from here. |
| `bench/` | This project's code (described below). |
| `paper/` | The report. |
| `pretrained_models/` | Model weights (see [Weights](#weights)). |
| `outputs/guidance_delta/` | Each run's manifest, per-batch metadata, result tables and figures (see [Outputs](#outputs)). |

Clone with the SiT submodule:

```sh
git clone --recurse-submodules https://github.com/nonname001/cfg-project.git
```

Not included, because of size or licensing (see `.gitignore`): the raw experiment arrays (`.npz`, `.npy`, about 130 GB),
the model weights, and the decoded samples and DINOv2/CLIP embeddings derived from the raw arrays. Regenerate them with the
commands below; each batch's `metadata.json` records the SHA-256 of its noise and final latents, so a rerun can be checked
against them. The one array included is `outputs/guidance_delta/SiT-L_2_16seeds/analysis/intervention_tables.npz`, the
per-class subspaces the interventions need, so the intervention experiments can run without regenerating the 16-seed run.

## Weights

| File | Source | SHA-256 |
|---|---|---|
| `pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors` | Community upload, [BiliSakura/SiT-diffusers](https://huggingface.co/BiliSakura/SiT-diffusers), pinned to revision `2f02393a3079dfa6210c10302cc927f59d2cb142`. The SiT authors only released XL/2. | `fd3c19ec1a7675f5648f5200acf4f1a94ae6a44ff5ff81a1011de8242d12edd6` |

All experiments use SiT-L/2. To fetch it:

```sh
mkdir -p pretrained_models/SiT-L-2-256_BiliSakura_2f02393
curl -fL -o pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors \
  "https://huggingface.co/BiliSakura/SiT-diffusers/resolve/2f02393a3079dfa6210c10302cc927f59d2cb142/SiT-L-2-256/transformer/diffusion_pytorch_model.safetensors"
shasum -a 256 pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors
```

Only the `.safetensors` weights are used; the repo's own pipeline code (`trust_remote_code`) is never run.

## How the code hooks into SiT without editing it

SiT samples by integrating a velocity field from noise (t = 0) to data (t = 1). With CFG, `model.forward_with_cfg`
runs the network on a doubled batch (8 conditional rows, 8 rows with the "null" class) and mixes the two outputs.
The scripts observe and change this in three ways, none of which edit `sit/`:

1. **Replace `model.forward` on the instance.** `forward_with_cfg` calls `self.forward(...)` directly, so ordinary
   PyTorch forward hooks never see it. A wrapper assigned to `model.forward` records the conditional output
   `f_cond` and the unconditional output `f_uncond` at every step, then deletes itself afterwards.
2. **Forward hooks on `model.blocks[i]`** record each transformer block's output (activations), and on
   `model.y_embedder` change the class embedding for condition damping; hooks on blocks 13–23 do feature damping.
3. **Wrap the function the sampler integrates**, to adjust the guided velocity after the model has run (scalar, score,
   frequency and wrong-class damping) or to switch guidance off outside a window.

Sampling always uses SiT's own `Sampler.sample_ode` with a **fixed-step Euler** solver, 50 grid points, so 49
model calls. SiT's default solver (dopri5) is adaptive, which would make "step k" ill-defined.

## Scripts

| Script | What it does |
|---|---|
| `common.py` | Shared constants (the 8 classes, null class, guided channels) and helpers: model loading, the per-seed initial noise, hashing. |
| `guidance_capture.py` | Samples batches and stores, per batch, `trajectory.npz` (s_t, f_cond, f_uncond, Δ, initial noise, final latents) and `pooled.npy` (every block's output averaged over the 256 image tokens, both branches, every step). Options: several w values (`--ws`), a guidance window (`--window lo hi`), or named conditions including interventions (`--conditions`). `--token-steps k ...` also stores every block's unpooled output (256 tokens × 1024) at those steps as `tokens_uncond.npy` and `tokens_delta.npy` (cond − uncond, subtracted in float32), both float16, about 200 MB per step per batch. Writes each batch to a temporary folder and renames it when complete; rerunning skips finished batches. |
| `guidance_embed.py` | Decodes each batch's final latents with the SD VAE and embeds them with DINOv2 ViT-B/14 and CLIP ViT-B/16. |
| `guidance_analysis.py` | `stats`: within-class pairwise cosine and shared proportion S of Δ, f_cond, f_uncond. `subspace`: per-block effective rank at w = 1 and w = 4. Also the statistics helpers the other scripts import. |
| `guidance_window.py` | Share of each w = 4 effect reproduced by guidance only early or only late, with bootstrap intervals. |
| `guidance_intervene.py` | The interventions, saving final latents only. `--build-tables` fits the per-class subspaces score and feature damping need. The `Intervention` class here is also what `guidance_capture.py` uses. Its docstring defines every method. |
| `guidance_intervene_eval.py` | `--embed`: decodes, embeds and classifies every sample. `--stats`: diversity recovery and fidelity retention with bootstrap intervals. |
| `guidance_mech.py` | Per-block effective rank under each intervention. |
| `guidance_figures.py`, `guidance_mech_figures.py` | Figures 1–3. |

## Reproducing each result

`W` is the SiT-L/2 weights path and `SRC` a free-text note on where the weights came from (recorded in the manifest). Keep
the machine awake for long runs (`caffeinate -i ...`).

```sh
W=pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors
O=outputs/guidance_delta
```

**Figures 1 and 2: w = 4 vs w = 1, 16 seeds**

```sh
.venv/bin/python bench/guidance_capture.py --weights $W --weights-source "$SRC" --out $O/SiT-L_2_16seeds --seeds $(seq 0 15)
.venv/bin/python bench/guidance_embed.py --run $O/SiT-L_2_16seeds
.venv/bin/python bench/guidance_analysis.py --run $O/SiT-L_2_16seeds
.venv/bin/python bench/guidance_figures.py --run $O/SiT-L_2_16seeds
```

**Table 2: early vs late guidance** (needs the 16-seed run)

```sh
.venv/bin/python bench/guidance_capture.py --weights $W --weights-source "$SRC" --out $O/SiT-L_2_window_early --seeds $(seq 0 15) --ws 4 --window 0 0.3
.venv/bin/python bench/guidance_capture.py --weights $W --weights-source "$SRC" --out $O/SiT-L_2_window_late  --seeds $(seq 0 15) --ws 4 --window 0.3 1.01
.venv/bin/python bench/guidance_embed.py --run $O/SiT-L_2_window_early --ws 4
.venv/bin/python bench/guidance_embed.py --run $O/SiT-L_2_window_late  --ws 4
.venv/bin/python bench/guidance_window.py --base $O
```

The late window's upper bound is 1.01 so that the last step (t = 0.98) falls inside it.

**Table 3: interventions**

```sh
T=$O/SiT-L_2_16seeds/analysis/intervention_tables.npz
.venv/bin/python bench/guidance_intervene.py --build-tables --calibration $O/SiT-L_2_16seeds --tables $T
.venv/bin/python bench/guidance_intervene.py --tables $T --weights $W --out $O/SiT-L_2_interventions --seeds $(seq 16 31)
.venv/bin/python bench/guidance_intervene_eval.py --run $O/SiT-L_2_interventions --embed --stats
```

The tables are fit on seeds 0–15, and evaluation uses seeds 16–31. The script refuses overlapping seeds.

**Figure 3: effective rank under interventions**

```sh
.venv/bin/python bench/guidance_capture.py --weights $W --weights-source "$SRC" --out $O/SiT-L_2_mech --tables $T --seeds $(seq 16 31) \
  --conditions w4 score_1 cond_0.5 score_1+cond_0.5
.venv/bin/python bench/guidance_mech.py --run $O/SiT-L_2_mech
.venv/bin/python bench/guidance_mech_figures.py --run $O/SiT-L_2_mech
```

## Outputs

| Folder | Contents |
|---|---|
| `SiT-L_2_16seeds/` | `w4/` and `w1/` (16 seeds each), `manifest.json`, and `analysis/` (CSV tables, embeddings, figures, intervention tables) |
| `SiT-L_2_window_early/`, `SiT-L_2_window_late/` | Windowed-guidance runs, same layout |
| `SiT-L_2_interventions/` | One folder per condition with `seed_N.npz` (final latents), embeddings, samples, `intervention_results.csv` |
| `SiT-L_2_mech/` | One folder per condition and `analysis/` (`mech_blocks.csv`, figures) |

These folders also hold tables and figures from earlier analyses that are not in the report (see RESULTS.md); the code that
made them has been removed, and `SiT-L_2/`, `SiT-L_2_seeds8-15/` and `SiT-L_2_interventions_calibration/` come from runs
the report does not use. The data saved by the removed code differs in one way from what the current `guidance_capture.py`
writes: earlier runs also stored token activations, x̂ and per-step norms, which no report result uses.

**Reproducibility record.** Every run folder has a `manifest.json` with the model, weights hash, class labels, guidance
settings, sampler, device, `sit/` commit and package versions. Every batch has a `metadata.json` with its seed, condition
and the SHA-256 of its initial noise and final latents. Noise comes from `torch.Generator().manual_seed(seed)` on CPU, so it
does not depend on the device.

**Array conventions.** Time `s_t` runs from 0 (noise) to 1 (data). In `trajectory.npz`, arrays have axes
`(step, class, channel, 32, 32)`. In `pooled.npy`, the branch axis is 0 = conditional, 1 = unconditional.

## Requirements

Python 3.12 with torch 2.15, torchvision 0.29, torchdiffeq 0.2.5, diffusers 0.42, timm 1.0.31, numpy 2.5,
safetensors 0.8 and matplotlib 3.11. The VAE (`stabilityai/sd-vae-ft-mse`), DINOv2, CLIP and the ImageNet classifier
(`vit_base_patch16_224.augreg2_in21k_ft_in1k`) download from Hugging Face on first use.
