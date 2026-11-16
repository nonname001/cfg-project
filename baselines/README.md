# Baselines: diversity under CFG, baselines and basic analysis

Compares guidance methods that aim to recover diversity on SiT-XL/2, all running on one shared sampling loop, and measures
when diversity is lost during sampling. Nothing in `sit/` is modified.

## Files

| File | Purpose |
|---|---|
| `common.py` | Classes, noise, model loading, and the Euler sampling loop shared by every method |
| `methods.py` | All guidance methods (one small class each), plus `METHODS`, the table of method names and default parameters |
| `fit_score.py` | Fits each class's shared direction for our score method (runs only the first 30% of sampling) |
| `sample.py` | Samples every method for every seed; saves the final latents and x̂ at 25 time points (every other step) |
| `evaluate.py` | Decodes, extracts features, computes metrics; writes 2 CSV files and 2–3 figures |
| `check.py` | Correctness checks that need no weights |

## Methods

`w` follows SiT's convention: w = 1 means no guidance. The reference strength is w = 4.

| Name | Method | Default parameters | Notes |
|---|---|---|---|
| `cfg_w1` … `cfg_w4` | Plain CFG, w sweep | w = 1, 1.5, 2, 3, 4 | Reference curve for the quality–diversity trade-off |
| `interval` | Guidance Interval | w = 4 for t ∈ [0.50, 0.75) | Interval converted from the paper's DiT result; **needs re-searching on SiT** |
| `linear` | Increasing schedule (Schedulers paper) | w(t) = 1 + 2t(w−1) | Mean w equals 4 |
| `cads` | CADS: annealed noise on the class embedding early in sampling | τ1=0.5, τ2=0.9, s=0.15, ψ=1 | Paper's w=5 values. Whether noise is redrawn each step and whether the null embedding is also noised **must be checked against the paper's Algorithm 1**. Images currently degrade badly (see below) |
| `apg` | APG: projection, clipping and negative momentum on x̂ | η=0, r=5, β=−0.5 | Paper's DiT values; **r needs tuning on SiT** |
| `fdg` | FDG: separate w for low and high frequencies | w_low=1, w_high=4 | Pyramid blur kernel **must be checked against the paper's Algorithm 2** |
| `score` | Ours: remove the class-shared direction early | λ=1, t<0.3, rank 3 | Run `fit_score.py` first |
| `cond` | Ours: pull the class embedding toward null early | λ=0.5, t<0.3 | |

Parameters can be overridden on the command line, e.g. `interval:lo=0.4,hi=0.8` or `apg:r=2.5`. Each spec is saved in
its own folder, so parameter sweeps are easy.

## Setup

```bash
pip install timm "diffusers==0.35.2" "huggingface-hub<1.0" torchdiffeq
```

The latest `diffusers` upgrades `huggingface-hub` to 1.x or later, which conflicts with the installed `transformers 4.x`,
so it is pinned to 0.35.2. Tested with PyTorch 2.7.1 (Mac MPS), timm 1.0.31 and diffusers 0.35.2.

```bash
mkdir -p pretrained_models && curl -fL -o pretrained_models/SiT-XL-2-256x256.pt "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0"
```

The weights go in `pretrained_models/` at the project root (about 2.7 GB; the link comes from `sit/download.py`).
Evaluation downloads the VAE, DINOv2, CLIP and the classifier from Hugging Face automatically.

## Running (from `baselines/`)

```bash
python check.py
```

```bash
python fit_score.py --seeds 0-15
```

```bash
python sample.py --methods cfg_w1 cfg_w1.5 cfg_w2 cfg_w3 cfg_w4 interval linear cads cads:s=0.05 apg fdg score cond --seeds 100-107
```

```bash
python evaluate.py --run runs/main --grid-class 0
```

- Fitting uses seeds 0–15 and evaluation uses seeds from 100 up; the code refuses any overlap.
- Each batch is one image for each of the 8 classes. 13 methods × 8 seeds = 104 batches, about 1 minute each on an
  Apple-silicon Mac. Start with 8 seeds to see the trend, then go to 32: rerunning `sample.py --seeds 100-131` skips
  finished batches.
- Stopping midway never leaves a corrupt file; just rerun to continue.

## Outputs (`runs/main/`)

| File | Contents |
|---|---|
| `metrics.csv` | Per method: within-class diversity (Vendi and MSS on DINOv2, MSS on CLIP, layout diversity V_low) and quality (classifier confidence in the target class, top-1), with 90% subsample intervals |
| `over_time.csv` | Per method, the within-class spread of x̂ at 25 time points (all frequencies and low frequencies); with `--time-dino`, also the DINOv2 MSS of the decoded x̂ |
| `tradeoff.png` | Quality on x, diversity on y; the grey line is the CFG w sweep, other methods are points |
| `over_time.png` | Spread of x̂ against t, showing when diversity is lost (CFG sweep as grey dashed lines) |
| `samples.png` | With `--grid-class c`: one row per method, showing samples of class c |

All diversity metrics are computed within each class (across seeds) first, then averaged over the 8 classes. Intervals
come from repeatedly drawing 80% of the seeds. Vendi and spread shrink with sample size, so each interval keeps the
width of the subsample distribution but is centred on the full-sample estimate. Treat them as approximate.

## First results (8 seeds, `runs/main`)

- Layout diversity is settled by about t ≈ 0.2–0.3; the CFG curves separate in this window and stay flat afterwards.
- `score` recovers layout diversity (V_low 0.73, above w = 1) at w = 4 quality, but not semantic diversity: DINOv2
  Vendi and MSS stay at the w = 4 level. `fdg` and `linear` show the same pattern, more weakly.
- `interval` and `cond` lie on the CFG curve, so they act like a smaller w. `apg` is slightly below it.
- `cads` degrades images (p_target 0.45 at s = 0.15); s = 0.05 is better but still worse than no guidance. Likely an
  implementation or scale issue; check it before comparing.

## Not done yet

- FID and Precision/Recall need real ImageNet images, which we do not have yet. Later, export the decoded images as a
  `(N, 256, 256, 3)` uint8 array (key `arr_0`) and pass it to ADM's evaluation scripts:
  https://github.com/openai/guided-diffusion/tree/main/evaluations
