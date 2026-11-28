#!/bin/zsh
# SDE sampler test: one condition at a time over seeds 16-31, each embedded right after it finishes.
cd "$(dirname "$0")/../../.."  # project root (this file is in outputs/guidance_delta/logs)
O=outputs/guidance_delta; PY=.venv/bin/python
W=pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors; T=$O/SiT-L_2_16seeds/analysis/intervention_tables.npz
for c in w1 w4 score_1 cond_0.5 cond_0.75 cond_1; do
  $PY bench/guidance_intervene.py --tables $T --weights $W --out $O/SiT-L_2_sde --sampler sde --sde-steps 100 --seeds $(seq 16 31) --conditions $c >> $O/logs/sde.log 2>&1 || { echo "$(date +%H:%M:%S) FAILED $c" >> $O/logs/sde_progress.log; exit 1; }
  $PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_sde --embed >> $O/logs/sde_embed.log 2>&1
  echo "$(date +%H:%M:%S) DONE $c" >> $O/logs/sde_progress.log
done
echo "$(date +%H:%M:%S) ALL DONE" >> $O/logs/sde_progress.log
