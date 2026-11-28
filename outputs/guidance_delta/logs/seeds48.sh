#!/bin/zsh
# After the gstart 32-47 run: the realistic-frontier conditions on seeds 48-63, so the frontier rests on 48 seeds. Stops at 09:45.
cd "$(dirname "$0")/../../.."  # project root (this file is in outputs/guidance_delta/logs)
O=outputs/guidance_delta
while pgrep -f "conditions gstart" >/dev/null || ! grep -q "GSTART_DONE" $O/logs/gstart2.log; do sleep 60; done
STOP=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-28 09:45" +%s)
.venv/bin/python bench/guidance_intervene.py --tables $O/SiT-L_2_16seeds/analysis/intervention_tables.npz --token-tables $O/SiT-L_2_tokens/analysis/token_tables.npz --weights pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors --out $O/SiT-L_2_night --seeds $(seq 48 63) --conditions w4 cond_0.5 cond_0.75 cond_1 score_1 gstart_0.12 gstart_0.2 >> $O/logs/seeds48.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do [ $(date +%s) -ge $STOP ] && { echo "stopped at 09:45" >> $O/logs/seeds48.log; kill $pid; }; sleep 30; done
.venv/bin/python bench/guidance_intervene_eval.py --run $O/SiT-L_2_night --embed >> $O/logs/seeds48.log 2>&1
echo SEEDS48_DONE >> $O/logs/seeds48.log
