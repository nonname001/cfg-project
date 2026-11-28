#!/bin/zsh
# After queue 2: extend direction 5 (spec_1, swap_1) to seeds 32-47 if it starts before 09:05; sampling stops at 09:45,
# then embed what finished and rerun the paired test on the longest contiguous seed range from 16.
setopt NO_NOMATCH
cd "$(dirname "$0")/../../.."  # project root (this file is in outputs/guidance_delta/logs)
O=outputs/guidance_delta; L=$O/logs; P=$O/paired_tests; PY=.venv/bin/python
W=pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors
T=$O/SiT-L_2_16seeds/analysis/intervention_tables.npz; TT=$O/SiT-L_2_tokens/analysis/token_tables.npz
log() { echo "$(date +%H:%M:%S) $*" >> $L/overnight.log }
STARTBY=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-26 09:05" +%s); STOP=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-26 09:45" +%s)
while pgrep -f "overnight2.sh" >/dev/null; do sleep 30; done
[ $(date +%s) -lt $STARTBY ] || { log "queue 3: too late to extend direction 5"; exit 0; }
log "queue 3: START specswap seeds 32-47"
$PY bench/guidance_intervene.py --tables $T --token-tables $TT --weights $W --out $O/SiT-L_2_night --seeds $(seq 32 47) --conditions spec_1 swap_1 >> $L/specswap_more.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do [ $(date +%s) -ge $STOP ] && { log "queue 3: stopping specswap at 09:45"; kill $pid; }; sleep 20; done
log "queue 3: embed"
$PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_night --embed >> $L/specswap_embed2.log 2>&1
HI=$(ls $O/SiT-L_2_night/embeddings/swap_1 | sed 's/seed_//;s/.npz//' | sort -n | awk 'NR==1{lo=$1} {if ($1==lo+NR-1) hi=$1} END{print hi}')
REAL=outputs/real_imagenet_train $PY bench/paired_vs_cond.py $O/SiT-L_2_night 16 $HI spec_1 swap_1 score_1 scalar_1 -- spec_1:w4 swap_1:w4 spec_1:scalar_1 swap_1:score_1 > $P/specswap_16_$HI.txt 2>&1
REAL=outputs/real_imagenet_train $PY bench/points.py $O/SiT-L_2_night 16 $HI w4 spec_1 swap_1 score_1 scalar_1 > $P/specswap_points_16_$HI.txt 2>&1
log "queue 3 finished: dir5 analysis on seeds 16-$HI"
