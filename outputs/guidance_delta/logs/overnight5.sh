#!/bin/zsh
# After queue 4 (switch sweep seeds 4-15): (1) the low-band shared / seed-specific split on seeds 8-15, so the central patching
# result has 16 seeds; (2) if it can start by 08:10, the split at two more switch times (k = 8, 16) on seeds 0-7 to locate the
# crossover. Sampling stops at 09:20; whatever finished is then embedded and analysed.
setopt NO_NOMATCH
cd "$(dirname "$0")/../../.."  # project root (this file is in outputs/guidance_delta/logs)
O=outputs/guidance_delta; L=$O/logs; P=$O/paired_tests; PY=.venv/bin/python
W=pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors
STOP=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-27 09:20" +%s); LATE=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-27 08:10" +%s)
log() { echo "$(date +%H:%M:%S) $*" >> $L/overnight4.log }
gpu() {
  local name=$1; shift
  [ $(date +%s) -lt $STOP ] || { log "SKIP $name"; return 1; }
  log "START $name"; "$@" >> $L/$name.log 2>&1 & local pid=$!
  while kill -0 $pid 2>/dev/null; do [ $(date +%s) -ge $STOP ] && { log "STOP $name at 09:20"; kill $pid; }; sleep 20; done
  wait $pid; local rc=$?; log "END $name rc=$rc"; return $rc
}
analyse() { REAL=outputs/real_imagenet_train $PY bench/patch_analysis.py $O/SiT-L_2_patch > $P/patch_all.txt 2>&1; log "patch analysis done ($1)" }
while pgrep -f "overnight4.sh" >/dev/null; do sleep 60; done
log "queue 5 started"
gpu split_more $PY bench/patch_sweep.py --run $O/SiT-L_2_16seeds --weights $W --out $O/SiT-L_2_patch --seeds $(seq 8 15) --split-only
$PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_patch --embed >> $L/patch_embed2.log 2>&1; log "embedded"
analyse "split on seeds 0-15"
if [ $(date +%s) -lt $LATE ]; then
  gpu split_times $PY bench/patch_sweep.py --run $O/SiT-L_2_16seeds --weights $W --out $O/SiT-L_2_patch --seeds $(seq 0 7) --split-only --ks 8 16
  $PY bench/guidance_intervene_eval.py --run $O/SiT-L_2_patch --embed >> $L/patch_embed3.log 2>&1; log "embedded"
  analyse "with k = 8, 16"
else
  log "SKIP split_times (too late)"
fi
log "queue 5 finished"
