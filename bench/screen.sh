#!/bin/zsh
# Screen interventions: sample them into SiT-L_2_night for the given seeds, embed, and run the paired test against the
# condition-damping curve (bench/paired_vs_cond.py) on those seeds; the result goes to paired_tests/night_<tag>.txt.
# usage: bench/screen.sh <seed_lo> <seed_hi> <tag> cond ...
set -e
cd "$(dirname "$0")/.."
O=outputs/guidance_delta
OUT=outputs/guidance_delta/paired_tests
LO=$1; HI=$2; TAG=$3; shift 3
.venv/bin/python bench/guidance_intervene.py --tables $O/SiT-L_2_16seeds/analysis/intervention_tables.npz \
  --token-tables $O/SiT-L_2_tokens/analysis/token_tables.npz --weights pretrained_models/SiT-L-2-256_BiliSakura_2f02393/transformer.safetensors \
  --out $O/SiT-L_2_night --seeds $(seq $LO $HI) --conditions "$@"
.venv/bin/python bench/guidance_intervene_eval.py --run $O/SiT-L_2_night --embed > /dev/null
.venv/bin/python bench/paired_vs_cond.py $O/SiT-L_2_night $LO $HI "$@" tfeat_1.3333+score_1+late_6+smooth_0.1 > $OUT/night_$TAG.txt
echo SCREEN $TAG DONE
