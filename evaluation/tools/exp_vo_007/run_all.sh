#!/usr/bin/env bash
# EXP-VO-007: the six VO runs and their evaluations, in one place.
#
# Deliberately a flat list rather than a loop with clever defaults: every arm must be visibly
# identical except for the two selectors, and a reader should be able to check that by eye.
#
# Run from the repository root. Gates first -- plot_groundtruth.py exits non-zero if any ingest
# gate fails, and `set -e` then stops before the estimator sees a frame.
set -euo pipefail

PY="${PY:-python}"
JAVA_HOME="${JAVA_HOME:-C:/Program Files/Java/jdk-19}"
export JAVA_HOME
OUT="${OUT:-evaluations/exp-vo-007}"
PROFILE="${PROFILE:-}"          # optional: LiDAR height-profile JSON for the regime split

echo "=== ingest gates + ground-truth-only figures (before any VO) ==="
"$PY" evaluation/tools/exp_vo_007/plot_groundtruth.py \
    --dataset datasets/amtown01-c --out "$OUT/ingest" --target-agl 80 \
    ${PROFILE:+--height-profile "$PROFILE"}

echo "=== six runs ==="
for cfg in evaluation/eval_configs/exp-vo-007/run-amtown01-c-*.json; do
    echo "--- $cfg"
    ./gradlew.bat -q run -PmainClass=org.boofcv.evaluation.VoRunnerApp --args="--config $cfg"
done

echo "=== six evaluations ==="
for cfg in evaluation/configs/eval-amtown01-c-*.json; do
    name=$(basename "$cfg" .json)
    ( cd evaluation && "$PY" -m naveval.evaluate --config "configs/$(basename "$cfg")" \
        --out "../$OUT/${name#eval-}" )
done

echo "=== hypotheses + figures ==="
"$PY" evaluation/tools/exp_vo_007/analyse.py --out "$OUT" \
    ${PROFILE:+--height-profile "$PROFILE"}
"$PY" evaluation/tools/exp_vo_007/plot_exp_vo_007.py --out "$OUT/figures"
echo "done -> $OUT"
