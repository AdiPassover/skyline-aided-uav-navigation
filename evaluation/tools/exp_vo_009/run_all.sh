#!/usr/bin/env bash
# EXP-VO-009 end to end, in the order the record's Procedure fixes.
#
# Run from the repository root. Steps 1-4 are gates: each must pass before
# the similarity arm is allowed to touch flight data, and the script stops if one fails.
#
#   bash evaluation/tools/exp_vo_009/run_all.sh
#
# MARS-LVIG imagery is not part of the repository. HKAIRPORT_ROOT points at wherever the
# HKairport01 datasets live; AMtown01's are expected under datasets/ (built by EXP-VO-007's ingest).
set -euo pipefail

cd "$(dirname "$0")/../../.."
export JAVA_HOME="${JAVA_HOME:-C:/Program Files/Java/jdk-19}"
PY="${PY:-python}"
HKAIRPORT_ROOT="${HKAIRPORT_ROOT:-datasets}"
OUT=evaluations/exp-vo-009

echo "### Gate 1 -- the closed-form variance identity (LIT-VO-005 eq. 8)"
"$PY" evaluation/tools/exp_vo_009/variance_theory.py --mode per-sample --trials 40000 \
    --out "$OUT/variance_theory_per_sample.json"
"$PY" evaluation/tools/exp_vo_009/variance_theory.py --mode marginal --trials 2000 \
    --out "$OUT/variance_theory_marginal.json"

echo "### Gates 2-4 -- known-answer geometry, readout identity, and the untouched shipped arms"
./gradlew.bat test -q

echo "### Gate 4b -- the committed baseline still reproduces bit-identically"
./gradlew.bat run -PmainClass=org.boofcv.evaluation.VoRunnerApp \
    --args="--config evaluation/eval_configs/run-hkairport01-a.json \
            --dataset-dir $HKAIRPORT_ROOT/hkairport01-a --run-id gate4-hkairport01-a-repro" -q
"$PY" evaluation/tools/exp_vo_009/check_reproduction.py \
    --reference runs/hkairport01-a-run-v1 --candidate runs/gate4-hkairport01-a-repro
rm -rf runs/gate4-hkairport01-a-repro

echo "### Phase 5 -- Monte Carlo over the real estimators (tier 1)"
./gradlew.bat run -PmainClass=org.boofcv.stitching.diagnostics.ModelRotationMonteCarloApp \
    --args="--out $OUT/monte_carlo.json" -q

echo "### Phase 7 -- the similarity arm on all three windows"
for w in hkairport01-a hkairport01-b amtown01-c; do
  ./gradlew.bat run -PmainClass=org.boofcv.evaluation.VoRunnerApp \
      --args="--config evaluation/eval_configs/exp-vo-009/run-${w}-similarity-rigid.json" -q
done

echo "### Phase 6 -- per-frame rotation, every model, against RTK heading"
for w in hkairport01-a hkairport01-b amtown01-c; do
  "$PY" evaluation/tools/exp_vo_009/rotation_across_models.py \
      --dataset "datasets/${w}" --tag "$w" --out "$OUT/rotation_${w}.json"
done

echo "### Phases 7 + 9 -- five arms per window, plus support and cost"
"$PY" evaluation/tools/exp_vo_009/analyse.py --out "$OUT/comparison.json"

echo "### Phase 8 -- GT-heading substitution on the similarity arm (EXP-VO-008's method)"
for w in hkairport01-b amtown01-c; do
  "$PY" evaluation/tools/exp_vo_008/substitution.py \
      --dataset "datasets/${w}" --run "runs/${w}-similarity-rigid-v1" \
      --label "${w}-similarity" --out "$OUT/subst_${w}_similarity.json"
done

echo "### Phase 9 -- matched-conditions motion-stage timing, all three models back to back"
"$PY" evaluation/tools/exp_vo_009/timing_compare.py --out "$OUT/timing.json" \
    --hkairport-root "$HKAIRPORT_ROOT"

echo "### Figures"
"$PY" evaluation/tools/exp_vo_009/plot_exp_vo_009.py --out "$OUT/figures"

echo "### done -- artifacts in $OUT"
