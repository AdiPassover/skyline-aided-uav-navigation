#!/usr/bin/env bash
# EXP-VO-010 end to end, in the order the record's Procedure fixes.
#
# Run from the repository root. Steps 1-3 are gates: each must pass before
# the refined arms are allowed to touch flight data, and the script stops if one fails.
#
#   bash evaluation/tools/exp_vo_010/run_all.sh
#
# MARS-LVIG imagery is not part of the repository. HKAIRPORT_ROOT points at wherever the
# HKairport01 datasets live; AMtown01's are expected under datasets/ (built by EXP-VO-007's ingest).
set -euo pipefail

cd "$(dirname "$0")/../../.."
export JAVA_HOME="${JAVA_HOME:-C:/Program Files/Java/jdk-19}"
PY="${PY:-python}"
HKAIRPORT_ROOT="${HKAIRPORT_ROOT:-datasets}"
OUT=evaluations/exp-vo-010
WINDOWS="hkairport01-a hkairport01-b amtown01-c"
MODELS="affine homography"

echo "### Gates 2-3 -- refinement semantics, the observational probe, and the heading metric"
./gradlew.bat test -q
(cd evaluation && "$PY" -m pytest tools/exp_vo_010 -q)

echo "### Gate 1 -- the committed refine=false baseline still reproduces bit-identically"
./gradlew.bat run -PmainClass=org.boofcv.evaluation.VoRunnerApp \
    --args="--config evaluation/eval_configs/run-hkairport01-a.json \
            --dataset-dir $HKAIRPORT_ROOT/hkairport01-a --run-id gate1-hka-repro" -q
"$PY" evaluation/tools/exp_vo_009/check_reproduction.py \
    --reference runs/hkairport01-a-run-v1 --candidate runs/gate1-hka-repro
rm -rf runs/gate1-hka-repro

echo "### Gate 2b -- the probe does not perturb a real run, and its sidecar is self-consistent"
./gradlew.bat run -PmainClass=org.boofcv.evaluation.VoRunnerApp \
    --args="--config evaluation/eval_configs/exp-vo-010/gate2-probe-equivalence.json" -q
"$PY" evaluation/tools/exp_vo_009/check_reproduction.py \
    --reference runs/hkairport01-a-affine-rigid-v1 --candidate runs/gate2-hka-probed-refine-off
rm -rf runs/gate2-hka-probed-refine-off

echo "### Phase 3 -- known-answer synthetic, minimal sample vs refined (tier 1)"
./gradlew.bat run -PmainClass=org.boofcv.stitching.diagnostics.RefinementMonteCarloApp \
    --args="--out $OUT/monte_carlo.json" -q

echo "### Phase 6 -- the refined arms. refine=false reuses the committed EXP-VO-006/007 baselines."
for w in $WINDOWS; do
  for m in $MODELS; do
    ./gradlew.bat run -PmainClass=org.boofcv.evaluation.VoRunnerApp \
        --args="--config evaluation/eval_configs/exp-vo-010/run-${w}-${m}-rigid-refine.json" -q
  done
done

echo "### Phase 4 -- how far refinement moved the model, per frame"
RUNS=""
for w in $WINDOWS; do for m in $MODELS; do
  RUNS="${RUNS}${RUNS:+,}runs/${w}-${m}-rigid-refine-v1"
done; done
"$PY" evaluation/tools/exp_vo_010/refinement_delta.py --runs "$RUNS" --out "$OUT/refinement_delta.json"

echo "### Phases 5, 6, 7, 10 -- every arm scored, heading split into its three components"
"$PY" evaluation/tools/exp_vo_010/analyse.py --out "$OUT/comparison.json"

echo "### Phase 7 -- is any XY change mediated by heading? (EXP-VO-008's substitution)"
for w in hkairport01-b amtown01-c; do
  for m in $MODELS; do
    "$PY" evaluation/tools/exp_vo_008/substitution.py \
        --dataset "datasets/${w}" --run "runs/${w}-${m}-rigid-refine-v1" \
        --label "${w}-${m}-refine" --out "$OUT/subst_${w}_${m}_refine.json"
  done
done

echo "### Phase 9 -- matched-conditions timing, all four arms back to back"
"$PY" evaluation/tools/exp_vo_010/timing_compare.py --out "$OUT/timing.json" \
    --hkairport-root "$HKAIRPORT_ROOT"

echo "### Figures"
"$PY" evaluation/tools/exp_vo_010/plot_exp_vo_010.py --out "$OUT/figures"

echo "### done -- artifacts in $OUT"
