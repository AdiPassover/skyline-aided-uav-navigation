#!/usr/bin/env bash
# EXP-VO-011: the analysis half, end to end, from committed artifacts.
#
# The estimator runs (the synthetic mechanism grid, the time-reversed replays, the rendered
# distortion arms) are NOT re-run here -- they need Gradle and, for two of them, imagery that is
# gitignored. Their commands are in README.md. This script regenerates every number and every figure
# the record quotes from what is committed, and then verifies the record against them.
set -euo pipefail
cd "$(dirname "$0")/../../.."

PY="${PYTHON:-python}"
OUT=evaluations/exp-vo-011

$PY evaluation/tools/exp_vo_011/scale_budget.py     --out "$OUT"
$PY evaluation/tools/exp_vo_011/covariates.py       --out "$OUT"
$PY evaluation/tools/exp_vo_011/anisotropy_axis.py  --out "$OUT"
$PY evaluation/tools/exp_vo_011/budget.py           --out "$OUT"
$PY evaluation/tools/exp_vo_011/analyse.py          --out "$OUT"
$PY evaluation/tools/exp_vo_011/plot_exp_vo_011.py  --out "$OUT/figures"

( cd evaluation && $PY -m pytest tools/exp_vo_011 -q )
