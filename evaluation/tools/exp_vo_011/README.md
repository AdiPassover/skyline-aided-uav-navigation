# `exp_vo_011` — visual-scale bias attribution

Tooling for `EXP-VO-011`. Everything here is
analysis and diagnostics; **no estimator parameter, default, or navigation behaviour is changed by
any of it.**

## What each module does

| module | phase | reads | writes |
|---|---|---|---|
| `scale_series.py` | — | committed `runs/*/logical_transform.csv`, `datasets/*/{groundtruth,attitude}.csv`, `evaluations/exp-vo-007/candidate_probe.json` | (library) |
| `scale_budget.py` | 2 | the above | `scale_budget.json` |
| `covariates.py` | 6 | the above | `covariates.json` |
| `flow_direction.py` | 6b | committed run records, forward and reversed | (library) |
| `anisotropy_axis.py` | 6c | committed run records | `anisotropy_axis.json` |
| `build_reversed.py` | 6b | a real dataset | `datasets/<window>-reversed/` |
| `render_distorted.py` | 3b | `exp_vo_005/synth_planar.py` | `datasets/synth-dist-*/`, configs |
| `budget.py` | 8 | `mechanisms.csv`, `parity.csv`, run records | `scale_bias_budget.json` |
| `analyse.py` | all | the above + `mechanisms.csv` | `report.json` |
| `plot_exp_vo_011.py` | — | the above | `figures/fig1..fig12` |
| `test_exp_vo_011.py` | — | — | (pytest) |

The synthetic mechanism sweep itself is Java —
`org.boofcv.stitching.diagnostics.ScaleBiasMonteCarloApp` — because it drives the *real* BoofCV
RANSAC and the real model generators, which is the whole point: a mechanism is measured through the
estimator that ships, not through a reimplementation of it.

## Three conventions the whole record rests on

1. **`inc_log_scale` is read on `D_k^-1`, not `D_k`.** `RigidNavigationState.observe` composes
   `F_{k-1} . F_k^-1` and takes `log sqrt(|det J|)` at the **image centre**. Under `LIT-VO-003`
   eq. (2) that means **positive = the estimator believes the camera is climbing**. Two of
   `EXP-VO-004` R2's columns have opposite signs; this one is stated in `scale_series.py` and used
   everywhere.
2. **`rigid_y` is the pose's `y`, which is `-centreY`.** `flow_direction.image_flow` un-negates it
   before inverting the heading rotation. Skipping that step leaves the recovered vector a
   *reflection* of the true one, whose length still matches `inc_flow_px` exactly — so the obvious
   magnitude check passes while every angle is wrong. It happened; the comment marks it.
3. **The row filter drops `init` and `restart`, keeps `recenter`.** A restart row's increment is not
   that frame's motion estimate (the estimator re-initialises inside `processFrame`); a recenter
   row's is, because the fold is exact and the frame still moves (`EXP-VO-004` R3).

## Reproduction

Committed artifacts are enough for everything except the estimator runs. From the repository root:

```bash
# Phases 2, 6, 6c and the figures - committed artifacts only, no imagery, no estimator
python evaluation/tools/exp_vo_011/scale_budget.py --out evaluations/exp-vo-011
python evaluation/tools/exp_vo_011/covariates.py   --out evaluations/exp-vo-011
python evaluation/tools/exp_vo_011/anisotropy_axis.py --out evaluations/exp-vo-011

# Phases 3-5, the synthetic mechanism grid, through the real estimators
JAVA_HOME="C:/Program Files/Java/jdk-19" ./gradlew.bat run \
  -PmainClass=org.boofcv.stitching.diagnostics.ScaleBiasMonteCarloApp \
  --args="--out evaluations/exp-vo-011/mechanisms.csv --trials 250"

# Phase 8's parity test: which mechanisms are ODD under time reversal, plus the two arms whose
# bound is set by Monte Carlo error and therefore get 8,000 trials of their own
JAVA_HOME="C:/Program Files/Java/jdk-19" ./gradlew.bat run \
  -PmainClass=org.boofcv.stitching.diagnostics.ScaleBiasParityApp \
  --args="--out evaluations/exp-vo-011/parity.csv --trials 400 --precision-trials 8000"

# Phase 6b, the time-reversed replay (needs the source imagery)
python evaluation/tools/exp_vo_011/build_reversed.py --window hkairport01-a \
  --source datasets/hkairport01-a
python evaluation/tools/exp_vo_011/build_reversed.py --window amtown01-c --source datasets/amtown01-c
JAVA_HOME="C:/Program Files/Java/jdk-19" ./gradlew.bat run \
  -PmainClass=org.boofcv.evaluation.VoRunnerApp \
  --args="--config evaluation/eval_configs/exp-vo-011/run-hkairport01-a-reversed-affine-rigid.json"
# ... and the other two configs in that directory

# Phase 3b, the rendered distortion arms
python evaluation/tools/exp_vo_011/render_distorted.py --out datasets \
  --configs evaluation/eval_configs/exp-vo-011
# ... then VoRunnerApp over run-synth-dist-{none,distorted,rectified}-{affine,homography}.json

# Assemble and plot
python evaluation/tools/exp_vo_011/budget.py  --out evaluations/exp-vo-011
python evaluation/tools/exp_vo_011/analyse.py --out evaluations/exp-vo-011
python evaluation/tools/exp_vo_011/plot_exp_vo_011.py --out evaluations/exp-vo-011/figures
```

`run_all.sh` runs the analysis half end to end. Tests:

```bash
cd evaluation; python -m pytest tools/exp_vo_011 -q
JAVA_HOME="C:/Program Files/Java/jdk-19" ./gradlew.bat test --tests "org.boofcv.stitching.ScaleBiasMonteCarloTest"
```

## Two things that are deliberately *not* here

- **No real-imagery rectification arm.** Both MARS-LVIG datasets declare `k1 = k2 = k3 = p1 = p2 = 0`
  — a *declared* zero from the vendor calibration, not a measured one. There is no distortion model
  to rectify with, so a "corrected" real arm would apply the identity and measure nothing. The
  magnitude question is carried by the synthetic arms, which is the honest place for it.
- **No per-frame feature-radius covariate.** `frames.csv` and `logical_transform.csv` carry track and
  inlier *counts* but no spatial distribution, so the distortion-sensitivity covariate the brief
  asked for is not measurable without a new probe. The time-reversal arm answers the same underlying
  question — is the bias odd in the direction of travel? — directly and with no modelling, and
  `fig8` reports it in that figure's place.
