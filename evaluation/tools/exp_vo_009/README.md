# `EXP-VO-009` tooling — direct Sim(2) fitting versus affine/homography-then-projection

Experiment drivers for `EXP-VO-009`.
Nothing here is a library: each script answers one phase of that experiment and writes one artifact under
`evaluations/exp-vo-009/`. `evaluation/naveval/` is imported read-only and is **not** modified.

Reproduce the whole thing with `bash run_all.sh` from the repository root —
it runs the gates first and stops if one fails.

## The estimator side is not here

The similarity motion model itself is Java, in `src/main/java/org/boofcv/stitching/`
(`Sim2_F64`, `GenerateSimilarity2D`, `DistanceSimilarity2DSq`, `ModelManagerSim2_F64`,
`SimilarityStitchingTransform`, `MotionModelSupport.SIMILARITY`), selected by
`VoRunnerConfig.motion_model = "similarity"`. See `DEC-VO-006` for why it is written rather than
taken from BoofCV, and `LIT-VO-005` §5 for the source reading that forced that.

Configs: `evaluation/eval_configs/exp-vo-009/run-<window>-similarity-rigid.json` (capture) and
`evaluation/configs/eval-<window>-similarity-rigid.json` (evaluation).

## Scripts

| script | phase | what it produces |
|---|---|---|
| `variance_theory.py` | gate 1 | `variance_theory_*.json` — verifies `LIT-VO-005` eq. (8), the closed-form rotation-variance ratio `(2 + κ + 1/κ)/4`, per-sample by Monte Carlo. Pure algebra check: no VO, no imagery. |
| `check_reproduction.py` | gate 4b | asserts a re-run of `runs/hkairport01-a-run-v1` is bit-identical, so the third model has not perturbed the arm every committed baseline rests on. |
| `rotation_across_models.py` | 6 | `rotation_<window>.{json,npz}` — per-frame rotation error of every model against RTK heading: mean, median, SD, robust MAD-σ, bias `t`, lag-1 autocorrelation, cumulative error, tails, and the accumulated error as a multiple of its random-walk prediction. **This is the direct H1 test.** |
| `analyse.py` | 7, 9 | `comparison.json` — five arms per window (legacy ×2, rigid ×3), all re-scored through `EXP-VO-007`'s own `evaluate`/`diagnostics`, plus support and cost. |
| `timing_compare.py` | 9 | `timing.json` — all three models timed **back to back over the same window in one sitting**, with `--motion-timing` isolating KLT+RANSAC from mosaic rendering. |
| `plot_exp_vo_009.py` | — | `figures/fig{1,2,3,4}_*.png` — the record's nine required panels. |
| `test_exp_vo_009.py` | — | known-answer tests for the drivers themselves (11). `python -m pytest tools/exp_vo_009 -q` from inside `evaluation/`. |

Phase 5's Monte Carlo is deliberately **not** here: it lives in Java
(`org.boofcv.stitching.diagnostics.ModelRotationMonteCarlo{,App}`) so that it drives the real
BoofCV/ddogleg estimators and the real `Ransac` with its real seed, rather than a second Python
implementation whose own variance would confound the answer. Phase 8 reuses
`evaluation/tools/exp_vo_008/substitution.py` unchanged, because the similarity run's sidecar has
the same schema.

## Two properties worth knowing before reading any number

**The affine and similarity arms are RANSAC-sample-paired.** `Ransac.setModel` takes its sample size
from the generator's `getMinimumPoints()`, and `checkTrialGenerators` derives one `Random` per
*trial* independently of that size. `GenerateSimilarity2D` reports **3**, matching
`GenerateAffine2D`, so at an equal track list the two arms draw *identical* index sequences. The
homography arm is not paired with them (its minimum is 4) — as `LIT-VO-001` already recorded.

**The synthetic Monte Carlo is paired too.** Every configuration reseeds from the same constant, so
all four arms see byte-identical correspondence sets trial by trial.

## What is deliberately not done

No tuning of anything against ground truth. `refineEstimate` stays `false` in every flight arm even
though `LIT-VO-005` §4 argues it may matter more than the model class — its effect is bounded on
**synthetic** data only. No default moves. No evaluator change, no new metric, no frame-rate or
resolution change, no trajectory sub-selection. `AMtown01` is reported whatever it does.
