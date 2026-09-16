# `EXP-VO-010` tooling — does refitting the RANSAC model over its inlier set help?

Experiment drivers for `EXP-VO-010`. Nothing
here is a library: each script answers one phase of that experiment and writes one artifact under
`evaluations/exp-vo-010/`. `evaluation/naveval/` is imported read-only and is **not** modified.

Reproduce the whole thing with `bash run_all.sh` from the repository root — it runs the gates first
and stops if one fails.

## The ablation needs no estimator change

`VoRunnerConfig.refineEstimate` already existed and defaults to `false`;
`VoRunnerApp.{homography,affine}Motion` pass it straight to `FactoryMotion2D.createMotion2D`. So the
experiment is pure configuration, and the `refine = false` side is the **committed
`EXP-VO-006`/`EXP-VO-007` baselines**, reused verbatim rather than re-run.

What *was* added is observational: `refinement_sidecar` (`VoRunnerConfig`) makes `VoRunner` write
`refinement.csv` beside `frames.csv`, recording per accepted frame RANSAC's winning **minimal-sample**
model, the model actually shipped, and the inlier count the refiner was given. No model is
re-estimated — the `DEC-VO-001` provenance rule. Java side:
`org.boofcv.stitching.{RefinementDiagnostics,RefinementSnapshotTrackerKey}` and
`StitchingFactory.createProbedMotion2D`.

**Only `homography` and `affine` support it.** `GenerateSimilarity2D` implements `ModelGenerator`
only, so RANSAC has no final model to refine for the 4-DoF arm; the flag is *rejected* for it rather
than silently ignored.

## Scripts

| script | phase | what it produces |
|---|---|---|
| `heading_metrics.py` | 5 | the heading decomposition — raw yaw RMSE (continuity), the constant frame offset `EXP-VO-009` R2 identified, the offset-aligned tracking RMS underneath it, and per-frame/cumulative drift. Circular-safe throughout. |
| `refinement_delta.py` | 4 | `refinement_delta.json` — the distributions of `Δrotation`, `Δtranslation`, `Δlog-scale` between the minimal-sample and refined models, and how each relates to inlier count. |
| `analyse.py` | 6, 7, 10 | `comparison.json` — twelve arms (3 windows × 2 models × 2 refine settings), all re-scored through `EXP-VO-007`'s own `evaluate`/`diagnostics`, plus the heading split, support and cost. |
| `timing_compare.py` | 9 | `timing.json` — all four arms timed **back to back over the same window in one sitting**, with `--motion-timing` isolating KLT + RANSAC + refinement from mosaic rendering. |
| `plot_exp_vo_010.py` | — | `figures/fig{1..5}_*.png` — the record's ten required panels. |
| `test_exp_vo_010.py` | — | known-answer tests for the drivers, notably the heading decomposition. |

Phase 3's Monte Carlo is deliberately **not** here: it lives in Java
(`org.boofcv.stitching.diagnostics.RefinementMonteCarlo{,App}`) so that it drives the real
BoofCV/ddogleg fitters through the real `Ransac` with its real seed. Phase 7 reuses
`evaluation/tools/exp_vo_008/substitution.py` unchanged.

## The heading metric, and why it is validated twice

`EXP-VO-009` R2 found that the aligned yaw RMSE the project had quoted since `EXP-002` is **≈ 95 % a
constant frame offset**, and `EXP-VO-010` H2 is decided on the corrected quantity — so that quantity
had to be trustworthy before any flight number was read. `test_exp_vo_010.py` validates it against
(a) a second, independently written implementation that minimises over the offset numerically rather
than using the closed-form circular mean, and (b) known-answer synthetic series including ones that
straddle the 0°/360° branch cut, where a naive arithmetic mean is wrong by up to 180°.

**`naveval` is untouched and no historical metric is rewritten.** `raw_yaw_rmse_deg` is reproduced
and reported alongside, for continuity with every prior record.

## Two properties worth knowing before reading any number

**The arms are not estimator-identical, and cannot be.** Unlike `EXP-VO-009`'s readout comparisons,
which ran on bit-identical recorded transforms, changing the shipped model changes `worldToCurr`,
hence `checkLargeMotion` and the mosaic re-origin schedule, hence the track re-anchoring
`setToFirst()` performs. Track counts, inlier counts and event schedules therefore diverge after the
first re-origin **by construction**; the record's H6 is about degradation, not identity.

**Refinement cannot reclassify inliers within a frame.** It runs after `Ransac.selectMatchSet`, and
the `lastUsed` marking loop that follows re-reads the pre-refinement match set. Source-verified;
`RefinementSemanticsTest` makes it executable.

## What is deliberately not done

No tuning of anything against ground truth. No tracker change, no motion-model change, no
`RIGID_MOTION` change, no navigation-default change (`refineEstimate` stays `false` in
`VoRunnerConfig`'s default, so every pre-existing config and committed baseline is unaffected). No
`naveval` modification. No `FULL_LOGICAL` re-runs. No second broad yaw investigation — the record
carries a binding stopping rule.
