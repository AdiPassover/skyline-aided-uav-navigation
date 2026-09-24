# Reproducing the reported results

This maps the results of Chapter 4 of the thesis to the code that produced them. There is no single
command that rebuilds the chapter. Each result is a pipeline of the form *ingest → Java run → Python
analysis*, and most analysis scripts read the run records and intermediate outputs of earlier steps
from `runs/` and `evaluations/` (see `docs/data.md`).

The simulator recordings are archived at <https://doi.org/10.5281/zenodo.22802397>;
[`docs/data.md`](data.md) describes where the code expects them, and
[`docs/workflows.md`](workflows.md) explains the individual commands.
All commands run from the repository root unless stated otherwise. `$PY` is a Python with
`evaluation/requirements.txt` and `skyline/requirements.txt` installed, and `$DATA` the directory
the archive was extracted to. The Java drivers are run from the installed distribution:

```bash
./gradlew installDist
CP="build/install/skyline-aided-uav-navigation/lib/*"
java -cp "$CP" org.boofcv.evaluation.VoRunnerApp --config <run-config.json>
```

Identifiers such as `EXP-INT-003` or `DEC-VO-010` in file names, configs and comments refer to
experiment and decision records of the development log. Those records are not part of this
repository; the thesis is the citable account of each experiment.

Several thesis figures were restyled for the manuscript outside this repository. The scripts below
produce the numbers and research versions of those figures, not the final typeset images.

---

## Integrated navigation (Section 4.5)

Thirteen simulated trajectories from twelve recordings, flown in the UE5 environment. Each dataset is
archived as `INT/<dataset id>/` and read from `datasets/<dataset id>/`. The simulator run each one was
recorded as:

| dataset id | simulator run | role |
|---|---|---|
| `fig8-flat-const-v1` | `figure_8_flat_surface/Run_20260907_140138` | development |
| `fig8-flat-vary-v1` | `figure_8_flat_surface/Run_20260907_140549` | development |
| `mtn-r1-const-v1` | `eight_figure_mountains_less_steep/Run_20260907_145801` | development |
| `mtn-r2-vary-v1` | `eight_figure_mountains_less_steep/Run_20260907_150845` | development (also the injected-loss track) |
| `mtn-r3-vary-v1` | `eight_figure_mountains_less_steep/Run_20260907_151617` | development |
| `interesting-r1-vary-v1` | `interesting_path/Run_20260907_165110` | long evaluation recording |
| `interesting-r3-const-v1` | `interesting_path/Run_20260907_175403` | low-drift control |
| `easier-sq-const-v1` | `hopefully_easier/Run_20260907_231953` | development |
| `ho1-mtn-fig8-vary-v1` | `held_out1/Run_20260908_130747` | held out |
| `ho1-mtn-fig8-scaled-vary-v1` | `held_out1/Run_20260908_131337` | held out |
| `ho1-vil-fig8-vary-v1` | `held_out1/Run_20260908_133522` | held out |
| `ho1-mtn-trinity-vary-v1` | `held_out1/Run_20260908_141325` | held out |

The archived `ingested_dataset/` already holds the frames, ground truth, height channel and skyline
profiles, so reproduction starts at step 4. Steps 1–3 record how those files were produced.

**1. Ingest the nadir stream** (`--role constant_height` for `*-const-*`, `varying_height` for
`*-vary-*`). This step reads the simulator's raw nadir export (`vo/`), which is not in the archive;
`ingest_provenance.json` records the SHA-256 of every file it read.

```bash
$PY evaluation/tools/int/ingest_ue_run_int.py --source <simulator run directory> \
    --dataset-id ho1-mtn-fig8-vary-v1 --out-root datasets --role varying_height
```

**2. Ingest both horizon views** into observation sessions, one config per recording and view
(run from `skyline/`). The configs name the capture's development location in `run_dir`; use copies
that point at `INT/<dataset id>/raw_skyline_capture/` (`docs/workflows.md`, Skyline profiles). The
two `fig8-flat-*` datasets reuse the sessions ingested by `sky_matcher_resolution.py --stage ingest`
into `observations_sim_fig8/`.

```bash
cd skyline
$PY -m hsreloc.simret.cli ingest --config <copy of ../evaluation/eval_configs/int/skyline-sessions/Run_20260908_130747-north.json>
$PY -m hsreloc.simret.cli ingest --config <copy of ../evaluation/eval_configs/int/skyline-sessions/Run_20260908_130747-west.json>
cd ..
```

**3. Export the skyline profiles**, paired to nadir frames by the simulator's own capture identity.
The output reproduces the archived `skyline_profiles.csv` byte for byte.

```bash
$PY evaluation/tools/int/export_skyline_profiles.py \
    --session-dir observations_sim_int/Run_20260908_130747-north \
    --west-session-dir observations_sim_int/Run_20260908_130747-west \
    --source sim_exact \
    --vo-frames datasets/ho1-mtn-fig8-vary-v1/frames.csv \
    --sim-observations "$DATA/INT/ho1-mtn-fig8-vary-v1/raw_skyline_capture/skyline/observations.csv" \
    --out <output directory>
```

**4. Run the local-only baseline and the live integrated arm** for each dataset. The configs are
under `evaluation/eval_configs/int/exp-int-00{1,2,3}/` (`run-<id>-vo-only.json`,
`run-<id>-int-c0.json`; `exp-int-001/run-mtn-r2-vary-v1-vo-only-synthetic-loss.json` injects the two
losses). The frozen acceptance policy used by every integrated arm is
`evaluation/eval_configs/int/exp-int-002/reloc-c0-primary-retry20.json`
(SHA-256 `b5457092c59767319bf8c40d993549c667ff4238b13eb0f7789660643ef80399`).

```bash
java -cp "$CP" org.boofcv.evaluation.VoRunnerApp --config evaluation/eval_configs/int/exp-int-003/run-ho1-mtn-fig8-vary-v1-vo-only.json
java -cp "$CP" org.boofcv.evaluation.VoRunnerApp --config evaluation/eval_configs/int/exp-int-003/run-ho1-mtn-fig8-vary-v1-int-c0.json
```

**5. Compare the correction policies** (Tables "Correction policies compared" and "Trajectory error
under each correction policy", the recovery results). The script replays the frozen policy and the
two top-match policies in `evaluation/eval_configs/int/claim-closure-2026-09/` over the thirteen
recorded local-only tracks with `RelocalizationReplayApp` (byte-identical to the live loop, which
`RelocalizationReplayAppTest` asserts), then scores each with `evaluate_int_arms.py`:

```bash
$PY evaluation/tools/claim_closure/int_comparison_arms.py --out evaluations/claim-closure-2026-09/int
$PY evaluation/tools/claim_closure/int_figures.py --out evaluations/claim-closure-2026-09/int
```

The script reads the twelve datasets from `datasets/` and the thirteen local-only run records from
`runs/`, so first run the `run-<id>-vo-only.json` config of each dataset above and
`exp-int-001/run-mtn-r2-vary-v1-vo-only-synthetic-loss.json`. It calls the installed Java
distribution (`./gradlew installDist`).

A single dataset can be scored directly:

```bash
$PY evaluation/tools/int/evaluate_int_arms.py --dataset datasets/ho1-mtn-fig8-vary-v1 \
    --vo-only runs/exp-int-003-ho1-mtn-fig8-vary-v1-vo-only \
    --int int_c0=runs/exp-int-003-ho1-mtn-fig8-vary-v1-int-c0 --out evaluations/exp-int-003/ho1-mtn-fig8-vary-v1
```

The relocalization layer, its replay and the evaluation are deterministic functions of a local-only
track and its profiles. The local-only track itself is reproducible bit for bit on one machine but
not across processors or JVM builds (`docs/workflows.md`), so a re-run elsewhere can differ slightly
from the reported numbers. For example, re-running `ho1-mtn-fig8-vary-v1` from the archive on a second
machine gave an ATE of 88.4 m local-only and 54.8 m integrated (−38.0 %), against the reported
89.2 m and 55.2 m (−38.1 %), with the same two accepted corrections.

The gate values of the frozen policy were selected on the development recordings with
`evaluation/tools/int/dev_calibration_sweep.py`. The remaining scripts in `evaluation/tools/int/`
(`gt_revisit_events.py`, `snap_longitudinal.py`, `revisit_opportunity_table.py`,
`vo_quality_report.py`, `synthetic_drift_sweep.py`) are the diagnostics behind the discussion of
when a genuine correction helps or harms.

---

## Skyline place evidence (Section 4.4)

Run from `skyline/`. The three two-view study scripts have `audit`, `ingest`, `score` and `report`
stages; `ingest` also writes the frame list for SegFormer inference, which runs in its own
environment (`skyline/README.md`) between `ingest` and `score`. Profiles from the simulator's exact
masks need no inference.

| result | command(s) | data (placed as in `docs/data.md`) |
|---|---|---|
| One view against two, false acceptances on the hard city set | `scripts/sky_dual_study.py --config configs/sky-dual.json`, then `scripts/sky_dual_hardcity.py --config configs/sky-dual-hardcity.json`, then `scripts/sky_matcher_resolution.py --config configs/sky-matcher-resolution.json` (stages `audit ingest score score-terrain score-hardcity report`); consolidated by `../evaluation/tools/claim_closure/sky_consolidate.py` | `simulator_skyline_data_both_directions/`, including the two figure-eight captures |
| Recognition across the nine rendered conditions | `PYTHONPATH=. python scripts/sim_ext_inventory.py --root ../simulator_skyline_data_extended --out ../evaluations/sim-ext-inventory`, then `python -m hsreloc.simret.cli` with `configs/sim-final.json`: `validate`, `ingest`, `extract-dp`, `silver-list` (+ `scripts/silver_infer.py --config configs/silver-infer-ext.json`), `ext-index`, `ext-sets`, then `ext-run` and `ext-exp-exactpose` with `--population dev` (village) and `--population final` (mountains, city) | `simulator_skyline_data_extended/` |
| Sensitivity to the assumed viewing direction | `PYTHONPATH=. python ../evaluation/tools/claim_closure/sky_azimuth_tolerance.py --out ../evaluations/claim-closure-2026-09/sky` (reads the outputs of the two-view studies) | `simulator_skyline_data_both_directions/` |
| Recognition range from a stored reference | `scripts/sky_recog_radius.py --config configs/sky-recog.json --stage all` | the ingested appearance batch (`observations_sim/`, `evaluations/sim-ext-index/`) and its SegFormer label maps |
| Reference storage | `python ../evaluation/tools/claim_closure/sky_compactness.py --out ../evaluations/claim-closure-2026-09/sky` | `simulator_skyline_data_both_directions/`, including the two figure-eight captures |

The studies are ordered: `sky_dual_hardcity.py` scores against the memory built by
`sky_dual_study.py` and aborts unless it reproduces that study's frozen scores, and
`sky_matcher_resolution.py` rebuilds both and checks them the same way. `sky_dual_study.py` also
reads the site catalogue that `ext-index` builds from the appearance batch
(`evaluations/sim-ext-index/index.csv`), so that step comes first.

`sky_dual_study.py` takes every `Run_*` directory under the three level directories in its `audit`
and `ingest` stages. The study used the nine runs recorded on 2026-09-06; the archive's
`large_flat_city/` also holds `Run_20260907_125137`, the hard-city run that `sky_dual_hardcity.py`
reads on its own. Add that run to `simulator_skyline_data_both_directions/large_flat_city/` only after
`sky_dual_study.py --stage ingest`; with it present, the audit counts ten runs instead of nine.

The `--population final` commands compare everything they build with the committed pre-registration:
the digest over the appearance batch, which the archive reproduces, the index and task digests, the
matcher and acceptance settings, and the curve sources, including a digest of the SegFormer label
maps. Label maps that differ from the pre-registered ones in any pixel make the check fail, and the
guard treats that as a new experiment rather than a re-run.

---

## Local visual motion estimation (Section 4.2)

Recorded-flight results use MARS-LVIG, fetched from its MCAP mirror on Hugging Face
(`DapengFeng/MCAP`, `mars_lvig/<scene>/<scene>_0.mcap`). The windows (UTC seconds):

| window | dataset id | scene | cruise window |
|---|---|---|---|
| development (`hk-b`) | `hkairport01-b` | HKairport01 | 1671606510.406 – 1671607126.188 |
| prefix of `hk-b` | `hkairport01-a` | HKairport01 | 1671606510.406 – 1671606690.406 |
| validation (`am-c`) | `amtown01-c` | AMtown01 | 1658137128.011 – 1658137758.011 |
| metric scale (`am-d`) | `amtown01-d` | AMtown01 | 1658137128.011 – 1658138317.359 |
| health, held out (`hk-03`) | `hkairport03` | HKairport03 | 1671607449.978 – 1671607667.372 |
| health, held out (`am-03`) | `amtown03` | AMtown03 | 1658131910.487 – 1658132361.268 |

Datasets are built with `evaluation/tools/exp_vo_007/ingest_sequence.py`;
`exp_vo_007/fetch_amtown01.sh` and `exp_vo_013/fetch_amtown01_full.sh` record the complete argument
sets for the AMtown01 windows, and `exp_vo_007/derive_yaw_offset.py` derives the per-sequence
heading offset those scripts require.

**Camera intrinsics.** MARS-LVIG's MCAP files carry no camera calibration. The intrinsics come
from UAVScenes (Hugging Face `sijieaaa/UAVScenes`), which republishes MARS-LVIG with per-frame
calibration. They are in the `sampleinfos_interpolated.json` file for each scene inside
[`interval5_CAM_LIDAR.zip`](https://huggingface.co/datasets/sijieaaa/UAVScenes/resolve/main/interval5_CAM_LIDAR.zip)
(28,682,115,865 bytes), at `interval5_CAM_LIDAR/interval5_<scene>/sampleinfos_interpolated.json`.
The zip is large, but a single entry can be read with HTTP range requests, as
`exp_vo_007/probe_candidates.py` does. Every entry holds a `P3x3` matrix plus `K1`–`K3`, `P1`,
`P2`, `Width` and `Height`. Within each file these values are identical for every frame, and the
extraction step is to copy them:

| scene | entries | fx = fy | cx | cy |
|---|---|---|---|---|
| HKairport01, HKairport03 | 7,199 / 3,023 | 1471.0653076171875 | 1172.3576676454904 | 1046.3674075128438 |
| AMtown01, AMtown03 | 12,944 / 5,599 | 1469.4898681640625 | 1174.0027077275518 | 1049.91204868583 |

SHA-256 of the extracted files:

```text
dc53d5fead50913b7730e35eff3b908ed89c0e29ddebd180f1555034ace808d6  interval5_HKairport01/sampleinfos_interpolated.json
8f5e175681289ad7ba3ea367ade61a0727e6fdf26467c2021d66feafd6a6ff97  interval5_HKairport03/sampleinfos_interpolated.json
36df2da500f2f8eaf417760fefc77e895eab9d096752795d9540208055eac64a  interval5_AMtown01/sampleinfos_interpolated.json
c373b35bc746aca823261675de6530cff7ba1957ae408b1fa1c11b61a77d4ba4  interval5_AMtown03/sampleinfos_interpolated.json
```

Distortion is zero and the image is 2448 × 2048 in all four scenes. For the HKairport windows, when
`--intrinsics-json` is omitted, `ingest_sequence.py` uses the same values built into
`naveval.ingest_mars_lvig.CAMERA_INTRINSICS`.
For the AMtown windows, pass `--intrinsics-json <file>` (the `INTRINSICS` variable of the two
`fetch_amtown01*.sh` scripts), pointing at a JSON object with these keys:

```json
{"fx": 1469.4898681640625, "fy": 1469.4898681640625, "cx": 1174.0027077275518,
 "cy": 1049.91204868583, "k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0,
 "width": 2448, "height": 2048, "source": "UAVScenes sampleinfos_interpolated.json for AMtown01"}
```

The ingest writes the intrinsics into `dataset.json`. The metric-readout configs declare the same
fx as `fx_native_px`. No UAVScenes file is redistributed here; its license is on the
[dataset card](https://huggingface.co/datasets/sijieaaa/UAVScenes).

| result | code |
|---|---|
| Representation × motion model table, trajectory figure | capture configs `evaluation/eval_configs/exp-vo-{004,006,007,009}/`, evaluation configs `evaluation/configs/eval-*.json` (`python -m naveval.evaluate`), `evaluation/tools/exp_vo_006/rigid_readout.py`, `exp_vo_007/analyse.py` (or `exp_vo_007/run_all.sh`), `exp_vo_009/analyse.py`; tables collected by `vo_closure/canonical_results.py`; figure `thesis_figures/trajectory_comparison.py` |
| Residual accumulation figure | `thesis_figures/residual_accumulation.py`, `vo_closure/closure_figures.py` |
| Rigid-reduction granularity | `exp_vo_008/schedule_sweep.py` |
| Motion model as a sensitivity axis | `exp_vo_005/` (rendered planar imagery, ±10° roll), `exp_vo_009/` (`run_all.sh`; variance theory, rotation across models, timing) and `org.boofcv.stitching.diagnostics.ModelRotationMonteCarloApp` |
| Heading error, ground-truth substitution, refinement | `exp_vo_010/` (`run_all.sh`; `heading_metrics.py`, `refinement_delta.py`, `analyse.py`), `exp_vo_008/heading_baseline.py`, `exp_vo_008/substitution.py`, `org.boofcv.stitching.diagnostics.RefinementMonteCarloApp` |
| Visual scale bias, time-reversed replay | `exp_vo_011/` (see its README), `org.boofcv.stitching.diagnostics.ScaleBiasMonteCarloApp`, `ScaleBiasParityApp` |
| External height on analytically rendered terrain (arms A/B/C) | `exp_vo_012/synth_terrain.py` (renderer), configs `evaluation/eval_configs/exp-vo-012/`, `exp_vo_012/analyse.py`, `exp_vo_012/emit_tables.py` |
| Constant vs varying height in the game engine | `exp_vo_014/ingest_ue_run.py` on the simulator runs `Run_20260904_172033` (constant) and `Run_20260904_174825` (varying), configs `evaluation/eval_configs/exp-vo-014/`, `exp_vo_014/analyse.py`. These two recordings are not in the Zenodo archive. |
| Height channels over the `am-d` cruise | `exp_vo_013/` (`altitude_sources.py` reads the LiDAR, `analyse.py`, `plot_exp_vo_013.py`), config `evaluation/eval_configs/exp-vo-013/` |

The runtime metric and heading readouts are checked against an independent Python implementation:
`evaluation/tools/vo_metric_runtime/` holds the frozen reference that
`MetricReadoutPythonParityTest` reads, and `vo_heading_runtime/` the replay inputs and analysis for
the heading channel.

---

## Estimator health (Section 4.3)

Enable the observer with `"diagnostic_refit_sidecar": true` in a run config (it writes
`diagnostic_refit.csv` and leaves `frames.csv` unchanged).

| result | code |
|---|---|
| Target validity (RTK heading resolution, fused attitude) | `evaluation/tools/exp_conf_002/rtk_target_audit.py` |
| Independent reference panel, two groups | `exp_conf_002/panel.py` (needs imagery), `exp_conf_002/panel_analysis.py` |
| Audit examples figure | `exp_conf_002/build_audit.py`, `exp_conf_004/render_corrected_figures.py`, `reports/thesis_conf_figures.py` |
| Refit disagreement on the unseen flights | configs `evaluation/eval_configs/exp-conf-005/`, `exp_conf_005/analysis.py` |
| Common-mode analysis and refit counterfactual | `exp_conf_003/` (`phase_a.py` needs imagery), `exp_conf_004/analysis.py` |
| Comparison of eleven candidate signals | `claim_closure/conf_benchmark.py` |
| Navigation unchanged with the observer on and off | `exp_conf_001/verify_bit_identity.py` |

---

## Computational cost (Section 4.6)

`evaluation/tools/claim_closure/resources_summary.py` assembles the table from the per-frame timings
in the run records, `exp_vo_009/timing_compare.py` and `exp_vo_010/timing_compare.py` (motion stage),
the `timing` configs of `exp-conf-005` (observer), and the integration-layer benchmark:

```bash
java -cp "build/install/skyline-aided-uav-navigation/lib/*" evaluation/tools/claim_closure/IntLayerBench.java \
    <vo-run> <policy.json> <skyline_profiles.csv> <repetitions> <out.json>
```

Timings were taken on one development laptop and describe that machine only.
