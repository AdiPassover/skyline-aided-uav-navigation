# Workflows

How to build the software and run its main workflows on the published data.
[`docs/data.md`](data.md) describes the data layout; [`docs/reproducing.md`](reproducing.md) maps
each thesis result to its recordings, configs and scripts.

Commands run from the repository root unless a step says otherwise, and are written for a POSIX
shell. On Windows use `gradlew.bat`, `.venv\Scripts\python.exe`, and `$env:NAME = 'value'` to set an
environment variable.

## Build and test

Java needs JDK 19, with `JAVA_HOME` pointing at the JDK root; Gradle comes with the wrapper.

```bash
./gradlew build                                        # compile and run the JUnit tests
./gradlew test --tests "org.boofcv.relocalization.*"   # one package
./gradlew installDist                                  # jars in build/install/skyline-aided-uav-navigation/lib/
CP="build/install/skyline-aided-uav-navigation/lib/*"  # classpath used below
```

The test report is `build/reports/tests/test/index.html`. `JavaProducedRunFixtureTest` rewrites
`evaluation/tests/fixtures/java_produced_run/` with the local JVM version and timings; discard that
change.

Python needs 3.12. `python` below means this environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r evaluation/requirements.txt -r skyline/requirements.txt
(cd evaluation && ../.venv/bin/python -m pytest -q)
(cd skyline && ../.venv/bin/python -m pytest tests -q)
```

Tests that need recorded data are skipped when it is absent.

## Prepare the simulator data

The recordings are archived at <https://doi.org/10.5281/zenodo.22802397>. Download every part of the
datasets you need (for an integration recording, all files `INT_<dataset-id>_p*.zip`), check them
against `ZENODO_SHA256SUMS.txt`, and extract them into one directory, `$DATA` below:

```bash
for z in INT_ho1-mtn-fig8-vary-v1_p*.zip; do unzip -q -o "$z" -d "$DATA"; done
```

The record's `README.md` gives the same steps for PowerShell. The committed run configs read each dataset from `datasets/<dataset-id>/`, so link the ingested
dataset there:

```bash
mkdir -p datasets
ln -s "$DATA/INT/ho1-mtn-fig8-vary-v1/ingested_dataset" datasets/ho1-mtn-fig8-vary-v1
```

On Windows, create a directory junction with `New-Item -ItemType Junction` instead. `docs/data.md`
lists the paths the skyline study configs expect and what the archive does not contain.

## Run visual odometry over a dataset

```bash
java -cp "$CP" org.boofcv.evaluation.VoRunnerApp \
    --config evaluation/eval_configs/int/exp-int-003/run-ho1-mtn-fig8-vary-v1-vo-only.json
```

The run record goes to `runs/<run_id>/`: `manifest.json`, `frames.csv` and the sidecar files the
config enables (`docs/data.md`, Run record). `--run-id <id>` writes it under another name.
`--dataset-dir <dir>` replaces the config's `dataset_dir`, but the height, heading and profile files a
config names keep their own paths. `--motion-timing <file.csv>` also records per-frame
motion-estimation time.

The `run-*-vo-only.json` configs in `evaluation/eval_configs/int/` run the front end with the metric
and heading readouts described under Integrated navigation, and with relocalization off.
`alignment_sidecar` makes them write the local-only position in the same form as the integrated arm,
which is what that arm is compared against and what the replay below reads.

For another dataset in the same format, a config needs only `dataset_dir` and, preferably, `run_id`;
the pose is then published in reference-frame pixels. The keys that change the estimator:

| Key | Effect |
|---|---|
| `motion_model` | `homography` (default), `affine` or `similarity` |
| `navigation_source` | how the pose is read from the transforms; the default `rigid_motion` integrates translation and rotation only |
| `downsampleFactor` | integer box downsampling of each image before tracking (1 = none) |
| `metric_readout`, `heading_readout` | metric scale and an external heading (below) |
| `diagnostic_refit_sidecar` | writes the estimator-health signal to `diagnostic_refit.csv`; navigation is unchanged |

All keys and their defaults are in `src/main/java/org/boofcv/evaluation/VoRunnerConfig.java`. Paths in
a run config are resolved against the working directory, so run the driver from the repository root.

A run is deterministic on one machine: repeating it reproduces the run record except for the
per-frame processing time and the run timestamp. Across processors or JVM builds, tracked feature
positions can differ in the last bits and change an occasional RANSAC decision, so trajectories agree
closely but not bit for bit.

## Skyline profiles

Each ingested INT dataset contains `skyline_profiles.csv`, the North and West profiles its reported
runs used, built from the simulator's ground-truth sky masks. No step is needed to use them.

To regenerate them from `raw_skyline_capture/`:

1. Ingest both views as observation sessions. The configs in
   `evaluation/eval_configs/int/skyline-sessions/` (one per recording and view, named by simulator
   run; `docs/reproducing.md` maps runs to dataset ids) give the capture's development location as
   `run_dir`. Use a copy with `run_dir` set to `$DATA/INT/<dataset-id>/raw_skyline_capture` and
   `store_root` set to `observations_sim_int` in the repository root. Relative paths in the copy are
   resolved against the copy's own directory, so absolute paths are simplest. From `skyline/`:

   ```bash
   python -m hsreloc.simret.cli ingest --config <copy of Run_20260908_130747-north.json>
   python -m hsreloc.simret.cli ingest --config <copy of Run_20260908_130747-west.json>
   ```

2. Export the profiles, paired with the nadir frames through the simulator's capture identity
   (`vo_frame_id`):

   ```bash
   python evaluation/tools/int/export_skyline_profiles.py \
       --session-dir observations_sim_int/Run_20260908_130747-north \
       --west-session-dir observations_sim_int/Run_20260908_130747-west \
       --source sim_exact \
       --vo-frames datasets/ho1-mtn-fig8-vary-v1/frames.csv \
       --sim-observations "$DATA/INT/ho1-mtn-fig8-vary-v1/raw_skyline_capture/skyline/observations.csv" \
       --out evaluations/profiles/ho1-mtn-fig8-vary-v1
   ```

   Write to a directory of your own, not into the linked dataset. With `--source sim_exact` this
   reproduces the archived file byte for byte. `--help` lists the other profile sources.

## Integrated navigation

```bash
java -cp "$CP" org.boofcv.evaluation.VoRunnerApp \
    --config evaluation/eval_configs/int/exp-int-003/run-ho1-mtn-fig8-vary-v1-int-c0.json
```

This config is the local-only config plus the relocalization layer. Its blocks:

- `metric_readout` converts each frame's translation increment from pixels to metres with an
  external height: the takeoff-relative height stream (`height_csv`, here the `baro_relative_m`
  column of `terrain.csv`), the declared initial height above ground `h0_agl_m` and the focal length
  `fx_native_px`. A missing value is an error, not a default. The height sample used for each frame
  and its freshness are written to `metric_track.csv`.
- `heading_readout` rotates the increments by an external heading. `sim_nadir_camera_heading` reads
  the nadir camera's image-up azimuth (`heading_deg` in `groundtruth.csv`); `fc_ahrs_body_compass`
  reads an airframe heading and requires the camera mounting offset `delta_mount_deg`.
  `max_rate_deg_per_s` rejects implausible samples. The visually integrated yaw is logged, never used.
- `skyline_profiles` is the dataset's profile file, and `relocalization_config` the acceptance
  policy. The thesis policy, `evaluation/eval_configs/int/exp-int-002/reloc-c0-primary-retry20.json`,
  compares profiles by zero-shift NCC (`c0_frozen_ncc`) and scores a reference by the weaker of its
  North and West matches. A correction needs a score of at least 0.90, a margin of 0.30 over the best
  reference outside a 50 m ambiguity radius, agreement of the two views within 15 m, and a reference
  not stored in the last 30 s; failed searches are retried after 20 frames at the earliest.
  `src/main/java/org/boofcv/relocalization/RelocalizationConfig.java` documents every key; unknown
  keys are refused.

The layer requires both readouts. It adds `alignment_frames.csv` (persistent position and its
validity per frame), `alignment_events.csv` (search requests, retrievals, gate decisions,
insertions, re-anchors) and `relocalization_manifest.json`; `frames.csv` is identical to the
local-only run's.

### Injected tracking loss

A `synthetic_hard_loss` block makes the estimator take its own hard-loss branch on the listed frames:

```json
"synthetic_hard_loss": {"acknowledge_synthetic": true, "frames": [450, 1000], "note": "..."}
```

This is deterministic fault injection for exercising recovery, not an observed estimator failure.
Such losses are labelled `loss_source = synthetic` in `alignment_events.csv` and counted separately
in the manifest. The reported pair is `exp-int-001/run-mtn-r2-vary-v1-vo-only-synthetic-loss.json`
and `exp-int-002/run-mtn-r2-vary-v1-int-c0-retry20-synthetic-loss.json`.

## Replay relocalization over a recorded track

```bash
java -cp "$CP" org.boofcv.evaluation.RelocalizationReplayApp \
    --vo-run runs/exp-int-003-ho1-mtn-fig8-vary-v1-vo-only \
    --relocalization-config evaluation/eval_configs/int/exp-int-002/reloc-c0-primary-retry20.json \
    --skyline-profiles datasets/ho1-mtn-fig8-vary-v1/skyline_profiles.csv \
    --out runs/replay-ho1-mtn-fig8-c0
```

This re-drives the relocalization layer over a local-only run without repeating the image
processing, which takes about a second instead of minutes, so policies can be compared cheaply. The
input run must have been made with both readouts and `alignment_sidecar`, as the `run-*-vo-only.json`
runs are. The output holds the same three files as a live integrated run and is byte-identical to
them (asserted by `RelocalizationReplayAppTest`); its manifest says `mode = REPLAY`. A policy that
sets `instability_threshold_deg` is refused, because that signal is not in the recorded track.

## Evaluate a run

Local-only against integrated, on one dataset:

```bash
python evaluation/tools/int/evaluate_int_arms.py --dataset datasets/ho1-mtn-fig8-vary-v1 \
    --vo-only runs/exp-int-003-ho1-mtn-fig8-vary-v1-vo-only \
    --int int_c0=runs/exp-int-003-ho1-mtn-fig8-vary-v1-int-c0 \
    --out evaluations/ho1-mtn-fig8-vary-v1
```

Both arms are registered to ground truth by the frame-0 translation only: rotation and scale come from
the external channels and are not fitted. `metrics.json` holds each arm's ATE RMSE over the frames
where both arms have a valid position, the final error, and every accepted correction classified
against the ground-truth distance to its reference (genuine revisit, wrong place, harmful).
`reanchors_<arm>.csv`, `retrievals_<arm>.csv` and `error_curve.csv` hold the detail. `--int` can be
given several times, and a replay directory can stand in for a live run.

Any run record against its dataset, with a fitted Sim(2) alignment (for runs in pixel units), from
`evaluation/`:

```bash
python -m naveval.evaluate --config <evaluation.json> --out ../evaluations/<name>
```

The config names `evaluation_id`, `run_record` and `dataset`, with paths relative to the config file;
`configs/eval-golden.json` is an example. The evaluator writes `metrics.json` (ATE, RPE, drift,
heading, endpoint and failure statistics), `alignment.json`, `events.csv`, `segments.csv`,
`scale_series.csv` and `figures/`.

## Skyline studies

The skyline studies run from `skyline/` and read the SKY batches from the paths listed in
`docs/data.md`. Studies that compare extracted profiles also need SegFormer label maps, produced by
`scripts/silver_infer.py` in a separate environment (`skyline/README.md`).

| Study | Entry point and stages | Inputs |
|---|---|---|
| Recognition under nine appearance conditions | `PYTHONPATH=. python scripts/sim_ext_inventory.py --root ../simulator_skyline_data_extended --out ../evaluations/sim-ext-inventory`, then `python -m hsreloc.simret.cli <command> --config configs/sim-final.json` | `simulator_skyline_data_extended/` |
| One view against two | `python scripts/sky_dual_study.py --config configs/sky-dual.json --stage audit\|ingest\|score\|report` | `simulator_skyline_data_both_directions/`, the appearance study's index |
| Two views on the hard city run | `python scripts/sky_dual_hardcity.py --config configs/sky-dual-hardcity.json --stage audit\|ingest\|score\|report` | the hard-city run, the two-view study's outputs |
| Matcher alignment freedom and positional resolution | `python scripts/sky_matcher_resolution.py --config configs/sky-matcher-resolution.json --stage audit\|ingest\|score\|score-terrain\|score-hardcity\|report` | the two figure-eight captures, the two studies above |
| Recognition radius | `python scripts/sky_recog_radius.py --config configs/sky-recog.json --stage all` | the ingested appearance batch and its label maps |

`scripts/sky_relpose_study.py`, `sky_dual_report.py` and `sky_matcher_report.py` are helper modules
imported by these scripts. The frame lists the SegFormer configs (`configs/silver-infer-*.json`) read
are written by the drivers' `ingest` stages and, for the appearance study, by `silver-list`. The appearance study's `ext-run` and `ext-exp-*` commands take
`--population dev` (the village) or `--population final` (mountains and city); `final` refuses to
run unless the inputs it builds match the pre-registration `configs/frozen/sim-final.prereg`, and no
guarded command overwrites a complete record. The order of the commands, the pre-registration and
the population of the two-view study are covered in `docs/reproducing.md`.

## Reproducing the thesis results

For the mapping from thesis results to exact datasets, configurations and evaluation scripts, see
[`docs/reproducing.md`](reproducing.md).
