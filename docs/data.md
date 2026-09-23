# Data layout and file formats

No recorded data is stored in this repository. Two sources are used:

- **Simulator recordings** from the Unreal Engine 5 environment
  ([Optical-Navigation-UE5-Simulator](https://github.com/AdiPassover/Optical-Navigation-UE5-Simulator)),
  archived on Zenodo: A. Peisach, *Simulator Dataset for Stitching-Based UAV Optical Navigation With
  Skyline-Based Localization*, 2026, <https://doi.org/10.5281/zenodo.22802397> (CC BY 4.0). They
  back the skyline and integrated-navigation results.
- **MARS-LVIG** (Li et al., 2024), a public recorded-flight dataset, for the visual-odometry and
  estimator-health results. It is not redistributed; [`docs/reproducing.md`](reproducing.md) lists
  the windows and the scripts that fetch and ingest them.

## The Zenodo archive

The record holds 44 ZIP files that make up 14 logical datasets; `MANIFEST.csv` maps each file to its
dataset, and `ZENODO_SHA256SUMS.txt` gives a checksum per file (the record's `README.md` shows how to
verify them). Every ZIP is a complete archive, not a fragment. Extracting all parts of a dataset into
one directory, keeping their internal paths, reconstructs that dataset; extracting all 44 into one
directory gives

```
INT/<dataset-id>/ingested_dataset/            integration recordings, ingested (12 datasets)
INT/<dataset-id>/raw_skyline_capture/         their horizon views, as exported by the simulator
SKY/both_directions/<level>/Run_<stamp>/      two-view skyline batch (10 runs)
SKY/extended/<level>/Run_<stamp>/             nine-condition appearance batch (44 runs)
```

Datasets can be downloaded separately: an INT dataset is 2.1–14.0 GB (all files
`INT_<dataset-id>_p*.zip`), and the two SKY batches are 1.6 and 1.1 GB. The whole record is about
50 GB.

### INT: integration recordings

Twelve flights, each recorded with the nadir camera and the North- and West-facing horizon cameras
together. The dataset ids, the simulator run each one comes from and its role in the thesis are
listed in `docs/reproducing.md` (Integrated navigation).

`ingested_dataset/` is what the Java run driver and the Python evaluators read. It has the format
described under [Dataset](#dataset) below: `dataset.json`, `frames.csv`, `groundtruth.csv` (60 Hz),
`terrain.csv` (the height channel), `ue_body.csv`, `ingest_provenance.json`, the nadir images
(`images/frame_NNNNNN.png`, 1024 × 1024, 10 Hz) and the skyline profiles `skyline_profiles.csv` with
their manifest `skyline_profiles.json`. These are the profiles the reported runs used.

`raw_skyline_capture/` is the simulator's export of the horizon views: `settings.json` and
`skyline/` (`observations.csv`, the 512 × 512 North and West images, and the ground-truth sky masks).
It is needed only to regenerate `skyline_profiles.csv` and for the skyline study that uses the two
`fig8-flat-*` recordings.

The simulator's raw nadir export (`vo/`) is not in the archive. Its images are byte-identical to
`ingested_dataset/images/`, and `ingest_provenance.json` records the SHA-256 of every raw file the
ingest read. The nadir ingest (`evaluation/tools/int/ingest_ue_run_int.py`) therefore cannot be
repeated from the archive; the ingested dataset is the archived form.

### SKY: skyline batches

`SKY/both_directions/` has ten runs with a North- and a West-facing camera at every capture:
five in `asaian_village_hills_background`, three in `large_flat_city` (one of them,
`Run_20260907_125137`, is the hard-city run) and two in `mountains`. Each run holds `settings.json`,
`skyline/` as above, and `vo/frames.csv` and `vo/groundtruth.csv`, the 60 Hz pose trace the two-view
study uses to confirm segment boundaries.

`SKY/extended/` has 44 North-view runs: 14 in `asian_village_hills_background`, 15 in
`large_city_flat` and 15 in `mountains`. They capture 67 anchor places under each of nine conditions
(time of day `DAWN`, `DAY`, `DUSK` × clouds `CLEAR`, `CLOUDY`, `VERY_CLOUDY`), declared per run in
`settings.json` under `environment`, plus star-shaped swipes around selected anchors.

The level directory names are the simulator's and differ between the two batches;
`asaian_village_hills_background` is misspelled in the recordings themselves. The configs refer to
these names, so keep them. Image size, field of view and intrinsics are per run: read them from each
run's `settings.json`.

## Where the code expects the data

The committed configs name their inputs by fixed relative paths: Java run configs relative to the
working directory (run from the repository root), Python study configs relative to the config file,
which also resolves to the repository root. To use the configs unchanged, expose the extracted
archive under these names by a symbolic link, a directory junction or a copy:

| Path the committed configs use | Archive source | Read by |
|---|---|---|
| `datasets/<dataset-id>/` | `INT/<dataset-id>/ingested_dataset/` | run configs in `evaluation/eval_configs/int/`, `evaluation/tools/` |
| `simulator_skyline_data_extended/` | `SKY/extended/` | `skyline/configs/sim-final.json` |
| `simulator_skyline_data_both_directions/<level>/` | `SKY/both_directions/<level>/` | `sky-dual.json`, `sky-dual-hardcity.json`, `sky_compactness.py` |
| `simulator_skyline_data_both_directions/figure_8_flat_surface/Run_20260907_140138/` | `INT/fig8-flat-const-v1/raw_skyline_capture/` | `sky-matcher-resolution.json` |
| `simulator_skyline_data_both_directions/figure_8_flat_surface/Run_20260907_140549/` | `INT/fig8-flat-vary-v1/raw_skyline_capture/` | `sky-matcher-resolution.json` |

The two `simulator_skyline_data_*` names come from the development workspace. They stay because the
study configs use them, and the pre-registration `skyline/configs/frozen/sim-final.prereg` records the
root name `simulator_skyline_data_extended` together with a digest over every file under it; the
extracted `SKY/extended/` reproduces that digest.

With the archive extracted to `$DATA`, from the repository root (POSIX shell):

```bash
mkdir -p datasets simulator_skyline_data_both_directions/figure_8_flat_surface
for d in "$DATA"/INT/*/; do ln -s "${d}ingested_dataset" "datasets/$(basename "$d")"; done
ln -s "$DATA/SKY/extended" simulator_skyline_data_extended
for l in "$DATA"/SKY/both_directions/*/; do
    ln -s "$l" "simulator_skyline_data_both_directions/$(basename "$l")"; done
ln -s "$DATA/INT/fig8-flat-const-v1/raw_skyline_capture" \
    simulator_skyline_data_both_directions/figure_8_flat_surface/Run_20260907_140138
ln -s "$DATA/INT/fig8-flat-vary-v1/raw_skyline_capture" \
    simulator_skyline_data_both_directions/figure_8_flat_surface/Run_20260907_140549
```

Linked as a whole, `large_flat_city/` also exposes the hard-city run `Run_20260907_125137`, which the
two-view study must not see before its `ingest` stage; `docs/reproducing.md` (Skyline place
evidence) gives the order.

On Windows, directory junctions need no administrator rights (PowerShell):

```powershell
New-Item -ItemType Junction -Path datasets\ho1-mtn-fig8-vary-v1 `
    -Target "$DATA\INT\ho1-mtn-fig8-vary-v1\ingested_dataset"
```

All these paths are listed in `.gitignore`.

Where a command takes a path argument, use it instead of editing a committed config.
`VoRunnerApp --dataset-dir <dir>` replaces the config's `dataset_dir`, the directory the frames and
ground truth are read from. The other files a run config names (`metric_readout.height_csv`,
`heading_readout.heading_csv`, `skyline_profiles`) keep their configured paths, which in the committed
configs are under `datasets/<dataset-id>/`, so those configs need the dataset at that path. The Python
tools take their inputs as arguments (`--dataset`, `--vo-frames`, `--sim-observations`, …). The skyline
configs have no path arguments: `raw_root` and `run_dir` come from the config file. Point them
elsewhere only in a local copy, as `docs/workflows.md` (Skyline profiles) does for the INT session
configs, whose `run_dir` is the development location of each capture, and not for `sim-final.json`,
whose pre-registration records the root directory name.

## Not in the archive

- The raw nadir export of the INT recordings (see above).
- The constant/varying-height figure-eight pair of the game-engine height experiment
  (`uevo-fig8-const-v1` and `uevo-fig8-vary-v1`, simulator runs `Run_20260904_172033` and
  `Run_20260904_174825`), read by the configs in `evaluation/eval_configs/exp-vo-014/` and by the
  `run-uevo-*` configs in `vo-metric-runtime/` and `vo-heading-runtime/`.
- `interesting-r2-vary-v1`, a development recording with configs in
  `evaluation/eval_configs/int/exp-int-001/` and no reported result, and the pilot skyline batch read
  by `skyline/configs/sim-pilot.json`.
- MARS-LVIG and the UAVScenes calibration files (`docs/reproducing.md`), and the SegFormer weights
  (`skyline/README.md`).
- Everything the pipeline generates: datasets ingested from MARS-LVIG or rendered by
  `evaluation/tools/exp_vo_005/synth_planar.py`, `exp_vo_011/render_distorted.py` and
  `exp_vo_012/synth_terrain.py`; run records, observation sessions, label maps, reference sets and
  evaluation outputs (below).
- The third-party environment assets used for rendering; the archive holds rendered observations
  only.

Generated data goes to these directories at the repository root:

```
datasets/<dataset_id>/          ingested datasets (VO input, ground truth, skyline profiles)
runs/<run_id>/                  run records written by VoRunnerApp / RelocalizationReplayApp
observations_sim*/<session>/    ingested skyline observation sessions
silver_masks/<root>/            SegFormer label maps written by skyline/scripts/silver_infer.py
evaluations/<name>/             evaluator and study outputs
skyline_refdb/, skyline_runs/   reference sets and retrieval records of the appearance study
```

## Simulator recording

One directory per flight, as exported by the simulator. In the archive, an INT recording's
`raw_skyline_capture/` holds `settings.json` and `skyline/`; a two-view SKY run also has `vo/frames.csv`
and `vo/groundtruth.csv`; an appearance-batch run has the North view only. `observations.csv` may sit
at the run root or under `skyline/`; both are accepted.

```
Run_<stamp>/
├── settings.json            world frame and UE->ENU mapping, camera intrinsics and field of view,
│                            capture policy, height datums, environment (time of day, clouds)
├── vo/
│   ├── frames.csv           one row per nadir image (10 Hz)
│   ├── groundtruth.csv      one row per simulation tick (60 Hz), same pose/height columns, no image
│   └── images/frame_NNNNNN.png
└── skyline/
    ├── observations.csv     one row per horizon capture (North and West views)
    ├── images/sky_NNNNNN.png, sky_NNNNNN_west.png
    └── sim/sky_NNNNNN_sky.png, sky_NNNNNN_west_sky.png    binary ground-truth sky masks, white = sky
```

`vo/frames.csv` columns: `frame_id, sim_time_s, image_path`, the nadir camera pose in UE units
(`cam_ue_{x,y,z}_cm`, `cam_ue_q{w,x,y,z}`), the airframe pose (`body_ue_*`), the ENU camera position
`east_m, north_m, up_m`, `heading_deg` (image-up azimuth, clockwise from North), `camera_tilt_deg`,
and the height channels `baro_relative_alt_m` (takeoff-relative, ideal), `true_agl_m`,
`terrain_elevation_m`, `ground_hit`.

`skyline/observations.csv` columns: `observation_id, sim_time_s`, the North, nadir and West camera
poses (`{north,nadir,west}_ue_{x,y,z}_cm`, `_yaw_deg`, `_pitch_deg`, `_roll_deg`, `_quat_{w,x,y,z}`),
`image_path, sim_sky_mask_path, west_image_path, west_sim_sky_mask_path`, and `vo_frame_id,
vo_synchronized`, which identify the nadir frame a capture was taken on. Recordings made before the
synchronised capture was added carry no `vo_frame_id`; their profiles are paired to VO frames by
nearest timestamp and labelled as approximate. Positions are UE world coordinates in centimetres;
conversion to metres and East/North happens in `skyline/hsreloc/simret/conventions.py`.

The simulator's nadir camera is world-stabilised, so `heading_deg` is constant over a flight while
the airframe yaw follows the path. The ingest uses the camera heading and quarantines the airframe
pose in `ue_body.csv`.

## Dataset

Written by `evaluation/tools/int/ingest_ue_run_int.py` (simulator) or
`evaluation/tools/exp_vo_007/ingest_sequence.py` (MARS-LVIG). Frames are ENU; headings are degrees
clockwise from North.

```
datasets/<dataset_id>/
├── dataset.json            source type, evidence tier, clocks, conventions, position/heading/height
│                           quality classes, camera intrinsics, metadata
├── frames.csv              frame_index, timestamp_s, image_path
├── groundtruth.csv         timestamp_s, east_m, north_m, up_m, heading_deg, fix_quality, valid
├── terrain.csv             simulator: per-frame terrain_m, agl_m, up_m, baro_relative_m (the height channel)
├── ue_body.csv             simulator: airframe pose, never used as the camera pose
├── attitude.csv            MARS-LVIG: fused attitude sidecar (heading channel and health reference)
├── skyline_profiles.csv    North and West profiles keyed by this dataset's frame_index
├── skyline_profiles.json   export manifest (source, pairing mechanism, exactness, refusals)
├── ingest_provenance.json  SHA-256 of the source files
└── images/
```

A ground-truth row with `valid=false` is kept and counted, never dropped. Empty cells mean "not
available", never zero.

### `skyline_profiles.csv`

Written by `evaluation/tools/int/export_skyline_profiles.py`, read by
`org.boofcv.relocalization.SkylineProfileFile`. One row per horizon capture:

```
frame_index,timestamp_s,sync_dt_s,observation_id,valid,invalid_reason,profile,
west_observation_id,west_available,west_valid,west_invalid_reason,west_profile
```

`profile` and `west_profile` are 256 semicolon-separated samples: the sky-boundary elevation per
column as a fraction of frame height, resampled to 256 and mean-removed, empty when the view is not
valid. A capture whose profile cannot be formed (a column with no sky above it, or a degenerate
profile) is written with `valid=false` and a reason, never filled in. The West view may be
`unavailable`; it is never fabricated. With the exact `vo_frame_id` identity `sync_dt_s` is 0 and the
Java loader refuses any residual.

## Run record

Written by `org.boofcv.evaluation.VoRunnerApp`; read by `naveval`.

```
runs/<run_id>/
├── manifest.json                 run and dataset ids, estimator configuration, environment (JVM, OS,
│                                 CPU), frame and processed counts, completion flag
├── frames.csv                    one row per processed frame
├── metric_track.csv              with metric_readout: pose in metres and in pixels, the height sample
│                                 and heading used for each increment and their freshness, segment index
├── logical_transform.csv         with logical_transform_sidecar: per-frame transforms for offline analysis
├── alignment_frames.csv          with the relocalization layer: persistent position, validity flags,
│                                 segment / epoch / anchor, pending search request
├── alignment_events.csv          hard losses, search requests, retrievals, gate decisions, insertions,
│                                 re-anchors with the translation applied
├── relocalization_manifest.json  policy, matcher, counts of natural/synthetic losses and recenters
├── refinement.csv                with refinement_sidecar
└── diagnostic_refit.csv          with diagnostic_refit_sidecar (estimator-health observer)
```

`frames.csv` columns (located by header name): `frame_index, timestamp_s, est_x, est_y, est_z,
est_yaw_deg, success, event, reference_id, h00…h22, track_count, inlier_count, process_time_ns`,
optionally followed by the confidence columns. `est_x/est_y` are in the unit the configured
navigation source publishes (reference-frame pixels by default, metres for `metric_local`);
`est_yaw_deg` is clockwise with the start frame as datum; `event` is one of `none`, `init`,
`recenter`, `restart`. A recenter is a mosaic re-origin inside one tracking segment; only `restart`
(a hard loss) starts a new segment.

`frames.csv` is identical with the relocalization layer on or off; the layer writes only its own
sidecars.
