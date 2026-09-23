# Skyline-Aided UAV Navigation

Implementation accompanying the master's thesis *Stitching Based UAV Optical Navigation With
Skyline-based Localization* (Adi Peisach, Ariel University, 2026).

A downward-looking camera gives a UAV a continuous estimate of its own motion, but that estimate is
relative and drifts. This repository implements a navigation framework that combines a
stitching-based visual odometry front end with occasional position corrections obtained by
recognising previously visited places from the shape of the horizon, accepted only when the
evidence clearly identifies one place. Everything here runs offline, on simulator recordings from
Unreal Engine 5 and on the MARS-LVIG recorded-flight dataset. None of it has been run onboard an
aircraft.

## System overview

**Local visual motion estimation** (Java, every frame). Corners are tracked against a keyframe and a
planar transform is fitted robustly with BoofCV; the motion model is selectable (homography, affine,
or a direct similarity fit). Each transform is linearised at the image centre and only its
translation and rotation are integrated. Scale and the non-rigid part are recorded as diagnostics
and never applied. Translation is converted to metres with an external takeoff-relative height, a
declared focal length and a declared initial height above ground, and it is rotated by an external
heading. Neither quantity is estimated from the imagery or blended with it. A hard tracking loss
starts a new segment, and the displacement across the gap is reported as unknown rather than zero.

**Estimator-health observer** (Java, optional). Each accepted transform is refitted over all of its
inliers, and the rotation disagreement between the two fits is reported per frame. The observer
leaves navigation bit-identical.

**Skyline place evidence** (Python, per query). North- and West-facing horizon views are segmented
into sky and non-sky, using the simulator's own masks, SegFormer-B0 or a classical
dynamic-programming extractor. Each view is reduced to a 256-sample, mean-removed profile of the
sky-boundary height. Profiles reach the Java side only as a CSV file; no Java code opens a horizon
image.

**Conservative relocalization** (Java). The layer keeps a local position (metres within the current
segment) and a persistent position that differs from it by one translation. While the persistent
position is trusted, captures are stored as references. A search is requested after enough
accumulated motion, after a set time, or immediately after a hard loss. A query is compared with
every reference in both views by zero-shift normalised cross-correlation, and the weaker of the two
scores counts. The persistent position is moved onto a reference only if the candidate passes the
score threshold, is not a recently stored reference, beats the best-scoring reference outside its
region by a margin, and is named by both views. Anything short of that leaves the position
unchanged.

```
UE5 recording ─ingest─> dataset: frames, ground truth, height, heading ──────┐
      │                                                                       v
      └─ingest views─> sessions ─extract─> skyline_profiles.csv ──> VoRunnerApp
                                                                  (VO + metric/heading readout
                                                                   + relocalization layer)
                                                                        │
                                                        run record: frames.csv, metric_track.csv,
                                                        alignment_*.csv, manifests
                                                                        │
                                                   naveval + evaluation/tools ─> metrics, tables
```

## Repository layout

| Path | Contents |
|---|---|
| `src/main/java/org/boofcv/stitching/` | VO front end: estimator, motion models, rigid readout; `metric/` height and heading channels; `diagnostics/` Monte Carlo and probe drivers |
| `src/main/java/org/boofcv/relocalization/` | Integration layer: reference memory, search scheduler, matcher, region rules, acceptance gate, alignment |
| `src/main/java/org/boofcv/evaluation/` | Run drivers `VoRunnerApp` and `RelocalizationReplayApp`, run-record writer |
| `src/main/java/org/boofcv/confidence/` | Optional confidence scorer wired into the run driver; its calibrations are unvalidated and no reported result uses it |
| `src/test/` | JUnit 5 tests |
| `evaluation/` | Python evaluator `naveval`, run and evaluation configs, per-experiment tools ([README](evaluation/README.md)) |
| `skyline/` | Python skyline bench `hsreloc`: simulator ingest, extraction, matchers, studies ([README](skyline/README.md)) |
| `docs/` | [Data layout and formats](docs/data.md), [workflows](docs/workflows.md), [results-to-code map](docs/reproducing.md) |

## Requirements

- **JDK 19.** Gradle 8.2 is provided by the wrapper. The build declares no toolchain, so it uses the
  JDK that `JAVA_HOME` points to (the JDK root, not its `bin` directory). BoofCV 0.44, Jackson 2.17,
  Lombok 1.18 and JUnit 5.10 are fetched by Gradle.
- **Python 3.12** with `evaluation/requirements.txt` and `skyline/requirements.txt` (numpy,
  matplotlib, opencv-python, pytest). Last tested with numpy 2.5.2, matplotlib 3.11.1,
  opencv-python 5.0 and pytest 9.1.1.
- Optional: `zstandard` / `lz4` to ingest compressed MCAP files, and a separate environment with
  `torch` (CPU), `transformers` and `pillow` for SegFormer inference (see `skyline/README.md`).

No GPU is required.

## Build and test

```bash
./gradlew build            # compiles and runs the Java tests (gradlew.bat on Windows)
./gradlew installDist      # runnable jars in build/install/skyline-aided-uav-navigation/lib/

python -m venv .venv
.venv/bin/python -m pip install -r evaluation/requirements.txt -r skyline/requirements.txt

cd evaluation && ../.venv/bin/python -m pytest -q && cd ..
cd skyline && ../.venv/bin/python -m pytest tests -q && cd ..
```

On Windows the interpreter is `.venv\Scripts\python.exe`. Tests that need recorded data skip when it
is absent. `JavaProducedRunFixtureTest` rewrites `evaluation/tests/fixtures/java_produced_run/` with
the local JVM version and timings on every run; that change is expected and should not be
committed.

## Usage

[docs/workflows.md](docs/workflows.md) is the practical guide: building and testing, preparing the
published data, running the visual odometry and the integrated system, regenerating skyline
profiles, replaying relocalization over a recorded track, evaluating runs, and running the skyline
studies. The Java driver reads a JSON config and writes a run record to `runs/<run_id>/`; run it
from the repository root, for example:

```bash
java -cp "build/install/skyline-aided-uav-navigation/lib/*" org.boofcv.evaluation.VoRunnerApp \
    --config evaluation/eval_configs/int/exp-int-003/run-ho1-mtn-fig8-vary-v1-int-c0.json
```

## Data

The simulator recordings and processed datasets used in the thesis are archived on Zenodo:
<https://doi.org/10.5281/zenodo.22802397>. The archive holds the twelve integration recordings, each
as an ingested dataset (nadir images, ground truth, height channel, skyline profiles) together with
its raw horizon captures, and two skyline batches: North and West views over three environments,
and 67 places captured under nine appearance conditions. The recordings were produced with the
[Optical-Navigation-UE5-Simulator](https://github.com/AdiPassover/Optical-Navigation-UE5-Simulator).

[docs/data.md](docs/data.md) describes the archive, where the code expects each part, the file
formats, and what is not included. The visual-odometry and estimator-health results use the public
MARS-LVIG dataset, which is fetched from its MCAP mirror and not redistributed here.

## Reproducing the evaluation

There is no single reproduction command. [docs/reproducing.md](docs/reproducing.md) maps each section
of the thesis results chapter to the recordings, configs and scripts that produced it. For the
integrated result, starting from the archive:

1. Link each `INT/<dataset-id>/ingested_dataset/` as `datasets/<dataset-id>/`.
2. Run the local-only (`run-*-vo-only.json`) and integrated (`run-*-int-c0.json`) configs in
   `evaluation/eval_configs/int/exp-int-00{1,2,3}/` with `VoRunnerApp`.
3. Score a pair with `evaluation/tools/int/evaluate_int_arms.py`, or replay the three correction
   policies over the thirteen local-only runs with
   `evaluation/tools/claim_closure/int_comparison_arms.py`.

Stages that consume imagery need the recordings. The analysis stages need only the CSV and JSON
outputs of the earlier stages.

## Assumptions and limitations

- The front end assumes a near-nadir camera over locally planar ground. Vertical motion is not
  estimated.
- Metric scale is only as good as the external height *above the imaged surface*. A takeoff-relative
  height does not observe terrain relief, so over relief the metric track carries a scale error, and
  a translation-only correction does not remove it.
- Heading must come from an external channel with a declared datum. Before the first heading sample
  there is no metric increment, and the visually integrated rotation is never substituted.
- Relocalization only recognises places the flight has already stored. References are not stored
  while the position is unknown or once the drift budget since the last anchor is spent, so a flight
  that loses tracking early may never recover a position.
- A correction snaps to a stored reference, so its precision is bounded by the reference spacing.
  When the local track is already more accurate than that spacing, a correct recognition can
  increase the error.
- The skyline matcher assumes each view's absolute direction is known to about a degree.
- The skyline and integration results come from rendered environments, and the visual-odometry and
  health results from two MARS-LVIG scenes flown with one platform. Timings in the thesis come from
  one development laptop and are not onboard measurements.
- Some `run` and `ext-*` commands are guarded by pre-registration files in `skyline/configs/frozen/`
  and refuse to overwrite existing results.
- Identifiers such as `EXP-INT-003` or `DEC-VO-010` in configs and comments refer to records of the
  development log, which is not part of this repository.

## Citation

If you use this software, please cite it using the metadata in [`CITATION.cff`](CITATION.cff), or:

```bibtex
@software{peisach2026skyline,
  author  = {Peisach, Adi},
  title   = {Skyline-Aided {UAV} Navigation},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/AdiPassover/skyline-aided-uav-navigation}
}
```

The simulator dataset is a separate publication with its own DOI,
<https://doi.org/10.5281/zenodo.22802397>.

## License

The code is released under the MIT License ([`LICENSE`](LICENSE)). The Gradle wrapper files
(`gradlew`, `gradlew.bat`, `gradle/wrapper/`) are distributed under the Apache License 2.0, and
`StitchingFactory` reproduces the construction sequence of BoofCV's `FactoryMotion2D` (Apache
License 2.0). The simulator dataset is licensed separately, under CC BY 4.0.
