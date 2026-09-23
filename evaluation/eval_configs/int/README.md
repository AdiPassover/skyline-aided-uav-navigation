# INT relocalization policy configs

`RelocalizationConfig` files (`VoRunnerApp` key `relocalization_config`) and the `EXP-INT-001`
run configs. Every value is a **configurable selection**, recorded so a run's
`relocalization_manifest.json` can be traced to it (Principle VI) — none is a universal production
constant.

## `exp-int-001/` — the milestone experiment (2026-09-08, `DEC-INT-007`)

| file | what it is |
|---|---|
| `reloc-c0-primary.json` | **The frozen C0 policy** of `EXP-INT-001`: C0 pointwise NCC (no alignment search), weakest-view fusion, temporal confirmation fallback-only, `position_radius` with the **three explicit radii** and a **C0-calibrated** gate — all selected on the five DEV recordings by the pre-registered rule in `EXP-INT-001` §Calibration, before any evaluation-recording number was seen. |
| `reloc-c1-4-fallback.json` | The pre-registered **fallback / comparison** arm: C1 at `max_lag_samples = 4`, its own DEV-calibrated gate, otherwise identical. Run on the evaluation recording only under the experiment's fallback rule. |
| `run-<dataset>-vo-only.json` | The VO_ONLY arm: metric readout (`h0_agl_m` from the recording's `settings.json`, `fx` 512), authoritative nadir-camera heading, `alignment_sidecar` on, relocalization **off** — the persistent position is the VO's own metric track. |
| `run-<dataset>-int-c0.json` / `run-<dataset>-int-c1-4.json` | The INT arms: byte-identical VO settings plus `relocalization_config` and the dataset's exact-identity `skyline_profiles.csv`. |

Datasets: `fig8-flat-{const,vary}-v1`, `mtn-r{1,2,3}-*-v1` (DEV), `interesting-r1-vary-v1`
(evaluation), `interesting-r3-const-v1` (safety control), `interesting-r2-vary-v1` (hard-loss
candidate, not in the Zenodo archive) — roles frozen in `EXP-INT-001` §Dataset. The archived
datasets are read from `datasets/<id>/` (`docs/data.md`).

Run a config as a plain JVM process (after `./gradlew installDist`):

```
java -cp "build/install/skyline-aided-uav-navigation/lib/*" org.boofcv.evaluation.VoRunnerApp --config evaluation/eval_configs/int/exp-int-001/run-<dataset>-<arm>.json
```

Replay a policy over a recorded VO_ONLY track without re-running the VO (calibration only; a
replay's manifest says `mode = REPLAY` and is never a milestone arm):

```
java -cp "build/install/skyline-aided-uav-navigation/lib/*" org.boofcv.evaluation.RelocalizationReplayApp --vo-run runs/<vo-only run> --relocalization-config <policy.json> --skyline-profiles datasets/<id>/skyline_profiles.csv --out <dir>
```

Evaluate both arms against ground truth (`evaluation/tools/int/evaluate_int_arms.py`, definitions
in `EXP-INT-001` §Metrics).

## `exp-int-002/` — the "easier" recording, the cadence decision, the development proof-of-benefit (2026-09-08, `DEC-INT-008`)

| file | what it is |
|---|---|
| `run-easier-sq-const-v1-{vo-only,int-c0,int-c1-4}.json` | The **untouched first look** on `easier-sq-const-v1` (`hopefully_easier/Run_20260907_231953`): the `EXP-INT-001` policies referenced **unchanged** from `../exp-int-001/`. Result: 0 accepts in both INT arms, correctly (far-field skyline; `EXP-INT-002` R-B). |
| `reloc-c0-primary-retry20.json` | **The `DEC-INT-008` policy**: `reloc-c0-primary.json` with `min_retry_gap_frames` 30 → 20 (= the 15 m / 7.5 m/s / 10 Hz capture interval) and **nothing else changed**. Pre-registered primary arm for the next untouched recording. |
| `run-mtn-r2-vary-v1-int-c0-retry20.json`, `…-retry20-synthetic-loss.json` | The live arms that produced the **development proof-of-benefit** on `mtn-r2` (ATE −18.8 %, final 17.2 → 4.1 m, two genuine corrections) and the two hard-loss recoveries; VO_ONLY arms are the committed `../exp-int-001/run-mtn-r2-vary-v1-vo-only*.json` runs. `mtn-r2` is a calibration recording — development label. |
| `dev/reloc-c0-retry{20,0}.json` | The one-variable replay variants (development entries DEV-002 / DEV-003); `retry0` reverted (identical to 20). |

Analysis tools for these runs (`evaluation/tools/int/`): `gt_revisit_events.py`,
`vo_quality_report.py`, `snap_longitudinal.py`, `revisit_opportunity_table.py`,
`inject_synthetic_drift.py` + `synthetic_drift_sweep.py`, `plot_int_arms.py`.

## `exp-int-003/` — the `held_out1` held-out validation of the frozen method (2026-09-08)

**The frozen method is `../exp-int-002/reloc-c0-primary-retry20.json`** (SHA-256 `b5457092…`); this
directory adds no policy for the primary arm. Everything here was written **before any INT arm was run on `held_out1`** (14:27:49 local) and
committed in the pre-registration commit `1ed75806` (14:38:02, after INT arms 1–3 had run —
`EXP-INT-003` R-10); no parameter was chosen or changed using these recordings.

| file | what it is |
|---|---|
| `run-ho1-{mtn-fig8,mtn-fig8-scaled,vil-fig8,mtn-trinity}-vary-v1-vo-only.json` | The VO_ONLY baselines on the four held-out recordings (simulator runs `held_out1/Run_20260908_{130747,131337,133522,141325}`): field-for-field the `EXP-INT-001` VO_ONLY config, only dataset paths, `h0_agl_m` (from each recording's `settings.json`) and ids differ. |
| `run-ho1-…-int-c0.json` | **The PRIMARY held-out arms**: the VO_ONLY config plus the frozen `reloc-c0-primary-retry20.json` and the dataset's exact-identity North+West profiles. Result (`EXP-INT-003`): **STRONG on three of four** — ATE −38.1 % / −34.2 % / −46.0 %, final 100.9 → 4.4 m, 127.5 → 1.9 m, 47.7 → 6.2 m, 7 genuine accepts, 0 wrong-place / harmful; the village run refused everything, correctly. |
| `run-ho1-…-int-c1-4.json` | The **pre-declared comparison** arm: `../exp-int-001/reloc-c1-4-fallback.json` verbatim (retry gap 30). Not the milestone arm. |
| `reloc-c1-4-fallback-retry20.json`, `run-ho1-…-int-c1-4-retry20.json` | The pre-declared comparison arm with the accepted `DEC-INT-008` gap applied to the fallback (one key changed, created in the pre-registration). On held-out data C1-4 accepts references 14–31 m away (its lag freedom raises far scores, `EXP-SKY-013`), once severely harmfully on the village run — the held-out confirmation of `DEC-INT-007`. |

## `skyline-sessions/` — SKY-adapter ingest configs for the INT recordings

One config per recording × view (`north` / `west`), for the unmodified SKY adapter
(`hsreloc.simret.cli ingest`, run from `skyline/`), writing to
`observations_sim_int/` with `image_mode: reference` (no imagery copied). Same conventions and
options as the `EXP-SKY-013` figure-eight sessions, which the two flat figure-eight
datasets reuse from `observations_sim_fig8/`. Profiles are then exported per dataset with
`export_skyline_profiles.py --sim-observations … --vo-frames …` (exact `vo_frame_id` identity) into
`datasets/<id>/skyline_profiles.{csv,json}`. Their `run_dir` is the capture's development location;
the archive holds each capture as `INT/<id>/raw_skyline_capture/`, and `docs/workflows.md` (Skyline
profiles) shows how to ingest it from there.

## Historical — the 2026-09-07 starting policy (superseded as a starting default by `DEC-INT-007`)

| file | what it was | evidence |
|---|---|---|
| `reloc-starting-dual-weakest.json` | The starting experimental INT skyline policy of `DEC-INT-003` (amended 2026-09-07): both views per query and per reference, `weakest_view` fusion, region-level competing margin on the SKY τ = 125 m ball (legacy single `region_radius_m`), gate 0.95 / 0.20, temporal confirmation `fallback`, **matcher C1-32**. | `EXP-SKY-011` + `EXP-SKY-012`: 0 observed dual false accepts in 4 007 tested hard negatives. |
| `reloc-alternative-dual-strict.json` | The equally false-free, more conservative comparison: `strict_agreement`, otherwise identical. | `EXP-SKY-012` R3. |

Both are kept **verbatim** so the 2026-09-07 plumbing runs stay reproducible; neither is a
starting default any more. `EXP-SKY-013` showed that C1-32's lag freedom raises the score of
*distant* references (the 70–186 m snaps of those runs) and that its 0.95 / 0.20 gate does not
carry over to another matcher; `DEC-INT-007` makes C0 the primary matcher with its own calibrated
gate. Their single legacy `region_radius_m = 125` was SKY's recognition-*evaluation* τ-ball and has
no runtime standing (`DEC-INT-005`); the run manifest labels it `region_radii.source = LEGACY`.
Since 2026-09-08 a config declares either that legacy key or all three of
`ambiguity_region_radius_m` / `dual_agreement_radius_m` / `temporal_region_radius_m`, never a
mixture.

The North-only single-view baseline is the code default of `RelocalizationConfig` (`north_only`,
temporal `required`), not a file here, because `EXP-SKY-010`/`012` measured it unsafe on the flat
city; use it only as a comparison arm.

## Ground-truth revisit metadata

`gt-revisits-2026-09-08.json` — revisit/intersection structure of the six 2026-09-07 UE5 recordings
(`eight_figure_mountains_less_steep` ×3, `interesting_path` ×3), derived from **simulator ground
truth alone**: no skyline score, no matcher, no retrieval defines any of it. Per event: both passes'
frame/time/XY, physical separation, elapsed time, AGL and heading at each pass, the nearest
synchronised skyline observation to each, and the VO's own accumulated position error at each pass
from a metric+heading run with relocalization off. It is a description of the data, not a result.

`skyline_sync_tolerance_s: 0.0` in every policy here — the configs expect the **exact** capture
identity of the synchronised recordings (`vo_frame_id`); an approximately paired legacy export
needs an explicit tolerance and is labelled as such in the manifest.
