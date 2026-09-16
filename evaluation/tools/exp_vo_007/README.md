# `EXP-VO-007` tooling — independent-sequence validation of `RIGID_MOTION`

Acquisition, ingest, integrity and analysis for `EXP-VO-007`, which tests whether the ~1 %
normalised ATE `EXP-VO-006` measured on `HKairport01` generalises to a flight the readout was not
designed on (decision `DEC-VO-005`).

Run everything from the repository root:

```
.venv/Scripts/python.exe evaluation/tools/exp_vo_007/<script>.py ...
```

## The scripts, in the order they are used

| script | what it does | validated against |
|---|---|---|
| `probe_candidates.py` (+ `height_profile.py`, the shared loader for the series it emits) | Measures candidate MARS-LVIG sequences **before** one is chosen: cruise window and path from the RTK track, and height above the *imaged surface* plus in-footprint depth spread from the dataset's own LiDAR. ~250 MB, no imagery, no VO. | reproduces the committed `hkairport01-b` cruise window to the millisecond (`--control`, non-zero exit if it does not) |
| `derive_yaw_offset.py` | Re-derives the `rtk_yaw` → compass offset empirically for a sequence. `LIT-006` states the offset is sequence-specific and must never be inherited; getting it wrong is a ~90° heading error that looks entirely plausible. | on `HKairport01` recovers **269.44 ± 0.73°** against the committed **269.16 ± 0.64°**, inside the field's own 1° quantisation |
| `mcap_source.py` | I/O policy for the MCAP mirror — `naveval` deliberately has none. Disk-backed range cache, 512 MiB chunked fetches with independent retries (`EXP-002`'s fetch-mechanics fix, as code rather than shell history), and sub-window splitting. | — |
| `ingest_sequence.py` | Builds a `contracts/dataset.md` dataset from any MARS-LVIG MCAP window, with the attitude sidecar decoded from the same bytes in the same pass. | rebuilds a slice of the committed `hkairport01-b` and compares |
| `plot_groundtruth.py` | The ten ingest gates and the ground-truth-only figure, produced **before** the estimator sees a frame. Exits non-zero if any gate fails. | — |
| `analyse.py` | Scores the pre-registered hypotheses H1–H6 against legacy and full-logical arms computed on the same sequence. | parallel to `exp_vo_006/rigid_readout.py`, same metric code paths |
| `plot_exp_vo_007.py` | The three result figures, each stamped with the run ids it was built from. | — |

## Two constraints that shaped the design

**Memory.** `naveval.ingest_mars_lvig.load_hkairport01_mcap_window` materialises the whole
overlapping chunk span as one `bytes` object. `AMtown01`'s 1,189 s cruise window spans ~29 GB, and
this machine has 15.7 GB of RAM. So the window is split into sub-windows, the loader is called once
per piece **unmodified**, and frames are streamed to disk rather than accumulated — a lazy handle
with the same `.timestamp_s` / `.data` interface is passed to the builder in place of the payload.

**Bandwidth.** ~2–3 MB/s measured, and it does not improve with parallel connections. A full-cruise
fetch is therefore a multi-hour job, so every sub-window is checkpointed: imagery is already durable
as it is written, and the RTK/attitude samples are persisted alongside it, so a re-run resumes
instead of restarting. Sub-windows are fetched in chronological order, which means a partial
acquisition always yields a complete *prefix* of the frozen window rather than a hole in the middle.

## What is deliberately not here

No estimator code, no `naveval` behaviour. The one `naveval` change this work needed —
parameterising the MARS-LVIG adapter's seven sequence-specific fields, whose defaults remain the
`HKairport01` values — lives in `naveval.ingest_mars_lvig` itself.
