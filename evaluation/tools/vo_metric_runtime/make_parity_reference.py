"""Freeze the Python metric readout's answer on committed inputs, so Java can be checked against it.

The runtime port of `DEC-VO-007` D3 into Java has to reproduce the already-validated offline
implementation, not merely "look right". This script produces the frozen side of that comparison:
for each case it writes

    reference/<case>/samples.csv     the height stream, as a SENSOR STREAM (availability time, value)
    reference/<case>/expected.csv    what `metric_readout.integrate_metric` produces, frame by frame

and `reference/cases.json`, which names the run, the dataset, `h0`, `f_working` and `tau_stale` for
each case. `MetricReadoutPythonParityTest` (Java) reads exactly these files, drives the production
`RigidNavigationState` + `MetricNavigationState` + `CausalHeightChannel` over them, and requires
agreement. Nothing is recomputed on either side: the increments come from the committed
`logical_transform.csv`, and the heights come from the committed dataset.

**The height stream is materialised, not described.** `samples.csv` is what a live sensor would have
delivered and when; Java replays it through `CausalHeightChannel.submit` in that order. Both
implementations therefore see one identical input, which is what makes a disagreement diagnosable as
a convention difference rather than an input difference.

**Nothing here is fitted, and no case makes a physical claim.** Case `hka-restart-synth` uses a
*constructed* height ramp on a real flight that has no barometer (`EXP-VO-013`): it exists to
exercise the restart and varying-height code paths numerically, and its numbers are not a
measurement of anything.

    PY=.venv/Scripts/python.exe
    $PY evaluation/tools/vo_metric_runtime/make_parity_reference.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVALUATION_DIR = HERE.parents[1]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP012 = EVALUATION_DIR / "tools" / "exp_vo_012"
baro = _load("vo_metric_runtime_baro", EXP012 / "baro.py")
mr = _load("vo_metric_runtime_metric_readout", EXP012 / "metric_readout.py")

REFERENCE = HERE / "reference"

#: 17 significant digits round-trips an IEEE-754 double exactly, so the frozen reference loses
#: nothing and a Java/Python disagreement can never be an artefact of the file format.
FMT = "%.17g"


def fmt(x: float) -> str:
    return FMT % x


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


# --------------------------------------------------------------------------- the height streams


def stream_every_frame(t_frames: np.ndarray, relative: np.ndarray):
    """One sample per frame, at the frame's own time. The `EXP-VO-014` arms' channel."""
    return t_frames.copy(), np.asarray(relative, float).copy()


def stream_at_rate(t_frames: np.ndarray, relative: np.ndarray, rate_hz: float,
                   latency_s: float = 0.0, dropouts: tuple = ()):
    """A slower channel: samples on their own grid, optional latency and dropout windows.

    Sample times are anchored at the first frame exactly as `baro.py::_sample_times` does, so the
    reference frame always carries a genuine sample and the relative datum is exact by construction.
    """
    t0, t1 = float(t_frames[0]), float(t_frames[-1])
    n = int(np.floor((t1 - t0) * rate_hz)) + 1
    ts = t0 + np.arange(n) / rate_hz
    values = np.interp(ts, t_frames, relative)
    alive = np.ones(ts.shape, bool)
    for start, dur in dropouts:
        alive &= ~((ts >= t0 + start) & (ts < t0 + start + dur))
    if not alive.any():
        raise ValueError("every sample dropped")
    return ts[alive] + latency_s, values[alive]


# --------------------------------------------------------------------------- one case


def build_case(name: str, run_id: str, width: int, height: int, f_working: float, h0: float,
               tau_stale_s: float, nominal_interval_s: float,
               t_frames: np.ndarray, t_avail: np.ndarray, values: np.ndarray,
               note: str) -> dict:
    run_dir = REPO / "runs" / run_id
    inc = mr.load_increments(run_dir, width, height)
    if len(inc) != len(t_frames):
        raise SystemExit(f"{name}: {len(inc)} increments but {len(t_frames)} frame times")

    # The frozen resampler, driven by the materialised stream. Using baro.py's own `_resample`
    # rather than a local copy is the point: the reference is produced by the code the experiments
    # ran, not by a re-implementation of it.
    spec = baro.BaroSpec(rate_hz=1.0 / nominal_interval_s, tau_stale_s=tau_stale_s)
    sampled = baro._resample(t_frames, t_avail, values, spec)
    if not sampled.causal:
        raise SystemExit(f"{name}: the reference resampler reported a non-causal policy")

    h_agl = mr.h_agl_from_baro(h0, sampled.h_baro)
    track = mr.integrate_metric(inc, h_agl, f_working, arm=name, valid=sampled.valid)

    # The height and ground sampling distance the increment INTO frame k is converted with are the
    # REFERENCE frame's -- index k-1. Emitted explicitly so the Java comparison does not have to
    # re-derive the offset and get it wrong (LIT-VO-003 section 10.1).
    n = len(inc)
    h_used_for_increment = np.concatenate([[np.nan], track.h_used_m[:-1]])
    gsd_used_for_increment = np.concatenate([[np.nan], track.gsd_m_per_px[:-1]])

    case_dir = REFERENCE / name
    case_dir.mkdir(parents=True, exist_ok=True)

    with (case_dir / "samples.csv").open("w", newline="") as f:
        f.write("t_avail_s,relative_m\n")
        for t, v in zip(t_avail, values):
            f.write(f"{fmt(t)},{fmt(v)}\n")

    with (case_dir / "expected.csv").open("w", newline="") as f:
        f.write("frame_index,event,timestamp_s,h_frame_m,h_valid,h_stale,"
                "h_used_for_increment_m,gsd_used_m_per_px,east_m,north_m,yaw_deg\n")
        for k in range(n):
            f.write(",".join([
                str(int(inc.frame_index[k])), inc.events[k], fmt(t_frames[k]),
                fmt(track.h_used_m[k]),
                "1" if bool(track.valid[k]) else "0",
                "1" if bool(sampled.stale[k]) else "0",
                "" if not np.isfinite(h_used_for_increment[k]) else fmt(h_used_for_increment[k]),
                "" if not np.isfinite(gsd_used_for_increment[k]) else fmt(gsd_used_for_increment[k]),
                fmt(track.east_m[k]), fmt(track.north_m[k]), fmt(track.yaw_deg[k]),
            ]) + "\n")

    meta = {
        "name": name,
        "run_id": run_id,
        "frame_width": width,
        "frame_height": height,
        "f_working_px": f_working,
        "h0_agl_m": h0,
        "tau_stale_s": tau_stale_s,
        "nominal_sample_interval_s": nominal_interval_s,
        "n_frames": n,
        "n_samples": int(len(t_avail)),
        "n_stale_frames": int(sampled.stale.sum()),
        "n_invalid_frames": int((~sampled.valid).sum()),
        "n_restarts": sum(1 for e in inc.events if e == "restart"),
        "endpoint_east_m": float(track.east_m[-1]),
        "endpoint_north_m": float(track.north_m[-1]),
        "note": note,
    }
    print(f"  {name}: {n} frames, {len(t_avail)} samples, "
          f"{meta['n_stale_frames']} stale, {meta['n_invalid_frames']} invalid, "
          f"{meta['n_restarts']} restarts, end=({track.east_m[-1]:.3f}, {track.north_m[-1]:.3f}) m")
    return meta


# --------------------------------------------------------------------------- the cases


def ue_inputs(dataset_id: str):
    ds = REPO / "datasets" / dataset_id
    meta = json.loads((ds / "dataset.json").read_text())
    intr = meta["metadata"]["camera_intrinsics"]
    terrain = read_csv(ds / "terrain.csv")
    return (meta, intr, col(terrain, "timestamp_s"), col(terrain, "baro_relative_m"))


def main() -> int:
    REFERENCE.mkdir(parents=True, exist_ok=True)
    cases = []

    # --- EXP-VO-014's varying-height figure-8: the headline geometry, x3.15 of altitude change ---
    meta, intr, t, rel = ue_inputs("uevo-fig8-vary-v1")
    h0 = float(meta["metadata"]["h0_agl_m"])
    f_working = mr.f_working(intr["fx"], 1)
    w, hgt = int(intr["width"]), int(intr["height"])
    run = "uevo-fig8-vary-homography-rigid-v1"

    ta, va = stream_every_frame(t, rel)
    cases.append(build_case(
        "uevo-vary-baro", run, w, hgt, f_working, h0, 2.0, 0.1, t, ta, va,
        "EXP-VO-014 arm B (BARO) on the varying-height figure-8: one ideal sample per frame."))

    ta, va = stream_every_frame(t, np.zeros_like(rel))
    cases.append(build_case(
        "uevo-vary-fixed", run, w, hgt, f_working, h0, 2.0, 0.1, t, ta, va,
        "EXP-VO-014 arm A (FIXED): the same run with the height channel reporting nothing, so "
        "h_AGL == h0 throughout. The null the barometer arm is measured against."))

    ta, va = stream_at_rate(t, rel, rate_hz=1.0, latency_s=0.0)
    cases.append(build_case(
        "uevo-vary-zoh-1hz", run, w, hgt, f_working, h0, 2.0, 1.0, t, ta, va,
        "A 1 Hz channel against 10 Hz imagery: nine of every ten frames are a zero-order hold. "
        "Exercises the FRESH/HELD boundary."))

    ta, va = stream_at_rate(t, rel, rate_hz=1.0, latency_s=0.0)
    cases.append(build_case(
        "uevo-vary-held", run, w, hgt, f_working, h0, 2.0, 0.1, t, ta, va,
        "The same 1 Hz stream, but with the nominal interval DECLARED as the imagery's 0.1 s, so "
        "nine of every ten frames report HELD rather than FRESH. Included because the "
        "FRESH/HELD boundary is a reporting choice and must move the flags WITHOUT moving the "
        "trajectory: this case's east/north are identical to uevo-vary-zoh-1hz's."))

    ta, va = stream_at_rate(t, rel, rate_hz=1.0, dropouts=((60.0, 8.0), (200.0, 15.0)))
    cases.append(build_case(
        "uevo-vary-dropout", run, w, hgt, f_working, h0, 2.0, 1.0, t, ta, va,
        "The same 1 Hz channel with 8 s and 15 s dropouts: HELD, then STALE beyond tau_stale = 2 s, "
        "then recovery. The height is HELD across the gap and the frames are FLAGGED, never "
        "silently substituted (DEC-VO-007 D6)."))

    # --- a real flight WITH a stitching restart, on a CONSTRUCTED height series -------------------
    # HKairport01 has no barometer (EXP-VO-013), so this height is invented. It is here to exercise
    # the restart path and a varying height numerically; it measures nothing.
    ds = REPO / "datasets" / "hkairport01-a"
    dmeta = json.loads((ds / "dataset.json").read_text())
    dintr = dmeta["metadata"]["camera_intrinsics"]
    frames = read_csv(ds / "frames.csv")
    t_hka = col(frames, "timestamp_s")
    t_hka = t_hka - t_hka[0]
    # A deterministic, clearly artificial ramp-and-dip. No fitting, no ground truth.
    rel_hka = 12.0 * np.sin(t_hka / 37.0) + 0.05 * t_hka
    ta, va = stream_at_rate(t_hka, rel_hka, rate_hz=2.0, latency_s=0.0)
    cases.append(build_case(
        "hka-restart-synth", "hkairport01-a-homography-rigid-v1",
        int(dintr["width"]) // 2, int(dintr["height"]) // 2,
        mr.f_working(dintr["fx"], 2), 80.0, 2.0, 0.5, t_hka, ta, va,
        "Real HKairport01 increments (1 stitching restart, 13 recenters) with a CONSTRUCTED height "
        "series -- MARS-LVIG has no barometer (EXP-VO-013). Numerical parity fixture only; it is "
        "not a measurement and no physical claim rests on it."))

    (REFERENCE / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"\nwrote {len(cases)} cases to {REFERENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
