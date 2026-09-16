"""The frozen parity reference must keep matching the Python code it was generated from.

`MetricReadoutPythonParityTest` (Java) checks the runtime against
`evaluation/tools/vo_metric_runtime/reference/`. That is only evidence of anything while the
reference is still what `exp_vo_012/metric_readout.py` and `baro.py` produce. If either of those
changes and the reference is not regenerated, the Java test would go on passing against a stale
answer and the agreement would quietly stop meaning what it claims.

So this suite re-derives every committed `expected.csv` from the committed `samples.csv`, the
committed run record and the current Python implementation, and requires **exact** equality. A
failure here means one of two things, and the message says which: the Python readout changed (in
which case regenerate the reference and re-run the Java test), or the committed run records changed
(in which case something much larger is wrong).

    cd evaluation && $PY -m pytest tools/vo_metric_runtime -q
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
EVALUATION_DIR = HERE.parents[1]
REPO = EVALUATION_DIR.parent
REFERENCE = HERE / "reference"
sys.path.insert(0, str(EVALUATION_DIR))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EXP012 = EVALUATION_DIR / "tools" / "exp_vo_012"
baro = _load("vo_metric_runtime_test_baro", EXP012 / "baro.py")
mr = _load("vo_metric_runtime_test_readout", EXP012 / "metric_readout.py")


def cases():
    path = REFERENCE / "cases.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


CASES = cases()
IDS = [c["name"] for c in CASES]


@pytest.mark.skipif(not CASES, reason="parity reference absent from this checkout")
def test_the_case_set_is_the_one_the_java_test_expects():
    """The Java side asserts at least five cases; keep the two halves from drifting apart."""
    assert len(CASES) >= 5
    names = {c["name"] for c in CASES}
    assert {"uevo-vary-baro", "uevo-vary-fixed"} <= names
    # At least one case must exercise each of the three things the reference exists to pin.
    assert any(c["n_invalid_frames"] > 0 for c in CASES), "no case crosses tau_stale"
    assert any(c["n_stale_frames"] > 0 for c in CASES), "no case exercises a held sample"
    assert any(c["n_restarts"] > 0 for c in CASES), "no case contains a stitching restart"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_expected_csv_still_equals_what_the_python_readout_produces(case):
    run_dir = REPO / "runs" / case["run_id"]
    case_dir = REFERENCE / case["name"]
    if not (run_dir / "logical_transform.csv").exists():
        pytest.skip(f"committed run record absent: {run_dir}")

    expected = read_csv(case_dir / "expected.csv")
    samples = read_csv(case_dir / "samples.csv")
    t_frames = np.array([float(r["timestamp_s"]) for r in expected])
    t_avail = np.array([float(r["t_avail_s"]) for r in samples])
    values = np.array([float(r["relative_m"]) for r in samples])

    inc = mr.load_increments(run_dir, case["frame_width"], case["frame_height"])
    assert len(inc) == len(expected), "the committed run record has changed length"

    spec = baro.BaroSpec(rate_hz=1.0 / case["nominal_sample_interval_s"],
                         tau_stale_s=case["tau_stale_s"])
    sampled = baro._resample(t_frames, t_avail, values, spec)
    h_agl = mr.h_agl_from_baro(case["h0_agl_m"], sampled.h_baro)
    track = mr.integrate_metric(inc, h_agl, case["f_working_px"], valid=sampled.valid)

    why = ("the frozen reference no longer matches the Python readout it was generated from. "
           "Regenerate it with make_parity_reference.py AND re-run the Java parity test -- do not "
           "regenerate it alone, or the agreement stops being evidence of anything.")

    for k, row in enumerate(expected):
        assert float(row["east_m"]) == track.east_m[k], f"{case['name']} frame {k} east: {why}"
        assert float(row["north_m"]) == track.north_m[k], f"{case['name']} frame {k} north: {why}"
        assert float(row["yaw_deg"]) == track.yaw_deg[k], f"{case['name']} frame {k} yaw: {why}"
        assert float(row["h_frame_m"]) == track.h_used_m[k], f"{case['name']} frame {k} h: {why}"
        assert (row["h_valid"] == "1") == bool(track.valid[k]), f"{case['name']} frame {k} valid"
        assert (row["h_stale"] == "1") == bool(sampled.stale[k]), f"{case['name']} frame {k} stale"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_the_sample_stream_is_causal_and_ordered(case):
    """A stream Java replays sample by sample must be monotone; an out-of-order row would be
    rejected by `CausalHeightChannel.submit`, and the reference must never contain one."""
    samples = read_csv((REFERENCE / case["name"]) / "samples.csv")
    t = np.array([float(r["t_avail_s"]) for r in samples])
    assert np.all(np.diff(t) >= 0.0)
    assert len(samples) == case["n_samples"]


@pytest.mark.skipif(not CASES, reason="parity reference absent from this checkout")
def test_the_fixed_arm_and_the_baro_arm_differ_materially():
    """The control on the whole exercise: if a fixed first-frame height gave the same answer as a
    causally sampled one, none of this would be worth implementing. `EXP-VO-014` measured the gap;
    this asserts the frozen reference still shows it, so a Java run that reproduces the fixed arm
    could never be mistaken for one that reproduces the barometer arm."""
    by_name = {c["name"]: c for c in CASES}
    fixed = by_name["uevo-vary-fixed"]
    baro_case = by_name["uevo-vary-baro"]
    d = np.hypot(fixed["endpoint_east_m"] - baro_case["endpoint_east_m"],
                 fixed["endpoint_north_m"] - baro_case["endpoint_north_m"])
    assert d > 100.0, f"the two arms should be hundreds of metres apart on this run, got {d:.1f} m"
