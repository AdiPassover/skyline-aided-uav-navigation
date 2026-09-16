"""The exporter's pairing and row semantics (``DEC-INT-002`` / ``DEC-INT-003`` boundary), on fake
sources so no skyline image or curve store is needed. The Java loader's parity with this format is
``SkylineProfileFileTest``. Run from ``evaluation/``: ``python -m pytest tools/int -q``."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import export_skyline_profiles as ex  # noqa: E402


def obs(observation_id: str, frame_index: int, timestamp_s: float, raw: str | None) -> SimpleNamespace:
    extra = {} if raw is None else {ex.PAIR_KEY: raw}
    return SimpleNamespace(observation_id=observation_id, frame_index=frame_index, timestamp_s=timestamp_s,
                           extra=extra)


class FakeSource:
    """``get(id)`` returns an array that the patched ``normalize`` passes through."""

    def __init__(self, curves: dict):
        self.curves = curves

    def get(self, observation_id):
        if observation_id not in self.curves:
            raise ex.CurveError(f"no curve for {observation_id}")
        return self.curves[observation_id]

    def describe(self):
        return {"kind": "fake"}


@pytest.fixture(autouse=True)
def passthrough_profile(monkeypatch):
    monkeypatch.setattr(ex, "normalize", lambda curve, config: np.asarray(curve, dtype=float))
    monkeypatch.setattr(ex, "is_degenerate", lambda p, floor: float(np.std(p)) < floor)


def config():
    return ex.ProfileConfig(n_samples=4, normalize_mean=True, detrend=False)


def test_nearest_time_pairing_records_the_residual_and_rejects_beyond_tolerance():
    frames = (np.arange(6), np.arange(6) * 0.1)
    assert ex.pair_by_time(0.2833, *frames) == (3, pytest.approx(0.2833 - 0.3))
    north = [obs("a", 0, 0.2833, "sky_1"), obs("b", 1, 0.58, "sky_2")]
    src = FakeSource({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})
    rows, stats = ex.build_rows(north, ex.pair_west(north, None), src, None, config(), 1e-6, None, 0, frames, 0.05)
    assert [r[0] for r in rows] == [3, 5]
    assert rows[0][4] == "true" and float(rows[0][2]) == pytest.approx(-0.0167, abs=1e-4)
    assert rows[1][4] == "false" and "sync residual" in rows[1][5], "0.58 is 80 ms from frame 5: beyond 50 ms → invalid"
    assert float(rows[1][2]) == pytest.approx(0.08)
    assert stats["n_sync_rejected"] == 1 and stats["n_valid"] == 1
    assert rows[0][7:] == ["", "false", "", "", ""], "no West session: unavailable, not fabricated"


def test_two_observations_on_one_vo_frame_are_refused():
    frames = (np.arange(3), np.arange(3) * 0.1)
    north = [obs("a", 0, 0.10, None), obs("b", 1, 0.11, None)]
    src = FakeSource({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})
    with pytest.raises(ex.ExportError, match="two observations"):
        ex.build_rows(north, ex.pair_west(north, None), src, None, config(), 1e-6, None, 0, frames, 0.05)


def test_west_is_paired_by_the_capture_key_and_never_copied_or_invented():
    north = [obs("n1", 0, 1.0, "sky_1"), obs("n2", 1, 2.0, "sky_2"), obs("n3", 2, 3.0, "sky_3")]
    west = [obs("w1", 0, 1.0, "sky_1"), obs("w3", 2, 3.0, "sky_3")]           # no West for sky_2
    west_of = ex.pair_west(north, west)
    assert west_of["n1"].observation_id == "w1" and west_of["n2"] is None and west_of["n3"].observation_id == "w3"
    ns = FakeSource({"n1": [0, 1, 0, 1], "n2": [1, 0, 1, 0], "n3": [0, 0, 1, 1]})
    ws = FakeSource({"w1": [2, 3, 5, 7], "w3": [1, 1, 1, 1]})                  # w3 is flat → degenerate
    rows, stats = ex.build_rows(north, west_of, ns, ws, config(), 1e-3, None, 0, None, 0.05)
    assert rows[0][7:10] == ["w1", "true", "true"] and rows[0][11] == "2.0;3.0;5.0;7.0"
    assert rows[0][6] == "0.0;1.0;0.0;1.0", "North's own profile, and West's is not it"
    assert rows[1][7:] == ["", "false", "", "", ""]
    assert rows[2][7:10] == ["w3", "true", "false"] and "degenerate" in rows[2][10] and rows[2][11] == ""
    assert stats == {**stats, "n_west_available": 2, "n_west_valid": 1}


def test_a_west_without_a_north_partner_or_a_mismatched_time_is_refused():
    north = [obs("n1", 0, 1.0, "sky_1")]
    with pytest.raises(ex.ExportError, match="no North partner"):
        ex.pair_west(north, [obs("w1", 0, 1.0, "sky_1"), obs("w9", 1, 9.0, "sky_9")])
    with pytest.raises(ex.ExportError, match="capture time"):
        ex.pair_west(north, [obs("w1", 0, 1.5, "sky_1")])
    with pytest.raises(ex.ExportError, match="share the capture key"):
        ex.pair_west(north, [obs("w1", 0, 1.0, "sky_1"), obs("w1b", 3, 1.0, "sky_1")])


def test_explicit_frame_map_and_offset_pair_by_index(tmp_path):
    north = [obs("n1", 4, 1.0, None), obs("n2", 5, 2.0, None)]
    src = FakeSource({"n1": [0, 1, 0, 1], "n2": [1, 0, 1, 0]})
    rows, _ = ex.build_rows(north, ex.pair_west(north, None), src, None, config(), 1e-6, None, 10, None, 0.05)
    assert [r[0] for r in rows] == [14, 15] and rows[0][2] == ""
    fm = tmp_path / "map.csv"
    fm.write_text("observation_id,frame_index\nn1,100\nn2,7\n")
    rows, _ = ex.build_rows(north, ex.pair_west(north, None), src, None, config(), 1e-6, ex.load_frame_map(fm),
                            0, None, 0.05)
    assert [r[0] for r in rows] == [7, 100], "sorted by VO frame"


def sim_rows(**kw):
    """raw id -> (sim_time_s, vo_frame_id, vo_synchronized)."""
    return {k: {"sim_time_s": v[0], "vo_frame_id": v[1], "vo_synchronized": v[2]} for k, v in kw.items()}


def test_exact_identity_pairs_by_vo_frame_id_and_never_by_proximity():
    frames = (np.arange(6), np.arange(6) * 0.1)
    # sky_2 was captured at t=0.3 but the simulator consumed it on frame 4 (t=0.4): the identity
    # contradicts the clock -> refused, even though frame 3 is "nearest".
    north = [obs("a", 0, 0.1, "sky_1"), obs("b", 1, 0.3, "sky_2")]
    src = FakeSource({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0]})
    good = sim_rows(sky_1=(0.1, 1, True), sky_2=(0.3, 3, True))
    rows, stats = ex.build_rows(north, ex.pair_west(north, None), src, None, config(), 1e-6, None, 0, frames, 0.05,
                                sim_obs=good)
    assert [r[0] for r in rows] == [1, 3] and [float(r[2]) for r in rows] == [0.0, 0.0]
    assert stats["unsynchronized"] == {} and stats["n_sync_rejected"] == 0
    bad = sim_rows(sky_1=(0.1, 1, True), sky_2=(0.3, 4, True))
    with pytest.raises(ex.ExportError, match="contradicts the clock"):
        ex.build_rows(north, ex.pair_west(north, None), src, None, config(), 1e-6, None, 0, frames, 0.05, sim_obs=bad)
    # a frame the VO dataset does not have: not one recording
    with pytest.raises(ex.ExportError, match="not a frame of the VO dataset"):
        ex.build_rows(north, ex.pair_west(north, None), src, None, config(), 1e-6, None, 0, frames, 0.05,
                      sim_obs=sim_rows(sky_1=(0.1, 1, True), sky_2=(0.3, 42, True)))
    # a session row whose capture time is not the simulator row's: a different recording
    with pytest.raises(ex.ExportError, match="not the same recording"):
        ex.build_rows(north, ex.pair_west(north, None), src, None, config(), 1e-6, None, 0, frames, 0.05,
                      sim_obs=sim_rows(sky_1=(0.1, 1, True), sky_2=(0.35, 3, True)))
    # exact identity without the VO dataset to verify against is refused
    with pytest.raises(ex.ExportError, match="needs the VO dataset"):
        ex.build_rows(north, ex.pair_west(north, None), src, None, config(), 1e-6, None, 0, None, 0.05, sim_obs=good)


def test_an_unsynchronised_capture_has_no_row_and_is_recorded_not_invented():
    frames = (np.arange(6), np.arange(6) * 0.1)
    north = [obs("a", 0, 0.1, "sky_1"), obs("b", 1, 0.3, "sky_2"), obs("c", 2, 0.5, "sky_3")]
    west = [obs("wa", 0, 0.1, "sky_1"), obs("wb", 1, 0.3, "sky_2"), obs("wc", 2, 0.5, "sky_3")]
    ns = FakeSource({"a": [0, 1, 0, 1], "b": [1, 0, 1, 0], "c": [0, 0, 1, 1]})
    ws = FakeSource({"wa": [2, 3, 5, 7], "wb": [1, 2, 4, 8], "wc": [3, 1, 4, 1]})
    sim = sim_rows(sky_1=(0.1, 1, True), sky_2=(0.3, -1, False), sky_3=(0.5, 5, True))
    rows, stats = ex.build_rows(north, ex.pair_west(north, west), ns, ws, config(), 1e-6, None, 0, frames, 0.05,
                                sim_obs=sim)
    assert [r[0] for r in rows] == [1, 5], "sky_2 had no synchronised VO frame: no row, no nearest-neighbour guess"
    assert list(stats["unsynchronized"]) == ["b"] and "vo_frame_id=-1" in stats["unsynchronized"]["b"]
    assert rows[0][7:10] == ["wa", "true", "true"] and rows[1][7:10] == ["wc", "true", "true"], \
        "North and West of one simulator row share the vo_frame_id"
    assert stats["n_valid"] == 2 and stats["n_west_valid"] == 2


def test_the_simulator_observations_file_must_carry_the_synchronisation_columns(tmp_path):
    p = tmp_path / "observations.csv"
    p.write_text("observation_id,sim_time_s,image_path\nsky_1,0.1,x.png\n")
    with pytest.raises(ex.ExportError, match="not a synchronised simulator export"):
        ex.load_sim_observations(p)
    p.write_text("observation_id,sim_time_s,vo_frame_id,vo_synchronized\nsky_1,0.1,27,1\nsky_2,0.3,-1,0\n")
    rows = ex.load_sim_observations(p)
    assert rows["sky_1"] == {"sim_time_s": 0.1, "vo_frame_id": 27, "vo_synchronized": True}
    assert rows["sky_2"]["vo_synchronized"] is False and rows["sky_2"]["vo_frame_id"] == -1
    p.write_text("observation_id,sim_time_s,vo_frame_id,vo_synchronized\nsky_1,0.1,27,1\nsky_1,0.3,28,1\n")
    with pytest.raises(ex.ExportError, match="duplicate"):
        ex.load_sim_observations(p)


def test_csv_round_trips_with_the_declared_header(tmp_path):
    north = [obs("n1", 0, 1.0, None)]
    src = FakeSource({"n1": [0.25, 0.5, 0.75, 1.0]})
    rows, _ = ex.build_rows(north, ex.pair_west(north, None), src, None, config(), 1e-6, None, 0, None, 0.05)
    path = tmp_path / ex.CSV_NAME
    ex.write_csv(path, rows)
    with path.open(newline="") as f:
        reader = csv.reader(f)
        assert next(reader) == ex.HEADER
        row = next(reader)
    assert len(row) == len(ex.HEADER) == 12
    assert row[6] == "0.25;0.5;0.75;1.0" and row[8] == "false"
