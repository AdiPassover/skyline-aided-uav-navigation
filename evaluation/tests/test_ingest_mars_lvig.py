"""Tests for the MARS-LVIG HKairport01 adapter (`naveval.ingest_mars_lvig`).

Built on a **constructed** RTK track and 1x1 JPEG stubs, so the suite never needs the real
dataset (~20 GB of rosbag, or 1.6 GB of redistributed JPEGs). The constructed track reproduces
the real flight's *shape* -- climb, 30 m hover, climb, cruise at 80 m, descent -- because the
trimming rule is the thing under test and a flat track would not exercise it.

One test additionally runs against the committed real RTK track
(`evaluation/tools/stage1/hkairport01_rtk_positions_raw.csv`, 490 KB) and skips if absent: the
constructed track proves the rule is implemented, the real one proves it produces the window
EXP-002 records.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from naveval.dataset import load_dataset
from naveval.ingest_flight import IngestError
from naveval.ingest_mars_lvig import (
    build_dataset,
    discover_frames,
    find_cruise_window,
    ground_datum_m,
    load_rtk_track,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_RTK = REPO_ROOT / "evaluation" / "tools" / "stage1" / "hkairport01_rtk_positions_raw.csv"

# Smallest valid JPEG (1x1 grey), so image handling is exercised without shipping real imagery.
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffc00011080001000103012200021101031101ffc400"
    "1f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303"
    "020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c115"
    "52d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a53545556"
    "5758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4"
    "a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7"
    "e8e9eaf1f2f3f4f5f6f7f8f9faffda0008010100003f00fb5e8a28a2803fffd9"
)

T0 = 1671606410.0
LAT0, LON0, GROUND_ALT = 22.41609, 114.04270, 99.71


def _agl_profile(t: float, ground_dwell_s: float) -> float:
    """Altitude profile mirroring the real flight, offset by a ground dwell at the start.

    `ground_dwell_s` of sitting at 0 m AGL, then climb to 30 m, hover, climb to 80 m, cruise,
    descend. The dwell matters: the datum rule takes the 1st percentile of the altitude series,
    which is only the true ground if enough samples were recorded at ground level.
    """
    t -= ground_dwell_s
    if t < 0:
        return 0.0
    if t < 40:
        return 30.0 * (t / 40.0)
    if t < 60:
        return 30.0
    if t < 100:
        return 30.0 + 50.0 * ((t - 60) / 40.0)
    if t <= 300:
        return 80.0
    return 80.0 * max(0.0, 1 - (t - 300) / 40.0)


def _constructed_track(tmp_path: Path, ground_dwell_s: float = 30.0, name: str = "rtk.csv") -> Path:
    """5 Hz track: ground dwell, climb to 30 m, hover, climb to 80 m, cruise north at 3 m/s
    from t=100+dwell to t=300+dwell, then descend. Mirrors the real profile's shape."""
    rows = []
    total = 340.0 + ground_dwell_s
    for k in range(int(total * 5) + 1):
        t = k / 5.0
        agl = _agl_profile(t, ground_dwell_s)
        tc = t - ground_dwell_s
        north_m = 3.0 * (min(tc, 300.0) - 100.0) if tc >= 100 else 0.0
        rows.append({
            "scenename": "HKairport01", "headerstamp": f"{T0 + t:.9f}",
            "lat": f"{LAT0 + north_m / 111320.0:.12f}", "lon": f"{LON0:.12f}",
            "alt": f"{GROUND_ALT + agl:.6f}",
            "easting": "195546.97", "northing": f"{2481881.73 + north_m:.4f}",
        })
    p = tmp_path / name
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


# Cruise runs from t = 100 + dwell to 300 + dwell; eroded by 5 s each end.
DWELL = 30.0
CRUISE_START_REL = DWELL + 100.0 + 5.0
CRUISE_END_REL = DWELL + 300.0 - 5.0


def _frames(tmp_path: Path, t_from: float, t_to: float, hz: float = 10.0) -> Path:
    d = tmp_path / "images"
    d.mkdir(exist_ok=True)
    n = int((t_to - t_from) * hz) + 1
    for k in range(n):
        (d / f"{T0 + t_from + k / hz:.9f}.jpg").write_bytes(TINY_JPEG)
    return d


class TestLoadRtkTrack:
    def test_reads_constructed_track(self, tmp_path):
        track = load_rtk_track(_constructed_track(tmp_path), expect_scene="HKairport01")
        assert len(track) == int((340.0 + DWELL) * 5) + 1
        assert track[0].scene == "HKairport01"

    def test_rejects_wrong_scene(self, tmp_path):
        with pytest.raises(IngestError, match="expected"):
            load_rtk_track(_constructed_track(tmp_path), expect_scene="AMtown01")

    def test_rejects_mixed_scenes(self, tmp_path):
        p = _constructed_track(tmp_path)
        lines = p.read_text().splitlines()
        lines[5] = lines[5].replace("HKairport01", "HKisland01", 1)
        p.write_text("\n".join(lines) + "\n")
        with pytest.raises(IngestError, match="multiple scenes"):
            load_rtk_track(p)

    def test_rejects_malformed_row(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("scenename,headerstamp,lat,lon,alt,easting,northing\nA,notanumber,1,2,3,4,5\n")
        with pytest.raises(IngestError, match="Malformed"):
            load_rtk_track(p)


class TestCruiseWindow:
    def test_datum_is_ground_not_cruise(self, tmp_path):
        track = load_rtk_track(_constructed_track(tmp_path))
        assert ground_datum_m(track) == pytest.approx(GROUND_ALT, abs=0.5)

    def test_window_excludes_takeoff_hover_and_descent(self, tmp_path):
        track = load_rtk_track(_constructed_track(tmp_path))
        start, end = find_cruise_window(track)
        assert start - T0 == pytest.approx(CRUISE_START_REL, abs=0.3)
        assert end - T0 == pytest.approx(CRUISE_END_REL, abs=0.3)

    def test_30m_hover_is_not_mistaken_for_cruise(self, tmp_path):
        # The 30 m hover is both the wrong altitude AND stationary; either condition alone
        # would exclude it, and the rule must not admit it.
        track = load_rtk_track(_constructed_track(tmp_path))
        start, _ = find_cruise_window(track)
        assert start - T0 > DWELL + 60.0

    def test_raises_when_no_cruise_exists(self, tmp_path):
        track = load_rtk_track(_constructed_track(tmp_path))
        with pytest.raises(IngestError, match="No cruise samples"):
            find_cruise_window(track, target_agl_m=500.0)

    def test_datum_rule_needs_ground_samples_and_fails_loudly_without_them(self, tmp_path):
        """Documented sensitivity of EXP-002's rule, found while writing these tests.

        The datum is the 1st percentile of the altitude series, so it is only the true ground
        if enough samples were recorded at ground level. A track that begins already climbing
        biases the datum upward, which biases AGL downward, which pushes the cruise outside the
        +/-0.5 m tolerance. The important property is that this **raises** rather than silently
        selecting a wrong window -- a silently-shifted datum would corrupt the altitude control
        variable that EXP-002's H4 depends on.
        """
        track = load_rtk_track(_constructed_track(tmp_path, ground_dwell_s=0.0, name="nodwell.csv"))
        assert ground_datum_m(track) > GROUND_ALT + 1.0  # biased high, as expected
        with pytest.raises(IngestError, match="No cruise samples"):
            find_cruise_window(track)


class TestDiscoverFrames:
    def test_sorted_by_timestamp(self, tmp_path):
        d = _frames(tmp_path, 100.0, 101.0)
        got = discover_frames(d)
        assert len(got) == 11
        assert [t for t, _ in got] == sorted(t for t, _ in got)

    def test_rejects_non_timestamp_name(self, tmp_path):
        d = _frames(tmp_path, 100.0, 100.2)
        (d / "frame_0001.jpg").write_bytes(TINY_JPEG)
        with pytest.raises(IngestError, match="epoch timestamp"):
            discover_frames(d)

    def test_raises_on_empty_dir(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(IngestError, match="No .jpg"):
            discover_frames(tmp_path / "empty")


class TestBuildDataset:
    def _build(self, tmp_path):
        rtk = _constructed_track(tmp_path)
        # deliberately wider than the cruise window at both ends, so trimming has work to do
        imgs = _frames(tmp_path, CRUISE_START_REL - 15.0, CRUISE_END_REL + 15.0)
        return build_dataset(
            tmp_path / "out", imgs, rtk,
            dataset_id="test-hkairport01", dataset_revision="t1",
            provenance="constructed test fixture", notes="unit test",
        )

    def test_produces_a_contract_conforming_dataset(self, tmp_path):
        summary = self._build(tmp_path)
        ds = load_dataset(summary["root"])  # the real loader is the assertion
        assert ds.dataset_id == "test-hkairport01"
        assert ds.evidence_tier == "T3"
        assert ds.position_quality.quality_class == "rtk_gnss"

    def test_frames_outside_the_cruise_window_are_trimmed(self, tmp_path):
        summary = self._build(tmp_path)
        supplied = int((CRUISE_END_REL + 15.0 - (CRUISE_START_REL - 15.0)) * 10) + 1
        expected = int((CRUISE_END_REL - CRUISE_START_REL) * 10) + 1
        assert summary["n_frames"] < supplied          # 30 s of frames really were dropped
        assert abs(summary["n_frames"] - expected) <= 10
        assert summary["agl_range_m"] == pytest.approx(0.0, abs=0.2)

    def test_jpeg_bytes_are_copied_unchanged(self, tmp_path):
        summary = self._build(tmp_path)
        out = sorted((summary["root"] / "images").glob("*.jpg"))
        assert out, "no images written"
        assert out[0].read_bytes() == TINY_JPEG  # byte-for-byte, never re-encoded

    def test_heading_is_empty_and_quality_unknown(self, tmp_path):
        """The RTK yaw convention is unverified (EXP-002). The adapter must emit no heading
        at all rather than guess one -- and the framework must therefore classify yaw
        unsupported rather than silently reporting a number."""
        summary = self._build(tmp_path)
        with (summary["root"] / "dataset.json").open() as f:
            dj = json.load(f)
        assert dj["heading_quality"]["class"] == "unknown"
        ds = load_dataset(summary["root"])
        assert not ds.has_heading
        with (summary["root"] / "groundtruth.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert all(r["heading_deg"] == "" for r in rows)

    def test_up_m_is_agl_not_ellipsoidal(self, tmp_path):
        summary = self._build(tmp_path)
        with (summary["root"] / "groundtruth.csv").open() as f:
            ups = [float(r["up_m"]) for r in csv.DictReader(f)]
        assert min(ups) == pytest.approx(80.0, abs=0.6)
        assert max(ups) == pytest.approx(80.0, abs=0.6)

    def test_height_quality_null_and_intrinsics_recorded(self, tmp_path):
        summary = self._build(tmp_path)
        with (summary["root"] / "dataset.json").open() as f:
            dj = json.load(f)
        assert dj["height_quality"] is None          # estimator produces no z at all
        assert dj["metadata"]["camera_intrinsics"]["fx"] == pytest.approx(1471.0653, abs=1e-3)
        assert "PROVISIONAL" in dj["metadata"]["position_quality_caveat"]

    def test_evidence_caveat_states_not_our_system(self, tmp_path):
        summary = self._build(tmp_path)
        with (summary["root"] / "dataset.json").open() as f:
            caveat = json.load(f)["evidence_caveat"]
        assert caveat and "NOT of this project's own" in caveat

    def test_explicit_window_overrides_altitude_rule(self, tmp_path):
        rtk = _constructed_track(tmp_path)
        imgs = _frames(tmp_path, 90.0, 310.0)
        s = build_dataset(
            tmp_path / "out2", imgs, rtk, dataset_id="w", dataset_revision="t1",
            window=(T0 + 150.0, T0 + 170.0),
        )
        assert s["n_frames"] == pytest.approx(201, abs=2)


class TestAgainstRealRtkTrack:
    """The constructed track proves the rule runs; this proves it yields EXP-002's numbers."""

    def test_real_track_reproduces_exp002_cruise_window(self):
        if not REAL_RTK.exists():
            pytest.skip("stage1 RTK track not present in this checkout")
        track = load_rtk_track(REAL_RTK, expect_scene="HKairport01")
        assert len(track) == 3900
        datum = ground_datum_m(track)
        assert datum == pytest.approx(99.708, abs=0.02)
        start, end = find_cruise_window(track)
        t0 = track[0].timestamp_s
        # EXP-002 Execution B: t_rel 100.4 -> 716.2 s
        assert start - t0 == pytest.approx(100.4, abs=0.3)
        assert end - t0 == pytest.approx(716.2, abs=0.3)
