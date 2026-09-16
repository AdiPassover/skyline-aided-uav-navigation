"""Integration tests for `naveval.ingest_mars_lvig`'s native-rate MCAP source
(`load_hkairport01_mcap_window`, `build_dataset_from_mcap`), added 2026-08-14 (gap G18).

Everything here is a small, fully synthesised MCAP file -- constructed the same way
`test_mcap_reader.py` constructs individual records, but assembled into a complete,
self-consistent file (Data + Summary + Footer + trailing magic) so the whole
fetch-by-time-window -> merge -> convert -> write pipeline can be exercised without the real
20 GB remote file. `mcap_reader`'s own record/CDR framing is already tested in
`test_mcap_reader.py`; this file tests the layer built on top of it.
"""

from __future__ import annotations

import csv
import json
import struct

import pytest

from naveval.dataset import load_dataset
from naveval.ingest_flight import IngestError
from naveval.metrics import vertical_motion_discarded
from naveval.ingest_mars_lvig import (
    HKAIRPORT01_RTK_YAW_TO_COMPASS_OFFSET_DEG,
    MARS_LVIG_CAMERA_TOPIC,
    MARS_LVIG_RTK_CONNECTION_TOPIC,
    MARS_LVIG_RTK_INFO_POSITION_TOPIC,
    MARS_LVIG_RTK_INFO_YAW_TOPIC,
    MARS_LVIG_RTK_POSITION_TOPIC,
    MARS_LVIG_RTK_YAW_TOPIC,
    McapFrame,
    RtkSample,
    build_dataset_from_mcap,
    find_cruise_window,
    load_hkairport01_mcap_window,
)
from naveval.mcap_reader import MAGIC, OP_CHANNEL, OP_CHUNK, OP_CHUNK_INDEX, OP_FOOTER

# ---------------------------------------------------------------- low-level MCAP builders
# Deliberately self-contained (not imported from test_mcap_reader.py) to keep this file's
# blast radius to itself; ~40 lines of framing primitives, the same shape as that file's own.


def _rec(op: int, payload: bytes) -> bytes:
    return bytes([op]) + struct.pack("<Q", len(payload)) + payload


def _mstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<I", len(b)) + b


def _channel_rec(cid: int, topic: str) -> bytes:
    return _rec(OP_CHANNEL, struct.pack("<HH", cid, 0) + _mstr(topic) + _mstr("cdr") + struct.pack("<I", 0))


def _message_rec(cid: int, log_time_ns: int, payload: bytes) -> bytes:
    return _rec(OP_MESSAGE_LOCAL, struct.pack("<HIQQ", cid, 0, log_time_ns, log_time_ns) + payload)


OP_MESSAGE_LOCAL = 0x05  # matches mcap_reader.OP_MESSAGE; avoided importing the private-ish name twice


def _chunk_rec(messages_bytes: bytes) -> bytes:
    payload = (struct.pack("<QQQ", 0, 0, len(messages_bytes)) + struct.pack("<I", 0)
               + _mstr("") + struct.pack("<Q", len(messages_bytes)) + messages_bytes)
    return _rec(OP_CHUNK, payload)


def _chunk_index_rec(start_ns: int, end_ns: int, offset: int, length: int, channel_ids) -> bytes:
    mio = b"".join(struct.pack("<HQ", c, 0) for c in channel_ids)
    payload = (struct.pack("<QQQQ", start_ns, end_ns, offset, length)
               + struct.pack("<I", len(mio)) + mio
               + struct.pack("<Q", 0) + _mstr("") + struct.pack("<QQ", length, length))
    return _rec(OP_CHUNK_INDEX, payload)


def _cdr(body: bytes) -> bytes:
    return b"\x00\x01\x00\x00" + body


class _CdrW:
    def __init__(self):
        self.b = bytearray()

    def _align(self, n):
        self.b += b"\x00" * ((-len(self.b)) % n)

    def _p(self, fmt, n, v):
        self._align(n)
        self.b += struct.pack("<" + fmt, v)
        return self

    def i8(self, v): self.b += struct.pack("<b", v); return self
    def u8(self, v): self.b += struct.pack("<B", v); return self
    def u16(self, v): return self._p("H", 2, v)
    def i16(self, v): return self._p("h", 2, v)
    def f64(self, v): return self._p("d", 8, v)

    def string(self, s):
        raw = s.encode() + b"\x00"
        self._p("I", 4, len(raw))
        self.b += raw
        return self

    def _header(self, sec, nsec, frame_id):
        self._p("i", 4, sec)
        self._p("I", 4, nsec)
        self.string(frame_id)
        return self

    def bytes_seq(self, data: bytes):
        self._p("I", 4, len(data))
        self.b += data
        return self

    def done(self) -> bytes:
        return _cdr(bytes(self.b))


def _navsatfix_payload(lat: float, lon: float, alt: float) -> bytes:
    w = _CdrW()
    w._header(0, 0, "")
    w.i8(0).u16(0)
    w.f64(lat).f64(lon).f64(alt)
    for _ in range(9):
        w.f64(0.0)
    w.u8(0)
    return w.done()


def _int16_payload(v: int) -> bytes:
    return _cdr(struct.pack("<h", v))


def _uint8_payload(v: int) -> bytes:
    return _cdr(struct.pack("<B", v))


def _compressed_image_payload(jpeg: bytes) -> bytes:
    w = _CdrW()
    w._header(0, 0, "")
    w.string("jpeg")
    w.bytes_seq(jpeg)
    return w.done()


TOPIC_IDS = {
    MARS_LVIG_CAMERA_TOPIC: 1,
    MARS_LVIG_RTK_POSITION_TOPIC: 2,
    MARS_LVIG_RTK_YAW_TOPIC: 3,
    MARS_LVIG_RTK_INFO_POSITION_TOPIC: 4,
    MARS_LVIG_RTK_INFO_YAW_TOPIC: 5,
    MARS_LVIG_RTK_CONNECTION_TOPIC: 6,
}


def build_synthetic_mcap(chunks_content: list) -> tuple[bytes, list]:
    """`chunks_content`: list of chunks, each a list of (topic, log_time_ns, payload).
    Returns (full_file_bytes, [(offset, length), ...] per chunk) for tests that need to
    assert on fetch economy."""
    data = bytearray()
    chunk_meta = []
    for messages in chunks_content:
        msg_bytes = b"".join(
            _message_rec(TOPIC_IDS[topic], lt, payload) for topic, lt, payload in messages
        )
        chunk_bytes = _chunk_rec(msg_bytes)
        offset = len(data)
        data += chunk_bytes
        times = [lt for _, lt, _ in messages]
        ids = sorted({TOPIC_IDS[topic] for topic, _, _ in messages})
        chunk_meta.append((min(times), max(times), offset, len(chunk_bytes), ids))

    summary = bytearray()
    for topic, cid in TOPIC_IDS.items():
        summary += _channel_rec(cid, topic)
    for start, end, offset, length, ids in chunk_meta:
        summary += _chunk_index_rec(start, end, offset, length, ids)

    summary_start = len(data)
    footer = _rec(OP_FOOTER, struct.pack("<QQI", summary_start, 0, 0))
    full = bytes(data) + bytes(summary) + footer + MAGIC
    return full, [(o, l) for _, _, o, l, _ in chunk_meta]


def _fetchers(full_bytes: bytes, call_log: list = None):
    def fetch_range(start, end):
        if call_log is not None:
            call_log.append((start, end))
        return full_bytes[start:end + 1]

    def fetch_tail(n):
        return full_bytes[-n:]

    return fetch_range, fetch_tail


def _rtk_cycle(t_ns: int, lat: float, lon: float, alt: float, rtk_yaw_raw: int,
              info_pos=50, info_yaw=50, conn=1, skip=()) -> list:
    """The five RTK-topic messages for one broadcast cycle. `skip` names topics to omit,
    for testing that an incomplete cycle is dropped rather than guessed."""
    msgs = [
        (MARS_LVIG_RTK_POSITION_TOPIC, t_ns, _navsatfix_payload(lat, lon, alt)),
        (MARS_LVIG_RTK_YAW_TOPIC, t_ns, _int16_payload(rtk_yaw_raw)),
        (MARS_LVIG_RTK_INFO_POSITION_TOPIC, t_ns, _uint8_payload(info_pos)),
        (MARS_LVIG_RTK_INFO_YAW_TOPIC, t_ns, _uint8_payload(info_yaw)),
        (MARS_LVIG_RTK_CONNECTION_TOPIC, t_ns, _uint8_payload(conn)),
    ]
    return [m for m in msgs if m[0] not in skip]


LAT0, LON0, ALT0 = 22.4160794, 114.0422944, 179.643


class TestNativeRateCameraFrames:
    def test_five_frames_at_10hz_spacing_are_read_with_correct_timestamps_and_bytes(self):
        jpegs = [f"frame{k}".encode() for k in range(5)]
        cam_msgs = [
            (MARS_LVIG_CAMERA_TOPIC, k * 100_000_000, _compressed_image_payload(jpegs[k]))
            for k in range(5)
        ]
        rtk_msgs = _rtk_cycle(0, LAT0, LON0, ALT0, 176) + _rtk_cycle(400_000_000, LAT0, LON0, ALT0, 176)
        full, _ = build_synthetic_mcap([cam_msgs + rtk_msgs])
        fetch_range, fetch_tail = _fetchers(full)

        rtk, frames = load_hkairport01_mcap_window(fetch_range, fetch_tail, len(full), 0, 500_000_000)

        assert [f.timestamp_s for f in frames] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
        assert [f.data for f in frames] == jpegs  # byte-for-byte, no re-encode
        assert len(rtk) == 2


class TestRtkTopicMerging:
    def test_merges_five_topics_by_matching_log_time(self):
        rtk_msgs = _rtk_cycle(0, LAT0, LON0, ALT0, 176, info_pos=50, info_yaw=50, conn=1)
        full, _ = build_synthetic_mcap([rtk_msgs])
        fetch_range, fetch_tail = _fetchers(full)

        rtk, _ = load_hkairport01_mcap_window(fetch_range, fetch_tail, len(full), 0, 1_000_000)

        assert len(rtk) == 1
        s = rtk[0]
        assert s.lat_deg == pytest.approx(LAT0)
        assert s.lon_deg == pytest.approx(LON0)
        assert s.alt_m == pytest.approx(ALT0)
        assert s.rtk_yaw_raw == 176
        assert s.rtk_info_position == 50
        assert s.rtk_info_yaw == 50
        assert s.rtk_connection_status == 1

    def test_incomplete_cycle_is_dropped_not_guessed(self):
        complete = _rtk_cycle(0, LAT0, LON0, ALT0, 176)
        incomplete = _rtk_cycle(200_000_000, LAT0, LON0, ALT0, 290, skip=(MARS_LVIG_RTK_YAW_TOPIC,))
        full, _ = build_synthetic_mcap([complete + incomplete])
        fetch_range, fetch_tail = _fetchers(full)

        rtk, _ = load_hkairport01_mcap_window(fetch_range, fetch_tail, len(full), 0, 1_000_000)

        assert len(rtk) == 1
        assert rtk[0].timestamp_s == pytest.approx(0.0)


class TestChunkTimeWindowing:
    def test_only_the_overlapping_chunk_is_fetched(self):
        c1 = _rtk_cycle(0, LAT0, LON0, ALT0, 176)
        c2 = _rtk_cycle(1_000_000_000, LAT0, LON0, ALT0, 290)
        c3 = _rtk_cycle(2_000_000_000, LAT0, LON0, ALT0, 353)
        full, chunk_meta = build_synthetic_mcap([c1, c2, c3])
        calls = []
        fetch_range, fetch_tail = _fetchers(full, calls)

        rtk, _ = load_hkairport01_mcap_window(
            fetch_range, fetch_tail, len(full), 900_000_000, 1_100_000_000)

        assert len(rtk) == 1
        assert rtk[0].rtk_yaw_raw == 290
        # load_hkairport01_mcap_window makes exactly two fetch_range calls: the Summary
        # section, then one contiguous range for the overlapping chunk(s). With a single
        # overlapping chunk, that second call's span must equal that chunk's own length --
        # not the whole file, and not any of the other two chunks.
        assert len(calls) == 2, f"expected exactly 2 fetch_range calls (summary + chunk data), got {calls}"
        fetched_len = calls[1][1] - calls[1][0] + 1
        assert fetched_len == chunk_meta[1][1], "fetched byte span should equal chunk 2's own length"
        assert fetched_len < len(full) // 2, "should not have fetched anywhere near the whole file"

    def test_missing_required_topic_raises(self):
        # Build a summary with only 5 of the 6 required channels.
        topics = dict(TOPIC_IDS)
        topics.pop(MARS_LVIG_RTK_YAW_TOPIC)
        data = bytearray()
        msgs = [m for m in _rtk_cycle(0, LAT0, LON0, ALT0, 176) if m[0] != MARS_LVIG_RTK_YAW_TOPIC]
        msg_bytes = b"".join(_message_rec(TOPIC_IDS[t], lt, p) for t, lt, p in msgs)
        chunk_bytes = _chunk_rec(msg_bytes)
        data += chunk_bytes
        summary = bytearray()
        for topic, cid in topics.items():
            summary += _channel_rec(cid, topic)
        ids = sorted({TOPIC_IDS[t] for t, _, _ in msgs})
        summary += _chunk_index_rec(0, 0, 0, len(chunk_bytes), ids)
        summary_start = len(data)
        footer = _rec(OP_FOOTER, struct.pack("<QQI", summary_start, 0, 0))
        full = bytes(data) + bytes(summary) + footer + MAGIC
        fetch_range, fetch_tail = _fetchers(full)

        with pytest.raises(IngestError, match="missing required topic"):
            load_hkairport01_mcap_window(fetch_range, fetch_tail, len(full), 0, 1_000_000)


class TestYawConversionWraparound:
    @pytest.mark.parametrize("raw_yaw,expected_compass", [
        (176, (176 - 269.16) % 360),   # 266.84, matches the real measured leg (~268.1 track)
        (0, (0 - 269.16) % 360),       # 90.84
        (270, (270 - 269.16) % 360),   # 0.84 -- wraps to just past zero
        (269, (269 - 269.16) % 360),   # 359.84 -- wraps to just under 360
    ])
    def test_compass_heading_wraps_correctly(self, raw_yaw, expected_compass, tmp_path):
        rtk_msgs = (_rtk_cycle(0, LAT0, LON0, ALT0, raw_yaw)
                   + _rtk_cycle(1_000_000_000, LAT0 + 0.0001, LON0, ALT0, raw_yaw))
        full, _ = build_synthetic_mcap([rtk_msgs])
        fetch_range, fetch_tail = _fetchers(full)
        rtk, frames = load_hkairport01_mcap_window(fetch_range, fetch_tail, len(full), 0, 2_000_000_000)
        frames = [McapFrame(0.0, b"x"), McapFrame(1.0, b"y")]  # dummy frames to satisfy build_dataset_from_mcap

        result = build_dataset_from_mcap(
            tmp_path / "out", rtk, frames, "yaw-test", "v1",
            window=(0.0, 1.0),
        )
        with (result["root"] / "groundtruth.csv").open() as f:
            rows = list(csv.DictReader(f))
        headings = [float(r["heading_deg"]) for r in rows if r["heading_deg"] != ""]
        assert headings
        for h in headings:
            assert h == pytest.approx(expected_compass, abs=1e-6)
            assert 0.0 <= h < 360.0


class TestRtkFixedQualityPropagation:
    def test_non_fixed_sample_marked_invalid_in_groundtruth(self, tmp_path):
        fixed = _rtk_cycle(0, LAT0, LON0, ALT0, 176, info_pos=50, info_yaw=50, conn=1)
        not_fixed = _rtk_cycle(500_000_000, LAT0 + 0.00001, LON0, ALT0, 176,
                               info_pos=34, info_yaw=50, conn=1)  # 34 = floating narrow-lane
        still_fixed = _rtk_cycle(1_000_000_000, LAT0 + 0.00002, LON0, ALT0, 176)
        full, _ = build_synthetic_mcap([fixed + not_fixed + still_fixed])
        fetch_range, fetch_tail = _fetchers(full)
        rtk, _ = load_hkairport01_mcap_window(fetch_range, fetch_tail, len(full), 0, 2_000_000_000)
        frames = [McapFrame(0.0, b"x"), McapFrame(1.0, b"y")]

        result = build_dataset_from_mcap(
            tmp_path / "out", rtk, frames, "fixquality-test", "v1", window=(0.0, 1.0),
        )
        with (result["root"] / "groundtruth.csv").open() as f:
            rows = list(csv.DictReader(f))
        valids = [r["valid"] for r in rows]
        assert "false" in valids, "the non-fixed sample must be marked invalid, not silently included"
        assert "true" in valids
        assert result["n_rtk_fixed"] == 2
        assert result["n_rtk_total"] == 3


class TestCruiseWindowTrimmingReused:
    def test_find_cruise_window_operates_on_mcap_merged_samples(self):
        """`find_cruise_window` (naveval.ingest_flight-adjacent logic already exercised by the
        CSV path's tests) must work unchanged on MCAP-sourced RtkSample instances, which carry
        extra Optional quality fields it never touches."""
        track = []
        for k in range(41):  # 0..4.0s at 100ms steps
            t = k * 0.1
            if t < 1.0:
                agl = 80.0 * (t / 1.0)
                north = 0.0
            elif t <= 3.0:
                agl = 80.0
                north = 3.0 * (t - 1.0)
            else:
                agl = 80.0 * max(0.0, 1 - (t - 3.0) / 1.0)
                north = 6.0
            track.append(RtkSample(
                scene="", timestamp_s=t, lat_deg=LAT0 + north / 111320.0, lon_deg=LON0,
                alt_m=100.0 + agl, easting=0.0, northing=north,
                rtk_yaw_raw=176, rtk_info_position=50, rtk_info_yaw=50, rtk_connection_status=1,
            ))

        start, end = find_cruise_window(track, target_agl_m=80.0, tolerance_m=0.5,
                                        min_speed_ms=2.0, erode_s=0.3)

        assert 1.0 <= start <= 1.5
        assert 2.5 <= end <= 3.0


class TestBuildDatasetFromMcapEndToEnd:
    N_CAM_FRAMES_SUPPLIED = 12  # t = 0.0 .. 1.1s; window (0.0, 1.0) admits 11 of these

    def _build(self, tmp_path):
        rtk_msgs = (_rtk_cycle(0, LAT0, LON0, ALT0, 176)
                   + _rtk_cycle(500_000_000, LAT0 + 0.00001, LON0, ALT0, 176)
                   + _rtk_cycle(1_000_000_000, LAT0 + 0.00002, LON0, ALT0, 176))
        cam_msgs = [
            (MARS_LVIG_CAMERA_TOPIC, k * 100_000_000, _compressed_image_payload(f"f{k}".encode()))
            for k in range(self.N_CAM_FRAMES_SUPPLIED)
        ]
        full, _ = build_synthetic_mcap([rtk_msgs + cam_msgs])
        fetch_range, fetch_tail = _fetchers(full)
        rtk, frames = load_hkairport01_mcap_window(fetch_range, fetch_tail, len(full), 0, 2_000_000_000)
        assert len(frames) == self.N_CAM_FRAMES_SUPPLIED  # sanity: window reader kept them all
        return build_dataset_from_mcap(
            tmp_path / "out", rtk, frames, "e2e-test", "v1",
            window=(0.0, 1.0), mcap_source_url="https://example.invalid/HKairport01_0.mcap",
        )

    def test_produces_a_contract_conforming_dataset(self, tmp_path):
        result = self._build(tmp_path)
        ds = load_dataset(result["root"])
        assert ds.evidence_tier == "T3"
        assert ds.position_quality.quality_class == "rtk_gnss"
        assert ds.heading_quality.quality_class == "rtk_gnss"
        assert ds.has_heading
        assert ds.height_quality is None  # estimator never produces z

    def test_frame_rate_and_provenance_recorded_in_metadata(self, tmp_path):
        result = self._build(tmp_path)
        with (result["root"] / "dataset.json").open() as f:
            dj = json.load(f)
        assert dj["metadata"]["frame_rate_hz"] == pytest.approx(10.0, abs=0.5)
        conv = dj["metadata"]["heading_conversion"]
        assert conv["offset_deg"] == HKAIRPORT01_RTK_YAW_TO_COMPASS_OFFSET_DEG
        assert "269.16" in conv["formula"]
        assert conv["quantisation_deg"] == 1.0
        assert dj["metadata"]["mcap_source_url"] == "https://example.invalid/HKairport01_0.mcap"
        assert "sequence" in conv["derivation"].lower() or "sequence" in dj["evidence_caveat"].lower()

    def test_frames_outside_window_are_trimmed(self, tmp_path):
        result = self._build(tmp_path)
        assert result["n_frames"] == 11  # excludes the k=11 frame at t=1.1s, outside (0.0, 1.0)
        assert result["n_frames"] < self.N_CAM_FRAMES_SUPPLIED


class TestGroundDatumOverride:
    """Found while running EXP-002 Execution A (2026-08-14, research log): a track fetched only
    for a cruise-only time window never touches the ground, so ground_datum_m's 1st-percentile
    estimate silently lands near the cruise altitude instead of raising. build_dataset_from_mcap
    must accept an externally-supplied datum for exactly this case."""

    CRUISE_ALT = 179.643  # matches the real HKairport01 cruise absolute altitude

    def _cruise_only_track_and_frames(self, tmp_path):
        # All samples at cruise altitude -- no ground-level sample anywhere, mirroring a
        # narrow MCAP fetch window.
        rtk_msgs = (
            _rtk_cycle(0, LAT0, LON0, self.CRUISE_ALT, 176)
            + _rtk_cycle(500_000_000, LAT0 + 0.00001, LON0, self.CRUISE_ALT, 176)
            + _rtk_cycle(1_000_000_000, LAT0 + 0.00002, LON0, self.CRUISE_ALT, 176)
        )
        cam_msgs = [
            (MARS_LVIG_CAMERA_TOPIC, k * 100_000_000, _compressed_image_payload(f"f{k}".encode()))
            for k in range(11)
        ]
        full, _ = build_synthetic_mcap([rtk_msgs + cam_msgs])
        fetch_range, fetch_tail = _fetchers(full)
        return load_hkairport01_mcap_window(fetch_range, fetch_tail, len(full), 0, 1_000_000_000)

    def test_default_datum_on_a_cruise_only_track_is_near_cruise_altitude_not_ground(self, tmp_path):
        """Documents the failure mode: with no ground-level sample in the track, the
        auto-computed datum lands near 0 m AGL instead of the true ~80 m."""
        rtk, frames = self._cruise_only_track_and_frames(tmp_path)
        result = build_dataset_from_mcap(
            tmp_path / "out", rtk, frames, "datum-bug-test", "v1", window=(0.0, 1.0),
        )
        assert abs(result["agl_mean_m"]) < 1.0  # the bug: nowhere near the real ~80 m AGL

    def test_explicit_datum_override_recovers_correct_agl(self, tmp_path):
        ground_datum = self.CRUISE_ALT - 80.0  # the "whole-sequence" datum a real flight would give
        rtk, frames = self._cruise_only_track_and_frames(tmp_path)
        result = build_dataset_from_mcap(
            tmp_path / "out", rtk, frames, "datum-fixed-test", "v1", window=(0.0, 1.0),
            ground_datum_override_m=ground_datum,
        )
        assert result["ground_datum_m"] == ground_datum
        assert result["agl_mean_m"] == pytest.approx(80.0, abs=0.01)
        with (result["root"] / "dataset.json").open() as f:
            dj = json.load(f)
        assert dj["local_frame_origin"]["alt_m"] == ground_datum
        assert dj["metadata"]["agl_mean_m"] == pytest.approx(80.0, abs=0.01)

    def test_datum_override_does_not_change_reported_vertical_motion_metric(self, tmp_path):
        """vertical_motion_discarded (naveval.metrics) computes range and RMS-about-mean, both
        shift-invariant -- the datum bug corrupts metadata, never a reported metric. Confirmed
        directly rather than assumed."""
        rtk, frames = self._cruise_only_track_and_frames(tmp_path)
        r_bug = build_dataset_from_mcap(
            tmp_path / "bug", rtk, frames, "a", "v1", window=(0.0, 1.0),
        )
        r_fixed = build_dataset_from_mcap(
            tmp_path / "fixed", rtk, frames, "b", "v1", window=(0.0, 1.0),
            ground_datum_override_m=self.CRUISE_ALT - 80.0,
        )
        with (r_bug["root"] / "groundtruth.csv").open() as f:
            up_bug = [float(r["up_m"]) for r in csv.DictReader(f)]
        with (r_fixed["root"] / "groundtruth.csv").open() as f:
            up_fixed = [float(r["up_m"]) for r in csv.DictReader(f)]
        m_bug = vertical_motion_discarded(up_bug)
        m_fixed = vertical_motion_discarded(up_fixed)
        assert m_bug["range_m"] == pytest.approx(m_fixed["range_m"], abs=1e-9)
        assert m_bug["rms_about_mean_m"] == pytest.approx(m_fixed["rms_about_mean_m"], abs=1e-9)


class TestSequenceSpecificFieldsAreParameters:
    """Added 2026-08-25 (`EXP-VO-007`) with the generalisation that let a second MARS-LVIG
    sequence be ingested by the same code.

    Two things must hold and neither is obvious from reading the signature: the **defaults must
    still produce exactly the HKairport01 dataset** that `datasets/hkairport01-{a,b}` were built
    from, and the overrides must actually reach `groundtruth.csv` -- the yaw offset is not merely
    metadata, it rotates every heading in the file. `LIT-006` derived that offset from 52 samples
    on one sequence and states it is sequence-specific; inheriting it silently would be a ~90 deg
    heading error that looks entirely plausible.
    """

    def _build(self, tmp_path, out, **kwargs):
        rtk_msgs = (_rtk_cycle(0, LAT0, LON0, ALT0, 176)
                    + _rtk_cycle(500_000_000, LAT0 + 0.00001, LON0, ALT0, 176)
                    + _rtk_cycle(1_000_000_000, LAT0 + 0.00002, LON0, ALT0, 176))
        cam_msgs = [
            (MARS_LVIG_CAMERA_TOPIC, k * 100_000_000, _compressed_image_payload(f"f{k}".encode()))
            for k in range(12)
        ]
        full, _ = build_synthetic_mcap([rtk_msgs + cam_msgs])
        fetch_range, fetch_tail = _fetchers(full)
        rtk, frames = load_hkairport01_mcap_window(
            fetch_range, fetch_tail, len(full), 0, 2_000_000_000)
        return build_dataset_from_mcap(
            tmp_path / out, rtk, frames, f"seq-{out}", "v1", window=(0.0, 1.0), **kwargs)

    def _meta(self, result):
        with (result["root"] / "dataset.json").open() as f:
            return json.load(f)

    def test_hkairport01_defaults_are_the_previously_hardcoded_values(self, tmp_path):
        dj = self._meta(self._build(tmp_path, "default"))
        m = dj["metadata"]
        assert m["environment"] == (
            "aero-model airfield, Yuen Long, Hong Kong; concrete runway, dry grass, tree canopy")
        assert m["capture_date"] == "2022-12-21"
        assert m["camera_intrinsics"]["fx"] == pytest.approx(1471.0653076171875)
        assert m["heading_conversion"]["offset_deg"] == HKAIRPORT01_RTK_YAW_TO_COMPASS_OFFSET_DEG
        assert "269.16" in m["heading_conversion"]["formula"]
        assert "52 RTK samples" in m["heading_conversion"]["derivation"]
        assert "Hikvision CA-050-11UC" in dj["evidence_caveat"]
        assert "rigidly mounted" in m["attitude_caveat"]

    def test_overrides_replace_every_sequence_specific_field(self, tmp_path):
        dj = self._meta(self._build(
            tmp_path, "override",
            environment="semi-desert village",
            capture_date="2022-07-18",
            camera_intrinsics={"fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 3.0,
                               "width": 640, "height": 480, "source": "test"},
            evidence_caveat="a different caveat",
            attitude_caveat="a different attitude caveat",
            heading_offset_derivation="re-derived for this sequence",
        ))
        m = dj["metadata"]
        assert m["environment"] == "semi-desert village"
        assert m["capture_date"] == "2022-07-18"
        assert (m["image_width"], m["image_height"]) == (640, 480)
        assert m["camera_intrinsics"]["fx"] == 1.0
        assert dj["evidence_caveat"] == "a different caveat"
        assert m["attitude_caveat"] == "a different attitude caveat"
        assert m["heading_conversion"]["derivation"] == "re-derived for this sequence"

    def test_yaw_offset_override_rotates_the_written_headings(self, tmp_path):
        """The offset is data, not documentation: it must move groundtruth.csv itself."""
        def headings(result):
            with (result["root"] / "groundtruth.csv").open() as f:
                return [float(r["heading_deg"]) for r in csv.DictReader(f)]

        base = headings(self._build(tmp_path, "yaw_base"))
        moved = headings(self._build(tmp_path, "yaw_moved",
                                     rtk_yaw_to_compass_offset_deg=-179.16))
        assert base and moved and len(base) == len(moved)
        for b, mv in zip(base, moved):
            assert (mv - b) % 360.0 == pytest.approx(90.0, abs=1e-6)
        dj = self._meta(self._build(tmp_path, "yaw_meta",
                                    rtk_yaw_to_compass_offset_deg=-179.16))
        assert dj["metadata"]["heading_conversion"]["offset_deg"] == -179.16
        assert "-179.16" in dj["metadata"]["heading_conversion"]["formula"]
