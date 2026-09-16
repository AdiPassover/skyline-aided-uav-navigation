"""Ingest MARS-LVIG `HKairport01` into a conforming dataset (contracts/dataset.md).

**Narrow by design.** This handles exactly one sequence family from one dataset. It is NOT a
general MARS-LVIG reader.

Two sources are supported, sharing the same cruise-window/ENU/dataset-writing logic:

- **`build_dataset`** — the UAVScenes-redistributed export (per-frame JPEGs named `<epoch>.jpg`
  + `rtk_positions_raw.csv`, interval-5 = 2 Hz on the copy actually hosted). Used for the A0
  plumbing check (research log 2026-08-13). Heading is NOT emitted from this path: this export
  carries no yaw topic at all.
- **`build_dataset_from_mcap`** (added 2026-08-14, gap G18) — the native 10 Hz ROS 2/MCAP mirror
  (`DapengFeng/MCAP` on HuggingFace; see `LIT-006` "Stage 2"). This is what EXP-002 Execution A/B
  requires: native camera rate, per-sample RTK fix-quality, and verified yaw. Reads only the
  chunks overlapping the requested time window (`naveval.mcap_reader`), so a 180 s window costs
  ~4.7 GB of a 20.4 GB file rather than the whole thing.

Neither path speaks ROS 1 / rosbag directly -- the 2026-08-13 conclusion that the rosbags were
unreachable was itself wrong (see `LIT-006`), but by the time that was corrected the MCAP mirror
was already the more efficient source for partial, time-windowed reads, so it remains the one this
module targets. `find_cruise_window`, `ground_datum_m` and the dataset-writing/quality functions
from `naveval.ingest_flight` are shared between both paths unchanged.
"""

from __future__ import annotations

import bisect
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from naveval import mcap_reader as mcap
from naveval.frames import geodetic_to_enu
from naveval.ingest_flight import (
    ClockSync,
    FrameRecord,
    IngestError,
    TelemetrySample,
    build_dataset_json,
    convert_geodetic_track_to_enu,
    write_dataset,
)
from naveval.mcap_reader import (
    HKAIRPORT01_RTK_YAW_TO_COMPASS_OFFSET_DEG,
    rtk_yaw_to_compass_deg,
)

# LIT-006: the RTK track is exported at 5 Hz; the native camera stream is 10 Hz. Both are
# timestamped on GPS UTC by a hardware PPS trigger, so no clock offset exists to estimate.
CLOCK_SYNC = ClockSync(
    offset_s=0.0,
    drift_s_per_s=0.0,
    source_description=(
        "none required: MARS-LVIG triggers the camera from a GPS-PPS-derived 10 Hz pulse and "
        "timestamps both imagery and RTK on GPS UTC (Li et al. 2024, IJRR, section 3.3), so the "
        "offset is structurally zero rather than estimated"
    ),
)

# EXP-002's cruise rule. Defaults are the rule as written there; callers may override for
# other MARS-LVIG sequences, whose cruise altitudes differ (HKisland 90 m, AMvalley 130 m).
DEFAULT_TARGET_AGL_M = 80.0
DEFAULT_AGL_TOLERANCE_M = 0.5
DEFAULT_MIN_SPEED_MS = 2.0
DEFAULT_ERODE_S = 5.0

# Dataset-provided calibration (UAVScenes `sampleinfos_interpolated.json`, HKairport01):
# fx = fy = 1471.0653, cx = 1172.358, cy = 1046.367, K1..K3 = P1..P2 = 0, 2448 x 2048.
# Recorded because COMP-001 section 2 notes no intrinsics are an input anywhere in the system;
# carrying them costs nothing now and cannot be recovered later.
CAMERA_INTRINSICS = {
    "fx": 1471.0653076171875, "fy": 1471.0653076171875,
    "cx": 1172.3576676454904, "cy": 1046.3674075128438,
    "k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0,
    "width": 2448, "height": 2048,
    "source": "UAVScenes sampleinfos_interpolated.json (DJI Terra calibration)",
}


@dataclass(frozen=True)
class RtkSample:
    """One RTK ground-truth sample, before any conversion.

    `easting`/`northing` are locally-flat projected metres used only for the internal
    speed/window calculations in this module (`_speeds`, `find_cruise_window`) -- never written
    to a dataset output directly. For the CSV source they are MARS-LVIG's own UTM easting/
    northing columns; for the MCAP source (`_fill_local_projected_coords`) they are ENU metres
    relative to the first sample. Both are locally Cartesian to well within the tolerance these
    heuristics need.

    `rtk_yaw_raw`/`rtk_info_position`/`rtk_info_yaw`/`rtk_connection_status` are populated only
    by the MCAP path (added 2026-08-14, gap G18) -- the CSV export carries none of them.
    """
    scene: str
    timestamp_s: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    easting: float
    northing: float
    rtk_yaw_raw: Optional[int] = None
    rtk_info_position: Optional[int] = None
    rtk_info_yaw: Optional[int] = None
    rtk_connection_status: Optional[int] = None


def load_rtk_track(csv_path: Path | str, expect_scene: Optional[str] = None) -> list[RtkSample]:
    """Read `rtk_positions_raw.csv`. Rejects a mixed-scene file rather than silently
    concatenating two flights -- the same class of check `verify_matches_dataset` makes."""
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            try:
                rows.append(RtkSample(
                    scene=r["scenename"], timestamp_s=float(r["headerstamp"]),
                    lat_deg=float(r["lat"]), lon_deg=float(r["lon"]), alt_m=float(r["alt"]),
                    easting=float(r["easting"]), northing=float(r["northing"]),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise IngestError(f"Malformed RTK row {i} in {csv_path}: {exc}") from exc
    if not rows:
        raise IngestError(f"No RTK samples in {csv_path}")
    scenes = {r.scene for r in rows}
    if len(scenes) != 1:
        raise IngestError(f"RTK track spans multiple scenes {sorted(scenes)}; expected exactly one")
    if expect_scene is not None and rows[0].scene != expect_scene:
        raise IngestError(f"RTK track is scene {rows[0].scene!r}, expected {expect_scene!r}")
    return rows


def ground_datum_m(track: Sequence[RtkSample]) -> float:
    """Altitude datum = 1st percentile of the raw altitude series (EXP-002's rule).

    The raw `alt` is ellipsoidal/MSL, not AGL; the 1st percentile picks the ground plane at
    takeoff without being thrown by a single low outlier. Linear interpolation between the two
    bracketing order statistics, so the result does not depend on sample count parity.
    """
    a = sorted(s.alt_m for s in track)
    if len(a) == 1:
        return a[0]
    pos = 0.01 * (len(a) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (pos - lo)


def _speeds(track: Sequence[RtkSample]) -> list[float]:
    """Central-difference horizontal speed in the projected (UTM) frame, m/s."""
    n = len(track)
    out = []
    for i in range(n):
        j0, j1 = max(0, i - 1), min(n - 1, i + 1)
        dt = track[j1].timestamp_s - track[j0].timestamp_s
        if dt <= 0:
            out.append(0.0)
            continue
        de = track[j1].easting - track[j0].easting
        dn = track[j1].northing - track[j0].northing
        out.append((de * de + dn * dn) ** 0.5 / dt)
    return out


def find_cruise_window(
    track: Sequence[RtkSample],
    target_agl_m: float = DEFAULT_TARGET_AGL_M,
    tolerance_m: float = DEFAULT_AGL_TOLERANCE_M,
    min_speed_ms: float = DEFAULT_MIN_SPEED_MS,
    erode_s: float = DEFAULT_ERODE_S,
) -> tuple[float, float]:
    """EXP-002's trimming rule, in code so it cannot drift from the record.

    Cruise = first..last sample with |AGL - target| <= tolerance AND speed >= min_speed,
    then eroded by `erode_s` at each end for settling. Excludes the takeoff preamble (climb,
    30 m hover, climb, hover at the start waypoint) and the descent by construction.

    Returns absolute (start, end) timestamps on the same clock as the track.
    """
    datum = ground_datum_m(track)
    sp = _speeds(track)
    idx = [
        i for i, s in enumerate(track)
        if abs((s.alt_m - datum) - target_agl_m) <= tolerance_m and sp[i] >= min_speed_ms
    ]
    if not idx:
        raise IngestError(
            f"No cruise samples at {target_agl_m} +/- {tolerance_m} m AGL with speed >= "
            f"{min_speed_ms} m/s (datum {datum:.3f} m)"
        )
    start = track[idx[0]].timestamp_s + erode_s
    end = track[idx[-1]].timestamp_s - erode_s
    if end <= start:
        raise IngestError(f"Cruise window collapsed after {erode_s} s erosion at each end")
    return start, end


def discover_frames(images_dir: Path | str) -> list[tuple[float, Path]]:
    """MARS-LVIG/UAVScenes name each frame `<epoch_seconds>.jpg`, so the filename *is* the
    GPS-UTC timestamp. Returned sorted by time; a non-numeric name is an error rather than
    something to skip quietly."""
    out = []
    for p in sorted(Path(images_dir).glob("*.jpg")):
        try:
            out.append((float(p.stem), p))
        except ValueError as exc:
            raise IngestError(f"Image name {p.name!r} is not an epoch timestamp") from exc
    if not out:
        raise IngestError(f"No .jpg frames in {images_dir}")
    out.sort(key=lambda t: t[0])
    return out


def build_dataset(
    out_root: Path | str,
    images_dir: Path | str,
    rtk_csv: Path | str,
    dataset_id: str,
    dataset_revision: str,
    *,
    scene: str = "HKairport01",
    window: Optional[tuple[float, float]] = None,
    target_agl_m: float = DEFAULT_TARGET_AGL_M,
    provenance: str = "",
    notes: str = "",
    nominal_speed_ms: float = 3.0,
    copy_images: bool = True,
) -> dict:
    """Write a conforming dataset directory. Returns a small summary dict for the caller/tests.

    `window` overrides the altitude-derived cruise window (EXP-002 Execution A uses a
    sub-interval of it); when omitted, `find_cruise_window` supplies it.

    Three deliberate choices, each recorded because a later reader will otherwise assume the
    opposite:

    - **JPEG payloads are copied byte-for-byte**, never re-encoded. Transcoding to PNG would add
      a second generation of loss to imagery that is already lossy (LIT-006).
    - **`up_m` is AGL**, achieved by passing the ground datum as the ENU origin altitude. Height
      metrics stay `unsupported` (`height_quality: null`) because the estimator does not produce
      `z` at all; altitude is carried purely as EXP-002's control variable.
    - **`heading_deg` is left empty and `heading_quality` is `unknown`.** The RTK yaw topic is not
      present in this export and its convention has never been verified. Emitting a heading here
      would be inventing one; leaving it empty makes the evaluator omit yaw metrics with a stated
      reason (SC-014), which is the honest outcome.
    """
    out_root = Path(out_root)
    track = load_rtk_track(rtk_csv, expect_scene=scene)
    datum = ground_datum_m(track)
    t_start, t_end = window if window is not None else find_cruise_window(track, target_agl_m)

    frames = [(t, p) for t, p in discover_frames(images_dir) if t_start <= t <= t_end]
    if not frames:
        raise IngestError(f"No frames inside window [{t_start:.3f}, {t_end:.3f}]")

    # Ground truth is kept slightly wider than the frames so every frame interpolates from
    # bracketing samples rather than extrapolating at the edges.
    pad = 1.0
    sel = [s for s in track if (t_start - pad) <= s.timestamp_s <= (t_end + pad)]
    if len(sel) < 2:
        raise IngestError("Fewer than 2 RTK samples in the window; cannot interpolate")

    samples = [
        TelemetrySample(
            timestamp_s=s.timestamp_s, lat_deg=s.lat_deg, lon_deg=s.lon_deg, alt_m=s.alt_m,
            heading_deg=None,  # see docstring: not present in this export, convention unverified
            fix_quality="rtk_fixed_assumed",
            valid=True,
        )
        for s in sel
    ]
    gt_rows = convert_geodetic_track_to_enu(
        samples, origin_lat_deg=sel[0].lat_deg, origin_lon_deg=sel[0].lon_deg,
        origin_alt_m=datum,  # -> up_m is AGL
    )

    frame_records, images_out = [], out_root / "images"
    if copy_images:
        images_out.mkdir(parents=True, exist_ok=True)
    for i, (t, p) in enumerate(frames):
        rel = f"images/{p.name}"
        if copy_images:
            shutil.copyfile(p, images_out / p.name)  # byte-for-byte; no re-encode
        frame_records.append(FrameRecord(frame_index=i, timestamp_s=t, image_path=rel))

    agl = [s.alt_m - datum for s in sel]
    dataset_json = build_dataset_json(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        evidence_tier="T3",
        clock_sync=CLOCK_SYNC,
        position_quality_class="rtk_gnss",
        heading_quality_class="unknown",
        height_quality_class=None,
        local_frame_origin={
            "lat_deg": sel[0].lat_deg, "lon_deg": sel[0].lon_deg, "alt_m": datum,
        },
        metadata={
            "flight_id": scene,
            "trajectory_type": "custom",
            "frame_rate_hz": round(
                (len(frames) - 1) / (frames[-1][0] - frames[0][0]), 3
            ) if len(frames) > 1 else 0.0,
            "image_width": CAMERA_INTRINSICS["width"],
            "image_height": CAMERA_INTRINSICS["height"],
            "environment": "aero-model airfield, Yuen Long, Hong Kong; concrete runway, dry grass, tree canopy",
            "nominal_altitude_m": target_agl_m,
            "nominal_speed_ms": nominal_speed_ms,
            "capture_date": "2022-12-21",
            "notes": notes,
            "provenance": provenance,
            "camera_intrinsics": CAMERA_INTRINSICS,
            "cruise_window_utc": [t_start, t_end],
            "ground_datum_m": datum,
            "agl_mean_m": sum(agl) / len(agl),
            "agl_min_m": min(agl),
            "agl_max_m": max(agl),
            "position_quality_caveat": (
                "declared rtk_gnss from the receiver specification published in Li et al. 2024 "
                "(1 cm + 1 ppm x D horizontal, baseline < 5 km). The per-sample RTK fix-quality "
                "and connection-status topics are NOT present in this export and have not been "
                "checked, so the class is PROVISIONAL; if those topics later show a non-fixed "
                "solution the class must be lowered (FR-062)."
            ),
        },
    )
    dataset_json["evidence_caveat"] = (
        "Third-party recorded flight (MARS-LVIG, Li et al. 2024): DJI M300 RTK, Hikvision "
        "CA-050-11UC, 5 mm lens, 80 m nadir. T3 real-sensor data, but NOT of this project's own "
        "system - it characterises the algorithm, never the deployed system. Imagery is JPEG and "
        "auto-exposure was enabled. Flight geometry (0.3 m baseline at 80 m) makes parallax "
        "sub-pixel, which is unusually favourable to a planar-homography estimator."
    )

    write_dataset(out_root, dataset_json, frame_records, gt_rows)
    return {
        "dataset_id": dataset_id, "root": out_root,
        "n_frames": len(frame_records), "n_gt": len(gt_rows),
        "t_start": t_start, "t_end": t_end,
        "duration_s": frames[-1][0] - frames[0][0],
        "ground_datum_m": datum,
        "agl_mean_m": sum(agl) / len(agl),
        "agl_range_m": max(agl) - min(agl),
    }


# ============================================================================================
# Native 10 Hz MCAP source (added 2026-08-14, gap G18) -- what EXP-002 Execution A/B needs.
# ============================================================================================

# HKairport01's own descriptive fields, factored out (2026-08-25, EXP-VO-007) so that
# `build_dataset_from_mcap`'s defaults are named constants rather than literals buried in the
# metadata block. Their values are unchanged from when they were hardcoded, and
# `test_ingest_mars_lvig_mcap.py` pins that.
HKAIRPORT01_ENVIRONMENT = (
    "aero-model airfield, Yuen Long, Hong Kong; concrete runway, dry grass, tree canopy"
)
HKAIRPORT01_CAPTURE_DATE = "2022-12-21"

MARS_LVIG_CAMERA_TOPIC = "/left_camera/image/compressed"
MARS_LVIG_RTK_POSITION_TOPIC = "/dji_osdk_ros/rtk_position"
MARS_LVIG_RTK_YAW_TOPIC = "/dji_osdk_ros/rtk_yaw"
MARS_LVIG_RTK_INFO_POSITION_TOPIC = "/dji_osdk_ros/rtk_info_position"
MARS_LVIG_RTK_INFO_YAW_TOPIC = "/dji_osdk_ros/rtk_info_yaw"
MARS_LVIG_RTK_CONNECTION_TOPIC = "/dji_osdk_ros/rtk_connection_status"

MCAP_REQUIRED_TOPICS = (
    MARS_LVIG_CAMERA_TOPIC, MARS_LVIG_RTK_POSITION_TOPIC, MARS_LVIG_RTK_YAW_TOPIC,
    MARS_LVIG_RTK_INFO_POSITION_TOPIC, MARS_LVIG_RTK_INFO_YAW_TOPIC, MARS_LVIG_RTK_CONNECTION_TOPIC,
)

# DJI Onboard SDK solution-status codes (LIT-006 "Stage 2", verified against DJI's own
# documentation): 50 = "integer narrow-lane ambiguity solution" = RTK fixed, cm-level.
RTK_FIXED_SOLUTION_CODE = 50
RTK_CONNECTED_CODE = 1

# The five RTK-related topics are polled synchronously from the same DJI OSDK broadcast loop at
# 5 Hz (LIT-006), so their log_times coincide far more tightly than this; the tolerance only
# needs to be smaller than half the 0.2 s period to avoid pairing across cycles.
RTK_MERGE_TOLERANCE_NS = int(0.15 * 1e9)


@dataclass(frozen=True)
class McapFrame:
    """One decoded camera message: a GPS-UTC timestamp and the original JPEG bytes, unchanged."""
    timestamp_s: float
    data: bytes


def _fill_local_projected_coords(samples: list[RtkSample]) -> list[RtkSample]:
    """Populate `easting`/`northing` as local ENU metres relative to the first sample, for
    MCAP-sourced samples (which carry only geodetic lat/lon, no projected columns). Used solely
    by this module's own speed/window heuristics -- never written to a dataset output."""
    if not samples:
        return samples
    origin = samples[0]
    out = []
    for s in samples:
        east_m, north_m, _ = geodetic_to_enu(
            s.lat_deg, s.lon_deg, s.alt_m, origin.lat_deg, origin.lon_deg, origin.alt_m,
        )
        out.append(RtkSample(
            scene=s.scene, timestamp_s=s.timestamp_s, lat_deg=s.lat_deg, lon_deg=s.lon_deg,
            alt_m=s.alt_m, easting=east_m, northing=north_m,
            rtk_yaw_raw=s.rtk_yaw_raw, rtk_info_position=s.rtk_info_position,
            rtk_info_yaw=s.rtk_info_yaw, rtk_connection_status=s.rtk_connection_status,
        ))
    return out


def _is_rtk_fixed(sample: RtkSample) -> bool:
    """True only when position AND yaw are both RTK-fixed and the link is connected. A sample
    missing any of these fields (should not happen after `_merge_rtk_topics` drops incomplete
    ones) is conservatively treated as not fixed rather than assumed fixed."""
    return (
        sample.rtk_info_position == RTK_FIXED_SOLUTION_CODE
        and sample.rtk_info_yaw == RTK_FIXED_SOLUTION_CODE
        and sample.rtk_connection_status == RTK_CONNECTED_CODE
    )


def _nearest_payload(msgs: list[tuple[int, bytes]], t_ns: int, tolerance_ns: int) -> Optional[bytes]:
    """`msgs` sorted by log_time. Nearest payload within `tolerance_ns`, or None."""
    if not msgs:
        return None
    times = [m[0] for m in msgs]
    i = bisect.bisect_left(times, t_ns)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(msgs)]
    if not candidates:
        return None
    best = min(candidates, key=lambda j: abs(times[j] - t_ns))
    if abs(times[best] - t_ns) > tolerance_ns:
        return None
    return msgs[best][1]


def _merge_rtk_topics(by_topic_msgs: dict[str, list[tuple[int, bytes]]]) -> list[RtkSample]:
    """Merge the five RTK topics into one `RtkSample` per `rtk_position` message, anchored on
    that topic's log_time. A position message with no partner within tolerance on any of the
    other four is dropped rather than guessed -- an incomplete quality picture must not be
    silently treated as a complete one."""
    out = []
    for log_time, payload in sorted(by_topic_msgs[MARS_LVIG_RTK_POSITION_TOPIC]):
        fix = mcap.dec_navsatfix(payload)
        yaw_p = _nearest_payload(by_topic_msgs[MARS_LVIG_RTK_YAW_TOPIC], log_time, RTK_MERGE_TOLERANCE_NS)
        info_pos_p = _nearest_payload(
            by_topic_msgs[MARS_LVIG_RTK_INFO_POSITION_TOPIC], log_time, RTK_MERGE_TOLERANCE_NS)
        info_yaw_p = _nearest_payload(
            by_topic_msgs[MARS_LVIG_RTK_INFO_YAW_TOPIC], log_time, RTK_MERGE_TOLERANCE_NS)
        conn_p = _nearest_payload(
            by_topic_msgs[MARS_LVIG_RTK_CONNECTION_TOPIC], log_time, RTK_MERGE_TOLERANCE_NS)
        if yaw_p is None or info_pos_p is None or info_yaw_p is None or conn_p is None:
            continue
        out.append(RtkSample(
            scene="", timestamp_s=log_time / 1e9,
            lat_deg=fix["lat"], lon_deg=fix["lon"], alt_m=fix["alt"],
            easting=0.0, northing=0.0,
            rtk_yaw_raw=mcap.dec_int16(yaw_p),
            rtk_info_position=mcap.dec_uint8(info_pos_p),
            rtk_info_yaw=mcap.dec_uint8(info_yaw_p),
            rtk_connection_status=mcap.dec_uint8(conn_p),
        ))
    return _fill_local_projected_coords(out)


def load_hkairport01_mcap_window(
    fetch_range: Callable[[int, int], bytes],
    fetch_tail: Callable[[int], bytes],
    file_size: int,
    t_start_ns: int,
    t_end_ns: int,
) -> tuple[list[RtkSample], list[McapFrame]]:
    """Read merged RTK ground truth and camera frames for `[t_start_ns, t_end_ns]` from the
    HKairport01 MCAP mirror, fetching only the chunks that overlap the window (`LIT-006`
    "Stage 2"; a 180 s Execution-A window costs ~4.7 GB of the 20.4 GB file, not the whole file).

    `fetch_range(start, end_inclusive)` and `fetch_tail(n)` are injected so tests can supply an
    in-memory constructed MCAP file and production code an HTTP range fetcher -- this function
    has no I/O policy of its own (`naveval.mcap_reader`'s own design, reused here unchanged).
    """
    summary_start, _ = mcap.read_footer(fetch_tail)
    if summary_start >= file_size:
        raise IngestError(f"Summary start {summary_start} is not before file_size {file_size}")
    summary_buf = fetch_range(summary_start, file_size - 1)
    channels, chunk_indexes, _stats = mcap.parse_summary(summary_buf)

    by_topic = {c.topic: c for c in channels.values()}
    missing = [t for t in MCAP_REQUIRED_TOPICS if t not in by_topic]
    if missing:
        raise IngestError(f"MCAP is missing required topic(s): {missing}")
    want_ids = {by_topic[t].id: t for t in MCAP_REQUIRED_TOPICS}

    overlapping = mcap.chunks_overlapping(chunk_indexes, t_start_ns, t_end_ns)
    if not overlapping:
        raise IngestError(f"No MCAP chunks overlap [{t_start_ns}, {t_end_ns}] ns")
    lo = min(c.chunk_start_offset for c in overlapping)
    hi = max(c.chunk_start_offset + c.chunk_length for c in overlapping)
    blob = fetch_range(lo, hi - 1)

    by_topic_msgs: dict[str, list[tuple[int, bytes]]] = {t: [] for t in MCAP_REQUIRED_TOPICS}
    for c in overlapping:
        off = c.chunk_start_offset - lo
        opcode = blob[off]
        length = int.from_bytes(blob[off + 1:off + 9], "little")
        if opcode != mcap.OP_CHUNK:
            continue
        inner = mcap.decode_chunk(blob[off + 9:off + 9 + length])
        for cid, log_time, _pub_time, payload in mcap.iter_chunk_messages(inner, set(want_ids)):
            if not (t_start_ns <= log_time <= t_end_ns):
                continue
            by_topic_msgs[want_ids[cid]].append((log_time, payload))

    frames = sorted(
        (
            McapFrame(timestamp_s=log_time / 1e9, data=mcap.dec_compressed_image(payload)["data"])
            for log_time, payload in by_topic_msgs[MARS_LVIG_CAMERA_TOPIC]
        ),
        key=lambda f: f.timestamp_s,
    )
    rtk = _merge_rtk_topics(by_topic_msgs)
    return rtk, frames


def build_dataset_from_mcap(
    out_root: Path | str,
    mcap_track: list[RtkSample],
    mcap_frames: list[McapFrame],
    dataset_id: str,
    dataset_revision: str,
    *,
    scene: str = "HKairport01",
    window: Optional[tuple[float, float]] = None,
    target_agl_m: float = DEFAULT_TARGET_AGL_M,
    provenance: str = "",
    notes: str = "",
    nominal_speed_ms: float = 3.0,
    mcap_source_url: str = "",
    ground_datum_override_m: Optional[float] = None,
    rtk_yaw_to_compass_offset_deg: float = HKAIRPORT01_RTK_YAW_TO_COMPASS_OFFSET_DEG,
    heading_offset_derivation: Optional[str] = None,
    environment: Optional[str] = None,
    capture_date: Optional[str] = None,
    camera_intrinsics: Optional[dict] = None,
    evidence_caveat: Optional[str] = None,
    attitude_caveat: Optional[str] = None,
) -> dict:
    """Write a conforming dataset from native-rate MCAP RTK samples and camera frames
    (`load_hkairport01_mcap_window`).

    Reuses `find_cruise_window`, `ground_datum_m`, `convert_geodetic_track_to_enu`,
    `build_dataset_json` and `write_dataset` exactly as `build_dataset` does -- only the source
    of the track and frames differs. Two differences from `build_dataset`, both because this
    source actually has the fields the other one lacks:

    `ground_datum_override_m` exists because `ground_datum_m` (1st percentile of the altitude
    series) is only a correct ground-level estimate when the supplied track actually touches the
    ground somewhere -- e.g. `build_dataset`'s CSV path, which always loads the whole-flight
    track. A track fetched for a narrow *cruise-only* time window (as EXP-002's Execution A/B
    windows both are, by design -- they start after climb-out) never goes near the ground, so
    `ground_datum_m` on it alone silently returns a value near the cruise altitude itself rather
    than raising. Found while running EXP-002 Execution A on 2026-08-14: the auto-computed datum
    came out ~179.6 m against the whole-sequence value of 99.708 m already established in Stage 1
    (`evaluation/tools/stage1/`), corrupting `dataset.json`'s `agl_*`/`local_frame_origin.alt_m`
    metadata to ~0.1 m AGL instead of ~80 m. (It does NOT corrupt any reported metric:
    `up_m`'s only consumer, `vertical_motion_discarded`, computes range and RMS-about-mean, both
    shift-invariant -- but the metadata is still wrong and must not be produced silently.) Pass
    the already-established whole-sequence datum explicitly when building from a cruise-only
    window; when omitted, falls back to `ground_datum_m(mcap_track)` as before (correct for a
    track that does include low-altitude samples).

    - **Heading IS populated**, via the empirically verified conversion
      `compass_heading_deg = (rtk_yaw_deg - 269.16) mod 360` (`LIT-006` "yaw convention",
      research log 2026-08-14). The formula, its derivation, and its caveats are written into
      `dataset.json`'s `metadata.heading_conversion` so the provenance travels with the data.
    - **Per-sample RTK fix-quality is checked, not assumed.** A sample is written with
      `valid=true` in `groundtruth.csv` only when `rtk_info_position`, `rtk_info_yaw` and
      `rtk_connection_status` all indicate a fixed, connected solution (`_is_rtk_fixed`); a
      non-fixed sample is marked `valid=false` rather than silently included (matching
      `contracts/dataset.md`'s "a `false` row is countable" rule) -- never dropped outright,
      since fix loss is itself a reportable data-quality finding.

    **Sequence-specific fields (added 2026-08-25, `EXP-VO-007`).** Everything above is generic
    across MARS-LVIG: the topic set, the CDR decoders, the RTK merge, the cruise/datum logic and
    the frame writer make no reference to which sequence is being read. Seven values were not:
    the `rtk_yaw` -> compass offset, its derivation text, the environment description, the capture
    date, the camera intrinsics, and the two caveat paragraphs. They are now optional parameters
    **whose defaults are exactly the `HKairport01` values that were previously hardcoded**, so
    `datasets/hkairport01-{a,b}` remain reproducible byte-for-byte by the code that made them
    (`test_hkairport01_defaults_are_the_previously_hardcoded_values`).

    The yaw offset in particular MUST be re-derived for any other sequence rather than inherited:
    `LIT-006` derived it empirically from 52 samples on `HKairport01` and states in the record it
    writes into `dataset.json` that it is sequence-specific. Passing the wrong one is a ~90 deg
    heading error that looks entirely plausible -- the exact failure `DEC-004` exists to prevent.
    """
    out_root = Path(out_root)
    if len(mcap_track) < 2:
        raise IngestError("Fewer than 2 merged RTK samples; cannot fit a cruise window")
    datum = (
        ground_datum_override_m if ground_datum_override_m is not None
        else ground_datum_m(mcap_track)
    )
    t_start, t_end = window if window is not None else find_cruise_window(mcap_track, target_agl_m)

    frames_in_window = sorted(
        (f for f in mcap_frames if t_start <= f.timestamp_s <= t_end), key=lambda f: f.timestamp_s
    )
    if not frames_in_window:
        raise IngestError(f"No frames inside window [{t_start:.3f}, {t_end:.3f}]")

    pad = 1.0
    sel = [s for s in mcap_track if (t_start - pad) <= s.timestamp_s <= (t_end + pad)]
    if len(sel) < 2:
        raise IngestError("Fewer than 2 RTK samples in the window; cannot interpolate")

    n_fixed = sum(1 for s in sel if _is_rtk_fixed(s))
    samples = [
        TelemetrySample(
            timestamp_s=s.timestamp_s, lat_deg=s.lat_deg, lon_deg=s.lon_deg, alt_m=s.alt_m,
            heading_deg=(
                rtk_yaw_to_compass_deg(s.rtk_yaw_raw, rtk_yaw_to_compass_offset_deg)
                if s.rtk_yaw_raw is not None else None
            ),
            fix_quality=(
                f"rtk_info_position={s.rtk_info_position} rtk_info_yaw={s.rtk_info_yaw} "
                f"rtk_connection_status={s.rtk_connection_status}"
            ),
            valid=_is_rtk_fixed(s),
        )
        for s in sel
    ]
    gt_rows = convert_geodetic_track_to_enu(
        samples, origin_lat_deg=sel[0].lat_deg, origin_lon_deg=sel[0].lon_deg,
        origin_alt_m=datum, heading_convention="compass_cw_from_north",
    )

    frame_records, images_out = [], out_root / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames_in_window):
        name = f"{f.timestamp_s:.6f}.jpg"
        (images_out / name).write_bytes(f.data)  # byte-for-byte; no re-encode
        frame_records.append(FrameRecord(frame_index=i, timestamp_s=f.timestamp_s,
                                         image_path=f"images/{name}"))

    measured_hz = (
        round((len(frames_in_window) - 1) / (frames_in_window[-1].timestamp_s
                                              - frames_in_window[0].timestamp_s), 3)
        if len(frames_in_window) > 1 else 0.0
    )
    agl = [s.alt_m - datum for s in sel]
    dataset_json = build_dataset_json(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        evidence_tier="T3",
        clock_sync=CLOCK_SYNC,
        position_quality_class="rtk_gnss",
        heading_quality_class="rtk_gnss",
        height_quality_class=None,
        local_frame_origin={"lat_deg": sel[0].lat_deg, "lon_deg": sel[0].lon_deg, "alt_m": datum},
        metadata={
            "flight_id": scene,
            "trajectory_type": "custom",
            "frame_rate_hz": measured_hz,
            "image_width": (camera_intrinsics or CAMERA_INTRINSICS)["width"],
            "image_height": (camera_intrinsics or CAMERA_INTRINSICS)["height"],
            "environment": environment if environment is not None else HKAIRPORT01_ENVIRONMENT,
            "nominal_altitude_m": target_agl_m,
            "nominal_speed_ms": nominal_speed_ms,
            "capture_date": capture_date if capture_date is not None else HKAIRPORT01_CAPTURE_DATE,
            "notes": notes,
            "provenance": provenance,
            "camera_intrinsics": (
                camera_intrinsics if camera_intrinsics is not None else CAMERA_INTRINSICS
            ),
            "cruise_window_utc": [t_start, t_end],
            "ground_datum_m": datum,
            "agl_mean_m": sum(agl) / len(agl),
            "agl_min_m": min(agl),
            "agl_max_m": max(agl),
            "mcap_source_url": mcap_source_url,
            "rtk_fixed_fraction": n_fixed / len(sel),
            "heading_conversion": {
                "formula": (
                    "compass_heading_deg = (rtk_yaw_deg "
                    f"{rtk_yaw_to_compass_offset_deg:+.2f}) mod 360"
                ),
                "offset_deg": rtk_yaw_to_compass_offset_deg,
                "derivation": heading_offset_derivation if heading_offset_derivation is not None else (
                    "Empirically derived 2026-08-14 from 52 RTK samples across four straight "
                    "legs in four quadrants of HKairport01, cross-checked against the FC "
                    "attitude quaternion (body_FLU -> ENU -> compass), which independently "
                    "matched direction of travel to within 2 deg. Measured "
                    "rtk_yaw - compass(attitude) = +269.16 +/- 0.64 deg (circular mean/sd). "
                    "Likely explanation: the dual-antenna RTK baseline is mounted across the "
                    "airframe rather than along it -- an INFERENCE, not documented by the "
                    "authors. Sequence/source-specific until independently re-verified "
                    "elsewhere; do NOT assume it transfers to other MARS-LVIG sequences or "
                    "platforms without re-measuring."
                ),
                "quantisation_deg": 1.0,  # rtk_yaw is std_msgs/msg/Int16
                "source_topic": MARS_LVIG_RTK_YAW_TOPIC,
                "reference_topic": "/dji_osdk_ros/attitude",
            },
            "position_quality_caveat": (
                f"declared rtk_gnss from the receiver specification (1 cm + 1 ppm x D "
                f"horizontal, Li et al. 2024) AND per-sample fix-quality topics "
                f"(rtk_info_position/rtk_info_yaw == {RTK_FIXED_SOLUTION_CODE}, "
                f"rtk_connection_status == {RTK_CONNECTED_CODE}). {n_fixed}/{len(sel)} RTK "
                f"samples in this window were fixed; any non-fixed sample is marked "
                f"valid=false in groundtruth.csv rather than silently included (FR-062)."
            ),
            "attitude_caveat": attitude_caveat if attitude_caveat is not None else (
                "The evaluated Hikvision camera is rigidly mounted, not gimbal-stabilised "
                "(the gimbal carries the DJI L1). /dji_osdk_ros/attitude showed airframe roll "
                "up to 9.06 deg and pitch up to 6.66 deg on the 2026-08-14 verification "
                "sample -- COMP-001 section 6 assumption 2 (roughly constant attitude) is only "
                "approximately satisfied. EXP-002 threat 14."
            ),
        },
    )
    dataset_json["evidence_caveat"] = evidence_caveat if evidence_caveat is not None else (
        "Third-party recorded flight (MARS-LVIG, Li et al. 2024): DJI M300 RTK, Hikvision "
        "CA-050-11UC, 5 mm lens, 80 m nadir, rigidly mounted (not gimbal-stabilised). T3 "
        "real-sensor data, but NOT of this project's own system - it characterises the "
        "algorithm, never the deployed system. Imagery is JPEG and auto-exposure was enabled. "
        "Flight geometry (0.3 m baseline at 80 m) makes parallax sub-pixel, which is unusually "
        "favourable to a planar-homography estimator. Yaw is converted from the RTK antenna-"
        "baseline heading via an empirically derived, sequence-specific offset -- see "
        "metadata.heading_conversion."
    )

    write_dataset(out_root, dataset_json, frame_records, gt_rows)
    return {
        "dataset_id": dataset_id, "root": out_root,
        "n_frames": len(frame_records), "n_gt": len(gt_rows),
        "t_start": t_start, "t_end": t_end,
        "duration_s": frames_in_window[-1].timestamp_s - frames_in_window[0].timestamp_s,
        "ground_datum_m": datum,
        "agl_mean_m": sum(agl) / len(agl),
        "agl_range_m": max(agl) - min(agl),
        "measured_frame_rate_hz": measured_hz,
        "n_rtk_fixed": n_fixed,
        "n_rtk_total": len(sel),
    }


# ============================================================================================
# Airframe attitude sidecar (added 2026-08-14, EXP-003).
#
# Deliberately SEPARATE from everything above. Three constraints shaped this, all recorded in
# `EXP-003` "Configuration":
#
#   1. `MCAP_REQUIRED_TOPICS` and `load_hkairport01_mcap_window` are NOT modified. They built
#      `datasets/hkairport01-{a,b}`; changing them would mean those datasets could no longer be
#      reproduced by the code that made them. `load_hkairport01_attitude_window` below reads the
#      same chunk range and decodes one additional channel, additively.
#   2. Attitude goes to a sidecar `attitude.csv`, NOT into `groundtruth.csv`, whose seven columns
#      are fixed by `contracts/dataset.md`. Roll/pitch are not ground truth for anything this
#      estimator estimates (`DEC-003` excludes them from the alignment for exactly that reason) --
#      they are a diagnostic covariate, and they do not belong in the ground-truth contract.
#   3. The sidecar stores NATIVE-RATE samples. `EXP-003`'s rate hypothesis (H3) requires
#      differentiating before any resampling; storing a pre-interpolated 10 Hz series would alias
#      the rate signal away.
# ============================================================================================

MARS_LVIG_ATTITUDE_TOPIC = "/dji_osdk_ros/attitude"

ATTITUDE_CSV_NAME = "attitude.csv"
ATTITUDE_CSV_HEADER = ("timestamp_s", "roll_deg", "pitch_deg", "tilt_deg", "yaw_compass_deg")


@dataclass(frozen=True)
class AttitudeSample:
    """One decoded `/dji_osdk_ros/attitude` message, in canonical degrees.

    `yaw_compass_deg` is carried alongside roll/pitch not because `EXP-003` needs it as a
    dependent variable, but as the ingest's own correctness check: it must reproduce the Stage-2
    agreement with RTK-derived direction of travel (`LIT-006`, -1.95 +/- 1.62 deg). A quaternion
    decoded with the wrong convention would still yield plausible-looking roll/pitch, and this
    dataset has already produced one ~90-degree convention trap (`rtk_yaw`) -- so the decode is
    checked against something independent rather than trusted. `EXP-003` gate G2.
    """
    timestamp_s: float
    roll_deg: float
    pitch_deg: float
    tilt_deg: float
    yaw_compass_deg: float


def load_hkairport01_attitude_window(
    fetch_range: Callable[[int, int], bytes],
    fetch_tail: Callable[[int], bytes],
    file_size: int,
    t_start_ns: int,
    t_end_ns: int,
) -> list[AttitudeSample]:
    """Read `/dji_osdk_ros/attitude` (100 Hz) for `[t_start_ns, t_end_ns]` from the HKairport01
    MCAP mirror, fetching only the overlapping chunks.

    Mirrors `load_hkairport01_mcap_window`'s structure and injection pattern exactly, but decodes
    one topic and returns no imagery, so it is cheap in memory even over a long window. The byte
    cost is unfortunately NOT cheap: the chunks are channel-interleaved, so reading attitude for a
    window costs the same download as reading everything else in it. The attitude messages were
    physically downloaded during EXP-002 and discarded unparsed; nothing short of having persisted
    the raw blob then would avoid re-fetching now.
    """
    summary_start, _ = mcap.read_footer(fetch_tail)
    if summary_start >= file_size:
        raise IngestError(f"Summary start {summary_start} is not before file_size {file_size}")
    summary_buf = fetch_range(summary_start, file_size - 1)
    channels, chunk_indexes, _stats = mcap.parse_summary(summary_buf)

    by_topic = {c.topic: c for c in channels.values()}
    if MARS_LVIG_ATTITUDE_TOPIC not in by_topic:
        raise IngestError(f"MCAP is missing required topic: {MARS_LVIG_ATTITUDE_TOPIC}")
    want_id = by_topic[MARS_LVIG_ATTITUDE_TOPIC].id

    overlapping = mcap.chunks_overlapping(chunk_indexes, t_start_ns, t_end_ns)
    if not overlapping:
        raise IngestError(f"No MCAP chunks overlap [{t_start_ns}, {t_end_ns}] ns")
    lo = min(c.chunk_start_offset for c in overlapping)
    hi = max(c.chunk_start_offset + c.chunk_length for c in overlapping)
    blob = fetch_range(lo, hi - 1)

    out: list[AttitudeSample] = []
    for c in overlapping:
        off = c.chunk_start_offset - lo
        opcode = blob[off]
        length = int.from_bytes(blob[off + 1:off + 9], "little")
        if opcode != mcap.OP_CHUNK:
            continue
        inner = mcap.decode_chunk(blob[off + 9:off + 9 + length])
        for _cid, log_time, _pub_time, payload in mcap.iter_chunk_messages(inner, {want_id}):
            if not (t_start_ns <= log_time <= t_end_ns):
                continue
            q = mcap.dec_quaternion_stamped(payload)
            roll, pitch = mcap.quaternion_to_roll_pitch_deg(q["x"], q["y"], q["z"], q["w"])
            out.append(AttitudeSample(
                timestamp_s=log_time / 1e9,
                roll_deg=roll,
                pitch_deg=pitch,
                tilt_deg=mcap.tilt_from_roll_pitch_deg(roll, pitch),
                yaw_compass_deg=mcap.quaternion_to_compass_deg(q["x"], q["y"], q["z"], q["w"]),
            ))
    out.sort(key=lambda s: s.timestamp_s)
    return out


def write_attitude_sidecar(dataset_root: Path | str, samples: Sequence[AttitudeSample]) -> Path:
    """Write `attitude.csv` into an existing dataset directory and return its path.

    Additive: touches no file the dataset contract names, so `naveval.dataset.load_dataset` and
    every contract test are unaffected by its presence. The directory must already exist -- this
    augments a built dataset, it does not create one.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise IngestError(f"Dataset directory does not exist: {root}")
    if not samples:
        raise IngestError("Refusing to write an empty attitude sidecar")
    path = root / ATTITUDE_CSV_NAME
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ATTITUDE_CSV_HEADER)
        for s in samples:
            w.writerow([
                f"{s.timestamp_s:.6f}", f"{s.roll_deg:.4f}", f"{s.pitch_deg:.4f}",
                f"{s.tilt_deg:.4f}", f"{s.yaw_compass_deg:.3f}",
            ])
    return path


def slice_attitude(
    samples: Sequence[AttitudeSample], t_start_s: float, t_end_s: float
) -> list[AttitudeSample]:
    """Inclusive time-slice. `EXP-002`'s Execution-A window is a strict prefix of Execution B's
    (both start at 1671606510.406), so one fetch over B's window supplies both datasets and the
    two series are guaranteed consistent because they are the same samples."""
    return [s for s in samples if t_start_s <= s.timestamp_s <= t_end_s]
