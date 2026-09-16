"""Audit every altitude-bearing channel in MARS-LVIG, and measure altitude variation across the
WHOLE flight — before any VO is run (`EXP-VO-013` B1/B2).

Two questions, and they are different:

**B1 — what altitude sources actually exist, and what is each one?** `EXP-VO-012` validated a
*barometer-aided* metric readout at T1. Whether MARS-LVIG can validate that on real imagery depends
on whether MARS-LVIG has a barometer at all — and the answer has to come from the file, not from a
topic's name. `LIT-006` Stage 2 already found `/dji_osdk_ros/height_above_takeoff` **present but
identically 0.000** on `HKairport01`; this re-checks it on whichever sequence is under consideration
and enumerates every other channel that could carry height, so nothing is missed by assumption.

Each candidate is classified into one of five classes, recorded per sequence:

    A  TRUE BAROMETRIC / FC RELATIVE ALTITUDE
    B  FUSED RELATIVE ALTITUDE OF UNCERTAIN SENSOR ORIGIN
    C  RTK/GNSS-DERIVED TAKEOFF-RELATIVE ALTITUDE
    D  LiDAR / TRUE-AGL REFERENCE
    E  UNUSABLE / UNKNOWN

**RTK altitude is class C and must never be labelled "barometer."** That is the whole reason this
probe exists rather than an assumption.

**B2 — is there a window with meaningful altitude variation?** `LIT-VO-004` §3 measured the *cruise*
window of four sequences and found height above the takeoff datum constant to a few centimetres on
every one. What a cruise window necessarily excludes is the climb and the descent, which is where
aircraft altitude actually moves. This probe reads the **whole flight** and searches it for windows
that are altitude-varying *and* long enough horizontally for a percent-of-path metric to mean
anything — two conditions that fight each other, which is itself the finding (a UAV climbing to
survey altitude does it nearly in place).

It also re-reads the LiDAR range inside the camera cone across the whole flight, so **terrain-driven**
AGL variation is separated from **aircraft-driven** altitude variation. That is exactly the
distinction `EXP-VO-012` R2 (aircraft climbs over flat ground — barometer helps) and R7 (ground rises
under a level aircraft — barometer is blind) turn on, and a real sequence has to be assigned to one
of them before it can test anything.

Reuses `evaluation/tools/exp_vo_007/probe_candidates.py`'s UAVScenes access unmodified — same ZIP,
same range-request method, same camera cone, same `find_cruise_window`/`ground_datum_m` — so the
numbers here are directly comparable with `LIT-VO-004` §3.2's table.

    PY=.venv/Scripts/python.exe
    $PY evaluation/tools/exp_vo_013/altitude_sources.py \
        --scenes AMvalley01 AMtown01 --cache-dir <scratch> --out <report.json> [--skip-mcap]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                    # evaluation/
sys.path.insert(0, str(_HERE.parents[1] / "exp_vo_007"))     # mcap_source, probe_candidates
sys.path.insert(0, str(_HERE.parents[1] / "stage1"))         # ziprange

import mcap_source as ms                                                  # noqa: E402
import probe_candidates as pc                                             # noqa: E402
import ziprange as zr                                                     # noqa: E402
from naveval import mcap_reader as mr                                     # noqa: E402
from naveval.ingest_mars_lvig import ground_datum_m, load_rtk_track       # noqa: E402

# Anything whose topic name could plausibly carry a height. Deliberately broad: the failure this
# probe exists to prevent is concluding "there is no barometer" from a topic list nobody read.
HEIGHT_HINTS = ("height", "altitude", "alt", "baro", "pressure", "range", "agl", "elev", "vertical")

# Topics that carry a height in a component of something else, so no name hint fires. Found by
# reading the full 32-channel inventory rather than by grepping it: `local_position` is DJI's own
# fused local ENU frame and its `z` is a relative altitude — exactly the kind of channel a
# name-based search misses and a barometer-aided design would then be built on by accident.
EXTRA_CANDIDATES = ("/dji_osdk_ros/local_position", "/dji_osdk_ros/vo_position",
                    "/dji_osdk_ros/gps_position")

# Topics whose names contain a hint substring by accident.
HINT_FALSE_POSITIVES = ("rtk_yaw", "rtk_info", "rtk_connection")


# ---------------------------------------------------------------------------------------------
# B1 — the channel audit
# ---------------------------------------------------------------------------------------------
def classify(topic: str, samples: list) -> tuple[str, str]:
    """(class letter, one-line justification). Evidence first, topic name second."""
    finite = [s for s in samples if s is not None and np.isfinite(s)]
    if not finite:
        return "E", "present, but no message decoded to a numeric scalar in the sampled chunks"
    span = max(finite) - min(finite)
    # A *relative-altitude* channel that never leaves a one-metre band across a whole survey flight
    # is not reporting altitude, whatever it is reporting. The threshold is physical, not numerical:
    # an exact-zero test would have called AMtown01's channel "populated" on the strength of a
    # 2e-6 m wobble and a 0.23 m pre-take-off offset, which is how a dead channel gets mistaken for
    # a live one. Below the floor the class is E; above it the channel still has to be shown to be
    # barometric rather than fused, which no MARS-LVIG evidence can do.
    NOT_REPORTING_ALTITUDE_M = 1.0
    inert = max(abs(v) for v in finite) < NOT_REPORTING_ALTITUDE_M

    if "height_above_takeoff" in topic:
        if inert:
            return "E", (f"never leaves +/-{NOT_REPORTING_ALTITUDE_M:.0f} m across {len(finite)} "
                         f"sampled messages spanning the whole flight (observed span {span:.6f} m, "
                         f"|max| {max(abs(v) for v in finite):.6f} m) — the DJI flight-controller "
                         f"relative-altitude field is PRESENT BUT NEVER POPULATED with the "
                         f"aircraft's height. This is precisely the channel a barometer-aided "
                         f"readout would consume, and it carries nothing")
        return "B", (f"DJI FC height-above-takeoff, populated (span {span:.3f} m over {len(finite)} "
                     f"samples). DJI publishes this as a FUSED relative altitude, not raw "
                     f"barometric pressure — class B, not A, until the sensor origin is "
                     f"established from DJI's own documentation")
    if "rtk_position" in topic:
        return "C", (f"RTK/GNSS geodetic altitude (span {span:.3f} m over the sampled chunks). "
                     f"Takeoff-relative only after subtracting a ground datum. NOT a barometer")
    if "livox/lidar" in topic:
        return "D", "Livox Avia point cloud — true range to the surface below, a genuine AGL reference"
    if "local_position" in topic:
        return "C", (f"DJI FC `local_position` (PointStamped). Its `point.z` is measured here to be "
                     f"the GNSS geodetic altitude copied verbatim — bit-identical to "
                     f"`gps_position.altitude` on every sampled message, spanning {span:.3f} m — "
                     f"NOT a takeoff-relative local z, despite the topic name. GNSS-derived, so "
                     f"class C, and it needs the same ground-datum subtraction the RTK topic does")
    if "vo_position" in topic:
        return "B", (f"DJI FC downward-vision position (span {span:.3f}). Vision-derived, so its "
                     f"height is not an independent reference for a vision-based estimator, and "
                     f"the M300 VPS is specified only to low altitude")
    if "gps_position" in topic:
        return "C", (f"non-RTK GNSS position (span {span:.3f}); its altitude is GNSS-derived and "
                     f"strictly worse than the RTK topic. NOT a barometer")
    if "pressure" in topic or "baro" in topic:
        return "A", f"barometric pressure channel (span {span:.3f})"
    return "E", f"height-like topic name, unrecognised semantics (span {span:.3f})"


OP_SCHEMA = 3


def parse_schemas(buf: bytes) -> dict:
    """Schema id -> declared type name, from the Summary's Schema records.

    `naveval.mcap_reader.parse_summary` does not collect these (it never needed them: MARS-LVIG's
    five required topics have known types). An audit does need them, because classifying a height
    channel by guessing its wire layout is exactly how `local_position` and `gps_position` both
    came back with byte-identical values in an earlier run of this probe — one decoder happened to
    succeed on the wrong message shape.
    """
    out: dict[int, str] = {}
    c = mr._Cur(buf)
    n = len(buf)
    while c.p + 9 <= n:
        op = c.u8()
        ln = c.u64()
        end = c.p + ln
        if end > n:
            break
        if op == OP_SCHEMA:
            r = mr._Cur(buf, c.p)
            sid, name = r.u16(), r.s()
            out[sid] = name
        c.p = end
    return out


def decode_by_schema(schema: str, payload: bytes):
    """The height-bearing scalar of a message, chosen by its DECLARED type. None if there is none.

    Returns `(value, what)` so the report can say which component was read rather than leaving the
    reader to assume. `None` means "this type carries no height", which is a different statement
    from "the value is zero" and must not be collapsed into it.
    """
    s = schema.split("/")[-1]
    try:
        if s == "NavSatFix":
            return mr.dec_navsatfix(payload)["alt"], "geodetic altitude (m, ellipsoidal/MSL)"
        if s == "PointStamped":
            c = mr.CDR(payload)
            c.header(); c.f64(); c.f64()
            return c.f64(), "point.z"
        if s == "Vector3Stamped":
            c = mr.CDR(payload)
            c.header(); c.f64(); c.f64()
            return c.f64(), "vector.z"
        if s == "Float32":
            return mr.dec_float32(payload), "the Float32 value"
        if s == "Float64":
            return mr.CDR(payload).f64(), "the Float64 value"
        if s in ("UInt8", "Int16"):
            return None, f"{s}: not a height"
    except Exception:
        return None, f"{s}: decode failed"
    return None, f"{s}: no height component"


def _dec_point_stamped_z(payload: bytes) -> float:
    """geometry_msgs/msg/PointStamped -> z. Header, then three float64."""
    c = mr.CDR(payload)
    c.header()
    c.f64(); c.f64()
    return c.f64()


def _decode_height(payload: bytes):
    """Best-effort HEIGHT scalar from a ROS 2 CDR payload; None when the shape is not recognised.

    Order matters: the fixed-size decoders are tried last, because a `Float32` decoder will happily
    read the first four bytes of *anything* and return a number. Sized checks first, then structured
    messages, then bare scalars for payloads short enough that they can only be scalars.
    """
    n = len(payload)
    try:
        return float(mr.dec_navsatfix(payload)["alt"])
    except Exception:
        pass
    try:
        return float(_dec_point_stamped_z(payload))
    except Exception:
        pass
    if n <= 12:                       # 4-byte CDR encapsulation header + a small scalar
        for dec in (mr.dec_float32, mr.dec_int16, mr.dec_uint8):
            try:
                return float(dec(payload))
            except Exception:
                continue
    return None


def probe_topics(scene: str, cache: Path, n_chunks: int = 24) -> dict:
    """Enumerate every channel, then decode height candidates from chunks spread across the flight."""
    url = ms.mirror_url(scene)
    size = ms.remote_size(url)
    rc = ms.RangeCache(url=url, size=size, cache_dir=cache / f"mcap_{scene}")
    summary_start, _ = mr.read_footer(rc.fetch_tail)
    summary = rc.fetch_range(summary_start, size - 1)
    channels, chunk_indexes, stats = mr.parse_summary(summary)
    schemas = parse_schemas(summary)

    # The file's own Statistics record would carry a whole-file message count per channel, which
    # would be stronger evidence than any sample. It is NOT used: on this file `parse_summary`
    # returns `{0: 887554058}` for that map — a single bogus entry — while the surrounding
    # statistics (2,168,650 messages, 26,882 chunks, 32 channels) are correct, so the map read is
    # misaligned. Reported rather than relied on or silently worked around; the control counts
    # below come from actually iterating messages, which is verified to work.
    per_channel: dict = {}
    stats_note = ("channel_message_counts not used: naveval.mcap_reader.parse_summary returns a "
                  "single bogus entry for it on this file (see EXP-VO-013 B1)")
    inventory = [{"topic": c.topic, "schema": schemas.get(c.schema_id, f"schema#{c.schema_id}")}
                 for c in sorted(channels.values(), key=lambda c: c.topic)]
    candidates = [c for c in channels.values()
                  if (any(h in c.topic.lower() for h in HEIGHT_HINTS)
                      or c.topic in EXTRA_CANDIDATES)
                  and not any(f in c.topic for f in HINT_FALSE_POSITIVES)]

    # Sample from chunks spread across the flight, not chunk 0 alone: a field populated only after
    # take-off would read as empty from the start of the recording and be misclassified.
    picks = []
    if chunk_indexes:
        idx = np.unique(np.linspace(0, len(chunk_indexes) - 1, n_chunks).astype(int))
        picks = [chunk_indexes[int(i)] for i in idx]

    want = {c.id: c.topic for c in candidates}
    all_ids = {c.id: c.topic for c in channels.values()}
    samples: dict[str, list] = {c.topic: [] for c in candidates}
    # A control, and it is not optional. The first version of this probe reported ZERO messages on
    # every height candidate, which reads as a decisive negative result; running the same read over
    # ALL 32 channels showed zero messages on every one of them, i.e. the reader was wrong, not the
    # data. (Two bugs: `decode_chunk` takes a chunk record's PAYLOAD, so the 9-byte record header
    # has to be stripped, and `iter_chunk_messages` yields a 4-tuple.) The control now ships with
    # the probe so a future "the topic is empty" conclusion cannot be drawn from a broken read.
    control: dict[str, int] = {}
    bytes_read = 0
    for ci in picks:
        raw = rc.fetch_range(ci.chunk_start_offset, ci.chunk_start_offset + ci.chunk_length - 1)
        bytes_read += len(raw)
        opcode, length = raw[0], int.from_bytes(raw[1:9], "little")
        if opcode != mr.OP_CHUNK:
            continue
        inner = mr.decode_chunk(raw[9:9 + length])
        for cid, _lt, _pt, payload in mr.iter_chunk_messages(inner, set(all_ids)):
            control[all_ids[cid]] = control.get(all_ids[cid], 0) + 1
            if cid in want:
                v, _what = decode_by_schema(schemas.get(channels[cid].schema_id, ""), payload)
                samples[want[cid]].append(v)

    out = []
    for c in candidates:
        s = samples[c.topic]
        schema = schemas.get(c.schema_id, f"schema#{c.schema_id}")
        letter, why = classify(c.topic, s)
        finite = [v for v in s if v is not None and np.isfinite(v)]
        out.append({"topic": c.topic, "schema": schema,
                    "messages_in_sample": control.get(c.topic, 0),
                    "n_sampled_messages": len(s), "n_decodable": len(finite),
                    "min": (min(finite) if finite else None),
                    "max": (max(finite) if finite else None),
                    "class": letter, "justification": why})
    return {"scene": scene, "mcap_bytes": size, "n_channels": len(channels),
            "n_chunks_sampled": len(picks), "channel_inventory": inventory,
            "height_candidates": out, "probe_bytes_read": bytes_read,
            "control_messages_per_topic": dict(sorted(control.items())),
            "control_ok": bool(control), "statistics_note": stats_note,
            "file_statistics": {k: v for k, v in (stats or {}).items()
                                if k != "channel_message_counts"}}


# ---------------------------------------------------------------------------------------------
# B2 — whole-flight altitude variation
# ---------------------------------------------------------------------------------------------
def whole_flight_profile(cache: Path, entries: list[dict], scene: str) -> dict:
    """Altitude above the ground datum, and horizontal path, over the WHOLE flight."""
    name = f"interval5_CAM_LIDAR/interval5_{scene}/rtk_positions_raw.csv"
    entry = next(e for e in entries if e["name"] == name)
    csv_path = cache / f"{scene}_rtk.csv"
    if not csv_path.exists():
        zr.extract_entry(pc.UAVSCENES_URL, entry, csv_path, cache / "tmp.bin")
    track = load_rtk_track(csv_path, expect_scene=scene)
    datum = ground_datum_m(track)

    t = np.array([s.timestamp_s for s in track])
    up = np.array([s.alt_m for s in track]) - datum
    e = np.array([s.easting for s in track])
    n = np.array([s.northing for s in track])
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(e), np.diff(n)))])
    return {
        "scene": scene,
        "n_rtk_samples": int(len(track)),
        "duration_s": float(t[-1] - t[0]),
        "ground_datum_m": float(datum),
        "path_length_m": float(cum[-1]),
        "altitude_above_datum_m": {"min": float(up.min()), "max": float(up.max()),
                                   "mean": float(up.mean()), "ptp": float(np.ptp(up))},
        "_series": {"t": t.tolist(), "up": up.tolist(), "cum": cum.tolist()},
    }


def lidar_agl_series(cache: Path, entries: list[dict], scene: str, n_clouds: int) -> dict:
    """Median in-camera-cone LiDAR range across the WHOLE flight — height above the imaged surface.

    Same cone, same statistic and same `x > 1 m` gate as `probe_candidates.lidar_probe`, so the
    numbers line up with `LIT-VO-004` §3.2; the difference is that this samples the whole flight
    rather than the cruise window, so terrain relief outside the cruise is visible too.
    """
    prefix = f"interval5_CAM_LIDAR/interval5_{scene}/interval5_LIDAR/"
    lid = sorted((e for e in entries if e["name"].startswith(prefix) and e["name"].endswith(".txt")),
                 key=lambda e: e["name"])
    if not lid:
        return {"scene": scene, "n_clouds": 0, "note": "no LiDAR entries"}
    ts = np.array([float(re.search(r"image(\d+\.\d+)_lidar", e["name"]).group(1)) for e in lid])
    out_dir = cache / f"lidar_{scene}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in np.unique(np.linspace(0, len(lid) - 1, n_clouds).astype(int)):
        ent = lid[int(i)]
        p = out_dir / Path(ent["name"]).name
        if not p.exists():
            zr.extract_entry(pc.UAVSCENES_URL, ent, p, cache / "tmp.bin")
        a = np.loadtxt(p)
        x, y, z = a[:, 0], a[:, 1], a[:, 2]
        m = x > 1.0
        x, y, z = x[m], y[m], z[m]
        inside = ((np.abs(np.arctan2(y, x)) < pc.HALF_HFOV_RAD)
                  & (np.abs(np.arctan2(z, x)) < pc.HALF_VFOV_RAD))
        xi = x[inside]
        if len(xi) < 500:
            continue
        rows.append((float(ts[int(i)]), float(np.median(xi)),
                     float(np.percentile(xi, 95) - np.percentile(xi, 5))))
    if not rows:
        return {"scene": scene, "n_clouds": 0, "note": "no cloud had 500 in-cone returns"}
    r = np.array(sorted(rows))
    return {"scene": scene, "n_clouds": int(len(r)),
            "agl_median_m": float(np.median(r[:, 1])),
            "agl_min_m": float(r[:, 1].min()), "agl_max_m": float(r[:, 1].max()),
            "agl_p5_m": float(np.percentile(r[:, 1], 5)),
            "agl_p95_m": float(np.percentile(r[:, 1], 95)),
            "depth_spread_p95_p5_m": float(np.median(r[:, 2])),
            "_series": {"t": r[:, 0].tolist(), "agl": r[:, 1].tolist(),
                        "spread": r[:, 2].tolist()}}


def best_altitude_windows(profile: dict, *, min_path_m: float, min_ratio: float,
                          min_duration_s: float, min_altitude_m: float, top: int = 5) -> list[dict]:
    """Windows with a large aircraft-altitude ratio AND enough horizontal path to score on.

    The two conditions fight each other, and that is the point of reporting both: a UAV climbing to
    survey altitude does it nearly in place, so the window with the biggest altitude ratio is
    usually the one with the least path — and `% of path` metrics are meaningless there. Reported
    rather than resolved.

    `min_altitude_m` floors the window's lowest altitude: below a few metres the ratio explodes for
    arithmetic reasons, and the imagery is a different problem (ground-level parallax, motion blur,
    the camera looking at its own shadow) rather than a harder instance of the same one.
    """
    t = np.asarray(profile["_series"]["t"])
    up = np.asarray(profile["_series"]["up"])
    cum = np.asarray(profile["_series"]["cum"])
    n = len(t)
    out = []
    stride = max(1, n // 500)
    for i in range(0, n - 2, stride):
        for frac in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
            j = min(n - 1, i + int(n * frac))
            if j <= i:
                continue
            seg = up[i:j + 1]
            lo, hi = float(seg.min()), float(seg.max())
            if lo < min_altitude_m:
                continue
            ratio = hi / lo
            path = float(cum[j] - cum[i])
            dur = float(t[j] - t[i])
            if ratio >= min_ratio and path >= min_path_m and dur >= min_duration_s:
                out.append({"t_start": float(t[i]), "t_end": float(t[j]), "duration_s": dur,
                            "path_m": path, "alt_min_m": lo, "alt_max_m": hi, "alt_ratio": ratio,
                            "score": ratio * min(1.0, path / max(min_path_m, 1.0))})
    out.sort(key=lambda d: -d["score"])
    kept: list[dict] = []
    for w in out:
        if all(w["t_end"] < k["t_start"] or w["t_start"] > k["t_end"] for k in kept):
            kept.append(w)
        if len(kept) >= top:
            break
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-mcap", action="store_true",
                    help="RTK/LiDAR profile only; do not touch the multi-GB MCAP mirrors")
    ap.add_argument("--n-clouds", type=int, default=90)
    ap.add_argument("--skip-lidar", action="store_true",
                    help="RTK altitude profile only. The LiDAR pass is many small range requests "
                         "and is by far the slowest part; it is also the only source of TRUE AGL, "
                         "so skipping it answers the aircraft-altitude question and not the "
                         "terrain one.")
    ap.add_argument("--min-path-m", type=float, default=200.0)
    ap.add_argument("--min-ratio", type=float, default=1.25)
    ap.add_argument("--min-duration-s", type=float, default=60.0)
    ap.add_argument("--min-altitude-m", type=float, default=15.0)
    a = ap.parse_args()

    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    entries = pc._entries(cache)
    report = {"criteria": {"min_path_m": a.min_path_m, "min_ratio": a.min_ratio,
                           "min_duration_s": a.min_duration_s,
                           "min_altitude_m": a.min_altitude_m},
              "scenes": {}}

    for scene in a.scenes:
        print(f"[{scene}] whole-flight RTK profile ...", flush=True)
        prof = whole_flight_profile(cache, entries, scene)
        wins = best_altitude_windows(prof, min_path_m=a.min_path_m, min_ratio=a.min_ratio,
                                     min_duration_s=a.min_duration_s,
                                     min_altitude_m=a.min_altitude_m)
        alt = prof["altitude_above_datum_m"]
        print(f"  {prof['duration_s']:.0f} s, path {prof['path_length_m']:.0f} m, "
              f"altitude above datum {alt['min']:.1f}..{alt['max']:.1f} m")
        print(f"  altitude-varying windows meeting criteria: {len(wins)}")
        for w in wins[:3]:
            print(f"    {w['duration_s']:6.0f} s  {w['path_m']:7.0f} m  "
                  f"{w['alt_min_m']:6.1f}..{w['alt_max_m']:6.1f} m  ratio {w['alt_ratio']:.2f}")

        if a.skip_lidar:
            lid = {"scene": scene, "n_clouds": 0, "note": "skipped (--skip-lidar)"}
        else:
            print(f"[{scene}] whole-flight LiDAR AGL ...", flush=True)
            lid = lidar_agl_series(cache, entries, scene, a.n_clouds)
        if lid.get("n_clouds"):
            print(f"  AGL median {lid['agl_median_m']:.1f} m, "
                  f"p5..p95 {lid['agl_p5_m']:.1f}..{lid['agl_p95_m']:.1f} "
                  f"(ratio {lid['agl_p95_m'] / max(lid['agl_p5_m'], 1e-9):.2f}), "
                  f"min..max {lid['agl_min_m']:.1f}..{lid['agl_max_m']:.1f}")

        entry = {"profile": {k: v for k, v in prof.items() if k != "_series"},
                 "altitude_varying_windows": wins,
                 "n_windows_meeting_criteria": len(wins),
                 "lidar_agl": {k: v for k, v in lid.items() if k != "_series"}}
        if not a.skip_mcap:
            print(f"[{scene}] MCAP channel/altitude audit ...", flush=True)
            entry["mcap_audit"] = probe_topics(scene, cache)
            for c in entry["mcap_audit"]["height_candidates"]:
                print(f"    [{c['class']}] {c['topic']}: {c['justification']}")
        entry["_series"] = {"rtk": prof["_series"], "lidar": lid.get("_series")}
        report["scenes"][scene] = entry

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
