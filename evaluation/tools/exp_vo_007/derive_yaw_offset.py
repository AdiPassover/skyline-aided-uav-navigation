"""Re-derive the `rtk_yaw` -> compass-heading offset for a MARS-LVIG sequence (`EXP-VO-007`).

`/dji_osdk_ros/rtk_yaw` is **not** a compass heading. `LIT-006` "Stage 2" measured
`rtk_yaw - compass(attitude) = +269.16 +/- 0.64 deg` on `HKairport01` -- roughly a quarter turn,
probably because the dual-antenna RTK baseline is mounted across the airframe rather than along it
(an inference, not documented by the authors). Using the field as though it were already a heading
would be wrong by ~90 deg while looking entirely plausible; `DEC-004` exists because that class of
error is otherwise invisible.

The record states in every `dataset.json` it writes that the offset is **sequence-specific until
independently re-verified**. This script does the re-verification, by the same method, for a
different sequence, and it costs ~1 GB rather than a whole-window fetch: only a handful of short
straight-leg windows are read.

Method (`LIT-006`, unchanged):

1. Pick short windows on **straight legs**, from the RTK track alone -- constant course, above a
   speed floor, spread over as many compass quadrants as the flight offers.
2. In each window take `compass(attitude)` from the FC quaternion (`body_FLU` -> ENU -> compass) and
   `course over ground` from consecutive RTK fixes. Their circular mean difference validates the
   quaternion decode independently of any yaw field -- on `HKairport01` it was -1.95 +/- 1.62 deg,
   consistent with crab rather than a convention error.
3. The wanted offset is then the circular mean of `rtk_yaw - compass(attitude)`.

Gate 2 is what makes this a check rather than a fit: if the quaternion decode disagreed with the
direction of travel, no offset derived from it would be trustworthy, and the script says so instead
of reporting a number.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mcap_source as src                                            # noqa: E402
from naveval import mcap_reader as mcap                              # noqa: E402
from naveval.ingest_mars_lvig import (                               # noqa: E402
    load_hkairport01_attitude_window,
    load_hkairport01_mcap_window,
    load_rtk_track,
)

WINDOW_S = 8.0            # long enough for a stable course, short enough to stay straight
MIN_SPEED_MS = 2.0
MAX_TURN_DEG_PER_S = 0.5  # "straight" leg


def _circ_mean_deg(vals: list[float]) -> tuple[float, float]:
    """Circular mean and sd in degrees, for values that may wrap."""
    r = [math.radians(v) for v in vals]
    s, c = sum(math.sin(x) for x in r), sum(math.cos(x) for x in r)
    mean = math.degrees(math.atan2(s, c)) % 360.0
    n = len(vals)
    rbar = math.hypot(s, c) / n
    sd = math.degrees(math.sqrt(-2.0 * math.log(rbar))) if 0 < rbar <= 1 else float("nan")
    return mean, sd


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def straight_legs(track, n_windows: int) -> list[tuple[float, float, float]]:
    """-> [(t_start, t_end, course_deg)] on straight, moving legs, spread over the flight and
    preferring distinct compass quadrants."""
    pts = [(s.timestamp_s, s.easting, s.northing) for s in track]
    cand = []
    for i in range(1, len(pts) - 1):
        (t0, e0, n0), (t1, e1, n1) = pts[i - 1], pts[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        de, dn = e1 - e0, n1 - n0
        if math.hypot(de, dn) / dt < MIN_SPEED_MS:
            continue
        cand.append((pts[i][0], math.degrees(math.atan2(de, dn)) % 360.0))
    legs = []
    i = 0
    while i < len(cand):
        t0, c0 = cand[i]
        j = i
        while j + 1 < len(cand) and cand[j + 1][0] - t0 <= WINDOW_S:
            j += 1
        if j > i:
            span = cand[j][0] - t0
            turn = abs(_wrap180(cand[j][1] - c0)) / max(span, 1e-6)
            if span >= WINDOW_S * 0.6 and turn <= MAX_TURN_DEG_PER_S:
                mean_c, _ = _circ_mean_deg([c for _, c in cand[i:j + 1]])
                legs.append((t0, cand[j][0], mean_c))
                i = j + 1
                continue
        i += 1
    if not legs:
        raise SystemExit("No straight legs found")
    # spread: one leg per quadrant first, then fill by even spacing over the flight
    chosen, used_q = [], set()
    for leg in legs:
        q = int(leg[2] // 90)
        if q not in used_q:
            chosen.append(leg)
            used_q.add(q)
    step = max(1, len(legs) // max(1, n_windows))
    for leg in legs[::step]:
        if len(chosen) >= n_windows:
            break
        if leg not in chosen:
            chosen.append(leg)
    return sorted(chosen)[:n_windows]


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-derive rtk_yaw -> compass offset for a sequence.")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--rtk-csv", required=True, help="UAVScenes rtk_positions_raw.csv, for leg picking")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--out", required=True, help="JSON report path")
    args = ap.parse_args()

    track = load_rtk_track(args.rtk_csv, expect_scene=args.scene)
    legs = straight_legs(track, args.n_windows)
    url = src.mirror_url(args.scene)
    size = src.remote_size(url)
    cache = src.RangeCache(url=url, size=size, cache_dir=Path(args.cache_dir))
    print(f"{args.scene}: {size / 1e9:.2f} GB; {len(legs)} straight legs", flush=True)

    d_att_track, d_yaw_att, per_leg, fetched = [], [], [], 0
    for k, (t0, t1, course) in enumerate(legs, 1):
        a_ns, b_ns = int(t0 * 1e9), int(t1 * 1e9)
        summary_start, _ = mcap.read_footer(cache.fetch_tail)
        _c, chunk_indexes, _s = mcap.parse_summary(cache.fetch_range(summary_start, size - 1))
        over = mcap.chunks_overlapping(chunk_indexes, a_ns, b_ns)
        if not over:
            # A leg picked from the UAVScenes whole-flight RTK track can fall outside the
            # MCAP's chunk coverage (found on AMtown03, EXP-CONF-005); skip it like the
            # no-data case rather than crash.
            print(f"  leg {k}: outside MCAP chunk coverage, skipped", flush=True)
            continue
        lo = min(c.chunk_start_offset for c in over)
        hi = max(c.chunk_start_offset + c.chunk_length for c in over) - 1
        fetched += cache.prefetch(lo, hi)
        rtk, _frames = load_hkairport01_mcap_window(
            cache.fetch_range, cache.fetch_tail, size, a_ns, b_ns)
        att = load_hkairport01_attitude_window(
            cache.fetch_range, cache.fetch_tail, size, a_ns, b_ns)
        cache.release()
        if not att or not rtk:
            print(f"  leg {k}: no data, skipped", flush=True)
            continue
        att_mean, _ = _circ_mean_deg([s.yaw_compass_deg for s in att])
        yaws = [s.rtk_yaw_raw for s in rtk if s.rtk_yaw_raw is not None]
        if not yaws:
            print(f"  leg {k}: no rtk_yaw, skipped", flush=True)
            continue
        yaw_mean, _ = _circ_mean_deg([float(v) for v in yaws])
        d1 = _wrap180(att_mean - course)
        d2 = _wrap180(yaw_mean - att_mean)
        d_att_track.append(d1)
        d_yaw_att.append(d2)
        per_leg.append({"t_start": t0, "t_end": t1, "course_deg": round(course, 2),
                        "attitude_compass_deg": round(att_mean, 2),
                        "rtk_yaw_deg": round(yaw_mean, 2),
                        "attitude_minus_course_deg": round(d1, 2),
                        "rtk_yaw_minus_attitude_deg": round(d2, 2),
                        "n_attitude": len(att), "n_rtk": len(rtk)})
        print(f"  leg {k}: course {course:7.2f}  attitude {att_mean:7.2f}  rtk_yaw {yaw_mean:7.2f}"
              f"   att-course {d1:+7.2f}   yaw-att {d2:+7.2f}", flush=True)

    if len(per_leg) < 3:
        raise SystemExit(f"Only {len(per_leg)} usable legs; refusing to derive an offset")
    g2_mean, g2_sd = _circ_mean_deg(d_att_track)
    g2_mean = _wrap180(g2_mean)
    off_mean, off_sd = _circ_mean_deg(d_yaw_att)
    report = {
        "scene": args.scene, "n_legs": len(per_leg), "legs": per_leg,
        "gate_attitude_vs_course": {"circular_mean_deg": round(g2_mean, 2),
                                    "circular_sd_deg": round(g2_sd, 2)},
        "rtk_yaw_minus_attitude": {"circular_mean_deg": round(off_mean, 2),
                                   "circular_sd_deg": round(off_sd, 2)},
        # naveval applies `compass = (rtk_yaw + offset) mod 360`, so the offset is the negation
        "offset_for_naveval_deg": round(_wrap180(-off_mean), 2),
        "fetched_bytes": fetched,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "legs"}, indent=2))
    if abs(g2_mean) > 10.0 or g2_sd > 10.0:
        print("\nGATE FAILED: compass(attitude) does not track direction of travel; the quaternion "
              "decode cannot be trusted, so neither can any offset derived from it.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
