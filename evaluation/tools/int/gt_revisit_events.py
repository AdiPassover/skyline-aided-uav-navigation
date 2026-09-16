"""Ground-truth revisit / crossing structure of one recording, derived from simulator geometry alone.

Nothing here reads a skyline score, a matcher output or a relocalization decision: an *event* is a
stretch of the trajectory that passes within ``--radius-m`` of a point it visited at least
``--min-elapsed-s`` earlier (the recency horizon a trusted-reference memory would apply). The
result is a description of the recording — where relocalization *could* help — not a result.

Per event: the later pass (frame range, closest-approach frame/time), the earlier pass it revisits
(frame/time at the closest approach, and the range of matched earlier frames), the physical
separation at closest approach (and its mean over the event), the elapsed time, the heading at both
passes, the traversal relation (same direction / reverse / crossing, from the ground-truth velocity
directions), the path length flown by each pass, and — when ``--skyline-profiles`` is given — every
synchronised skyline capture inside the later pass with the true distance to the nearest capture
that is older than the recency horizon (the best *possible* query-to-reference geometry, whatever the
memory policy does). With ``--vo-run`` (a VO_ONLY run's ``alignment_frames.csv``) each pass also
carries the VO's own registered position error at that frame, so an event says how much drift a
correction would have had to work with.

Run from the repository root::

    python evaluation/tools/int/gt_revisit_events.py --dataset datasets/<id> --out <dir> \\
        [--min-elapsed-s 30] [--radius-m 20] [--skyline-profiles datasets/<id>/skyline_profiles.csv] \\
        [--vo-run runs/<vo-only run>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import evaluate_int_arms as ev  # noqa: E402


def load_positions(dataset: Path):
    gt = ev.load_gt(dataset)
    frames = np.array(sorted(gt))
    t = np.array([gt[f][0] for f in frames])
    xy = np.array([[gt[f][1], gt[f][2]] for f in frames])
    return frames, t, xy


def cumulative_path(xy: np.ndarray) -> np.ndarray:
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return np.concatenate([[0.0], np.cumsum(d)])


def velocity_dir(xy: np.ndarray, i: int, half: int = 5) -> np.ndarray:
    a, b = max(0, i - half), min(len(xy) - 1, i + half)
    v = xy[b] - xy[a]
    n = np.hypot(*v)
    return v / n if n > 1e-9 else np.array([np.nan, np.nan])


def nearest_prior_pass(t: np.ndarray, xy: np.ndarray, min_elapsed_s: float):
    """For each frame: the distance to, and index of, the closest frame at least min_elapsed older."""
    n = len(t)
    d_min = np.full(n, np.nan)
    g_min = np.full(n, -1, dtype=int)
    for i in range(n):
        m = t <= t[i] - min_elapsed_s
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        d = np.hypot(xy[idx, 0] - xy[i, 0], xy[idx, 1] - xy[i, 1])
        j = int(np.argmin(d))
        d_min[i] = d[j]
        g_min[i] = idx[j]
    return d_min, g_min


def group_events(frames, t, xy, s, d_min, g_min, radius_m: float, max_gap_frames: int, heading, vo_err):
    inside = np.isfinite(d_min) & (d_min <= radius_m)
    events = []
    i = 0
    n = len(frames)
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        last_in = i
        while j < n and (inside[j] or j - last_in <= max_gap_frames):
            if inside[j]:
                last_in = j
            j += 1
        a, b = i, last_in
        seg = np.arange(a, b + 1)
        k = seg[int(np.argmin(d_min[seg]))]
        g = g_min[k]
        v_later = velocity_dir(xy, k)
        v_earlier = velocity_dir(xy, g)
        cosang = float(np.dot(v_later, v_earlier)) if np.isfinite(v_later).all() and np.isfinite(v_earlier).all() else float("nan")
        relation = ("same_direction" if cosang > 0.7 else "reverse" if cosang < -0.7 else "crossing") if np.isfinite(cosang) else "unknown"
        matched_g = g_min[seg][inside[seg]]
        events.append({
            "event_id": len(events),
            "later_pass": {"frame_start": int(frames[a]), "frame_end": int(frames[b]),
                           "time_start_s": float(t[a]), "time_end_s": float(t[b]),
                           "duration_s": float(t[b] - t[a]), "path_length_at_closest_m": float(s[k])},
            "closest_approach": {"frame": int(frames[k]), "time_s": float(t[k]),
                                 "east_m": float(xy[k, 0]), "north_m": float(xy[k, 1]),
                                 "separation_m": float(d_min[k]),
                                 "heading_deg": None if heading is None else float(heading[k]),
                                 "vo_error_m": None if vo_err is None else _nan_none(vo_err[k])},
            "earlier_pass": {"frame": int(frames[g]), "time_s": float(t[g]),
                             "east_m": float(xy[g, 0]), "north_m": float(xy[g, 1]),
                             "frame_range_matched": [int(frames[int(matched_g.min())]), int(frames[int(matched_g.max())])],
                             "path_length_m": float(s[g]),
                             "heading_deg": None if heading is None else float(heading[g]),
                             "vo_error_m": None if vo_err is None else _nan_none(vo_err[g])},
            "elapsed_s": float(t[k] - t[g]),
            "separation_mean_m": float(np.nanmean(d_min[seg][inside[seg]])),
            "frames_inside_radius": int(inside[seg].sum()),
            "traversal_relation": relation, "direction_cosine": cosang,
            "vo_error_growth_between_passes_m": (None if vo_err is None or not (np.isfinite(vo_err[k]) and np.isfinite(vo_err[g]))
                                                 else float(vo_err[k] - vo_err[g])),
        })
        i = b + 1
    return events, inside


def _nan_none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def load_capture_frames(profiles: Path) -> list[dict]:
    rows = ev.read_csv(profiles)
    out = []
    for r in rows:
        out.append({"frame": int(r["frame_index"]), "t": float(r["timestamp_s"]),
                    "north_valid": r["valid"].strip().lower() == "true",
                    "west_valid": r["west_valid"].strip().lower() == "true"})
    return out


def capture_geometry(events, captures, frames, t, xy, min_elapsed_s: float, margin_s: float):
    index = {int(f): i for i, f in enumerate(frames)}
    cap_idx = [(c, index[c["frame"]]) for c in captures if c["frame"] in index]
    for e in events:
        t0, t1 = e["later_pass"]["time_start_s"] - margin_s, e["later_pass"]["time_end_s"] + margin_s
        rows = []
        for c, i in cap_idx:
            if not (t0 <= c["t"] <= t1):
                continue
            older = [(c2, j) for c2, j in cap_idx if c2["t"] <= c["t"] - min_elapsed_s]
            if older:
                d = [float(np.hypot(*(xy[j] - xy[i]))) for _, j in older]
                k = int(np.argmin(d))
                nearest = {"frame": older[k][0]["frame"], "time_s": older[k][0]["t"], "distance_m": d[k],
                           "elapsed_s": c["t"] - older[k][0]["t"]}
            else:
                nearest = None
            rows.append({"frame": c["frame"], "time_s": c["t"], "north_valid": c["north_valid"],
                         "west_valid": c["west_valid"],
                         "nearest_older_capture": nearest})
        e["skyline_captures_in_later_pass"] = rows
        dists = [r["nearest_older_capture"]["distance_m"] for r in rows if r["nearest_older_capture"]]
        e["best_capture_to_older_capture_m"] = min(dists) if dists else None


def vo_error_curve(dataset: Path, vo_run: Path, frames):
    gt = ev.load_gt(dataset)
    cur = ev.error_curve(gt, ev.read_csv(vo_run / "alignment_frames.csv"))
    by = {int(f): e for f, e in zip(cur["frame"], cur["err"])}
    return np.array([by.get(int(f), np.nan) for f in frames])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-elapsed-s", type=float, default=30.0, help="recency horizon (default: the policy's 30 s)")
    ap.add_argument("--radius-m", type=float, default=20.0, help="revisit radius (default: the genuine-revisit 20 m)")
    ap.add_argument("--max-gap-frames", type=int, default=10, help="frames outside the radius that still join one event")
    ap.add_argument("--skyline-profiles", type=Path, default=None)
    ap.add_argument("--capture-margin-s", type=float, default=2.0)
    ap.add_argument("--vo-run", type=Path, default=None, help="a VO_ONLY run: adds the VO's own error at each pass")
    a = ap.parse_args(argv)

    frames, t, xy = load_positions(a.dataset)
    s = cumulative_path(xy)
    heading = None
    gt_rows = ev.read_csv(a.dataset / "groundtruth.csv")
    if gt_rows and "heading_deg" in gt_rows[0]:
        tt = np.array([float(r["timestamp_s"]) for r in gt_rows]); hh = np.array([float(r["heading_deg"]) for r in gt_rows])
        heading = np.array([hh[int(np.argmin(np.abs(tt - tk)))] for tk in t])
    vo_err = vo_error_curve(a.dataset, a.vo_run, frames) if a.vo_run else None

    d_min, g_min = nearest_prior_pass(t, xy, a.min_elapsed_s)
    events, inside = group_events(frames, t, xy, s, d_min, g_min, a.radius_m, a.max_gap_frames, heading, vo_err)
    if a.skyline_profiles:
        capture_geometry(events, load_capture_frames(a.skyline_profiles), frames, t, xy, a.min_elapsed_s, a.capture_margin_s)

    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "nearest_prior_pass.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "east_m", "north_m", "path_length_m", "nearest_prior_frame", "nearest_prior_time_s",
                    "separation_m", "elapsed_s", "inside_radius", "vo_error_m"])
        for i in range(len(frames)):
            g = g_min[i]
            w.writerow([int(frames[i]), f"{t[i]:.3f}", f"{xy[i,0]:.3f}", f"{xy[i,1]:.3f}", f"{s[i]:.2f}",
                        "" if g < 0 else int(frames[g]), "" if g < 0 else f"{t[g]:.3f}",
                        "" if g < 0 else f"{d_min[i]:.3f}", "" if g < 0 else f"{t[i]-t[g]:.1f}",
                        str(bool(inside[i])).lower(), "" if vo_err is None or math.isnan(vo_err[i]) else f"{vo_err[i]:.3f}"])
    with (a.out / "gt_revisit_events.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "later_frame_start", "later_frame_end", "closest_frame", "closest_time_s", "earlier_frame",
                    "earlier_time_s", "elapsed_s", "separation_m", "separation_mean_m", "frames_inside", "relation",
                    "path_length_later_m", "path_length_earlier_m", "vo_error_later_m", "vo_error_earlier_m",
                    "vo_error_growth_m", "captures_in_later_pass", "best_capture_to_older_capture_m"])
        for e in events:
            w.writerow([e["event_id"], e["later_pass"]["frame_start"], e["later_pass"]["frame_end"],
                        e["closest_approach"]["frame"], f"{e['closest_approach']['time_s']:.1f}", e["earlier_pass"]["frame"],
                        f"{e['earlier_pass']['time_s']:.1f}", f"{e['elapsed_s']:.1f}", f"{e['closest_approach']['separation_m']:.2f}",
                        f"{e['separation_mean_m']:.2f}", e["frames_inside_radius"], e["traversal_relation"],
                        f"{e['later_pass']['path_length_at_closest_m']:.1f}", f"{e['earlier_pass']['path_length_m']:.1f}",
                        _fmt(e["closest_approach"]["vo_error_m"]), _fmt(e["earlier_pass"]["vo_error_m"]),
                        _fmt(e["vo_error_growth_between_passes_m"]),
                        len(e.get("skyline_captures_in_later_pass", [])), _fmt(e.get("best_capture_to_older_capture_m"))])
    summary = {
        "dataset": str(a.dataset), "definition": {"min_elapsed_s": a.min_elapsed_s, "radius_m": a.radius_m,
                                                   "max_gap_frames": a.max_gap_frames,
                                                   "source": "simulator ground truth only; no skyline score, matcher or relocalization decision"},
        "n_frames": int(len(frames)), "duration_s": float(t[-1] - t[0]), "gt_path_length_m": float(s[-1]),
        "gt_start_end_distance_m": float(np.hypot(*(xy[-1] - xy[0]))),
        "frames_inside_radius": int(inside.sum()), "fraction_of_frames_inside_radius": float(inside.mean()),
        "n_events": len(events), "events": events,
    }
    (a.out / "gt_revisit_events.json").write_text(json.dumps(summary, indent=2, default=ev._json_default) + "\n")
    print(f"{a.dataset.name}: {len(frames)} frames, {s[-1]:.0f} m, {len(events)} revisit events "
          f"(radius {a.radius_m:g} m, elapsed >= {a.min_elapsed_s:g} s), {inside.sum()} frames inside")
    for e in events:
        print(f"  E{e['event_id']}: frames {e['later_pass']['frame_start']}-{e['later_pass']['frame_end']} "
              f"({e['closest_approach']['time_s']:.1f} s) revisit frame {e['earlier_pass']['frame']} ({e['earlier_pass']['time_s']:.1f} s), "
              f"sep {e['closest_approach']['separation_m']:.1f} m, elapsed {e['elapsed_s']:.0f} s, {e['traversal_relation']}, "
              f"VO err {_fmt(e['earlier_pass']['vo_error_m'])} -> {_fmt(e['closest_approach']['vo_error_m'])} m, "
              f"captures {len(e.get('skyline_captures_in_later_pass', []))}, best cap->older cap {_fmt(e.get('best_capture_to_older_capture_m'))} m")
    return 0


def _fmt(x):
    return "" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
