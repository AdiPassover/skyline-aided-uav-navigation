"""Is the nadir imagery of a recording trackable? — the VO's own health, from a VO_ONLY run's artifacts.

The operator's worry ("the ground looks featureless") is answered from what the tracker did, not
from a screenshot: per-frame success, feature-track and inlier counts (minimum and distribution),
inlier ratio, recenters versus restarts (a recenter is mosaic maintenance and leaves the metric
track continuous — `RecenterVersusRestartTest`; a restart is a hard loss), translation dropouts,
metric-frame usability, the height and heading channel statuses, the visual-yaw disagreement
diagnostic, and the ground-truth error of the VO's own persistent position (frame-0 translation
registration, as in `evaluate_int_arms.py`): the curve at 10 s marks, its final value, the
start-end closure of GT and of the VO, the estimated-to-true path-length ratio (a scale error would
show here), and the per-frame incremental error split into a systematic mean vector and a random
part. Drift is not tracking failure: for the INT milestone a stable tracker *with* moderate drift is
the desirable case.

The verdict fields are computed by a declared rule (below) so they can be argued with; the author's
GOOD / MARGINAL / BAD call is made from the numbers, not by this script alone.

Run from the repository root::

    python evaluation/tools/int/vo_quality_report.py --dataset datasets/<id> --run runs/<vo-only run> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import evaluate_int_arms as ev  # noqa: E402


def q(v, ps=(0, 1, 5, 25, 50, 75, 95, 100)):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {f"p{p}": float(np.percentile(v, p)) for p in ps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mark-interval-s", type=float, default=10.0)
    a = ap.parse_args(argv)

    manifest = json.loads((a.run / "manifest.json").read_text())
    est = manifest.get("estimator_config", {})
    abs_min = int(est.get("absoluteMinimumTracks", 30))
    frames = ev.read_csv(a.run / "frames.csv")
    metric = ev.read_csv(a.run / "metric_track.csv")
    align = ev.read_csv(a.run / "alignment_frames.csv")
    gt = ev.load_gt(a.dataset)

    # ---- tracker health (frames.csv) — frame 0 is 'init' and carries the detector count, not a track
    body = frames[1:]
    success = np.array([r["success"].strip().lower() == "true" for r in body])
    tracks = np.array([float(r["track_count"]) for r in body])
    inliers = np.array([float(r["inlier_count"]) for r in body])
    ratio = np.where(tracks > 0, inliers / np.maximum(tracks, 1), np.nan)
    events = {}
    for r in frames:
        events[r["event"]] = events.get(r["event"], 0) + 1
    ptime = np.array([float(r["process_time_ns"]) for r in body]) / 1e6

    # ---- metric readout / channels (metric_track.csv)
    segments = sorted({int(r["segment_index"]) for r in metric})
    usable = np.array([r["metric_frame_usable"].strip() in ("1", "true", "True") for r in metric])
    gaps = sum(1 for r in metric if r["unknown_translation_gap"].strip() in ("1", "true", "True"))
    h_status, y_status = {}, {}
    for r in metric:
        h_status[r["h_status"] or "(none)"] = h_status.get(r["h_status"] or "(none)", 0) + 1
        y_status[r["yaw_nav_status"] or "(none)"] = y_status.get(r["yaw_nav_status"] or "(none)", 0) + 1
    disagreement = np.array([ev._f(r["yaw_disagreement_deg"]) for r in metric])
    gsd = np.array([ev._f(r["gsd_m_per_px"]) for r in metric])
    align_events = {}
    for r in align:
        align_events[r["event"]] = align_events.get(r["event"], 0) + 1
    hard_losses = [int(r["frame_index"]) for r in align if r["event"] == "hard_loss"]
    dropouts = [int(r["frame_index"]) for r in align if r["event"] == "translation_dropout"]

    # ---- ground-truth error of the VO's own persistent position
    cur = ev.error_curve(gt, align)
    valid = cur["valid"]
    f0 = int(cur["frame"][0]); t0, e0, n0 = gt[f0]
    gt_xy = np.array([[gt[int(f)][1] - e0, gt[int(f)][2] - n0] for f in cur["frame"]])
    est_xy = np.array([[ev._f(r["global_east_m"]), ev._f(r["global_north_m"])] for r in align])
    err_vec = gt_xy - est_xy
    marks = []
    for tm in np.arange(0.0, cur["t"][-1] + 1e-9, a.mark_interval_s):
        i = int(np.argmin(np.abs(cur["t"] - tm)))
        marks.append({"time_s": float(cur["t"][i]), "frame": int(cur["frame"][i]),
                      "error_m": None if not valid[i] else float(cur["err"][i])})
    gt_path = float(np.sum(np.hypot(*np.diff(gt_xy, axis=0).T)))
    est_path = float(np.nansum(np.hypot(*np.diff(est_xy, axis=0).T)))
    inc_gt = np.diff(gt_xy, axis=0); inc_est = np.diff(est_xy, axis=0)
    inc_err = inc_gt - inc_est
    ok = np.isfinite(inc_err).all(axis=1)
    inc_err = inc_err[ok]
    step_len = np.hypot(*inc_gt[ok].T)
    mean_inc = inc_err.mean(axis=0)
    rand_inc = inc_err - mean_inc
    # linear growth fit of |e| against time (drift rate) and against path length
    ev_ok = valid & np.isfinite(cur["err"])
    slope_t = float(np.polyfit(cur["t"][ev_ok], cur["err"][ev_ok], 1)[0]) if ev_ok.sum() > 2 else float("nan")
    s_path = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(gt_xy, axis=0).T))])
    slope_s = float(np.polyfit(s_path[ev_ok], cur["err"][ev_ok], 1)[0]) if ev_ok.sum() > 2 else float("nan")
    e = cur["err"][ev_ok]

    report = {
        "dataset": str(a.dataset), "run": str(a.run), "estimator_config_excerpt": {
            k: est.get(k) for k in ("maxFeatures", "absoluteMinimumTracks", "respawnTrackFraction", "ransacIterations",
                                    "inlierThresholdSq", "motion_model", "downsampleFactor", "shrinkScale")},
        "frames": {"total": len(frames), "duration_s": float(cur["t"][-1] - cur["t"][0]),
                   "success": int(success.sum()), "success_fraction": float(success.mean()),
                   "events": events, "recenters": events.get("recenter", 0), "restarts": events.get("restart", 0),
                   "recenter_rate_per_min": 60.0 * events.get("recenter", 0) / max(cur["t"][-1] - cur["t"][0], 1e-9)},
        "tracks": {"track_count": q(tracks), "inlier_count": q(inliers), "inlier_ratio": q(ratio),
                   "frames_below_2x_absolute_minimum": int((tracks < 2 * abs_min).sum()),
                   "frames_below_absolute_minimum": int((tracks < abs_min).sum()),
                   "absolute_minimum_tracks": abs_min,
                   "lowest_track_frames": [{"frame": int(body[i]["frame_index"]), "tracks": int(tracks[i]), "inliers": int(inliers[i]),
                                            "event": body[i]["event"]} for i in np.argsort(tracks)[:8]],
                   "process_time_ms": q(ptime, (50, 95, 100))},
        "metric_readout": {"segments": len(segments), "segment_indices": segments, "frames_usable": int(usable.sum()),
                           "usable_fraction": float(usable.mean()), "unknown_translation_gap_frames": gaps,
                           "height_status": h_status, "heading_status": y_status,
                           "gsd_m_per_px": q(gsd, (0, 50, 100)),
                           "visual_yaw_disagreement_deg": q(np.abs(disagreement), (50, 95, 100)),
                           "visual_yaw_disagreement_final_deg": float(disagreement[-1]) if np.isfinite(disagreement[-1]) else None},
        "alignment": {"events": align_events, "natural_hard_loss_frames": hard_losses, "translation_dropout_frames": dropouts,
                      "frames_global_position_valid": int(valid.sum()), "frames_total": int(valid.size)},
        "gt_error": {"registration": "translation fixing frame 0; no rotation or scale fitted",
                     "ate_rmse_m": float(np.sqrt(np.mean(e ** 2))), "mean_m": float(np.mean(e)), "median_m": float(np.median(e)),
                     "max_m": float(np.max(e)), "final_m": float(e[-1]), "final_frame": int(cur["frame"][ev_ok][-1]),
                     "marks": marks, "growth_rate_m_per_s": slope_t, "growth_rate_m_per_100m_path": 100.0 * slope_s,
                     "final_error_over_path_pct": 100.0 * float(e[-1]) / gt_path,
                     "gt_path_length_m": gt_path, "estimated_path_length_m": est_path,
                     "path_length_ratio_est_over_gt": est_path / gt_path,
                     "gt_start_end_distance_m": float(np.hypot(*(gt_xy[-1] - gt_xy[0]))),
                     "estimated_start_end_distance_m": float(np.hypot(*(est_xy[-1] - est_xy[0]))),
                     "final_error_vector_m": {"east": float(err_vec[-1, 0]), "north": float(err_vec[-1, 1])},
                     "incremental_error_per_frame": {
                         "mean_vector_m": {"east": float(mean_inc[0]), "north": float(mean_inc[1])},
                         "mean_magnitude_m": float(np.hypot(*mean_inc)),
                         "random_sd_m": {"east": float(rand_inc[:, 0].std()), "north": float(rand_inc[:, 1].std())},
                         "abs_p50_m": float(np.percentile(np.hypot(*inc_err.T), 50)),
                         "abs_p95_m": float(np.percentile(np.hypot(*inc_err.T), 95)),
                         "gt_step_p50_m": float(np.percentile(step_len, 50)),
                         "systematic_share_of_final_error": float(np.hypot(*(mean_inc * len(inc_err))) / max(e[-1], 1e-9))}},
    }
    # declared verdict rule (argued with numbers, not a substitute for the author's call)
    rule = []
    verdict = "GOOD"
    if hard_losses or report["frames"]["success_fraction"] < 0.95 or report["metric_readout"]["usable_fraction"] < 0.95:
        verdict = "BAD"; rule.append("natural hard loss, or <95 % VO success / metric-usable frames")
    else:
        if report["tracks"]["track_count"]["p5"] < 2 * abs_min:
            verdict = "MARGINAL"; rule.append(f"5th-percentile track count {report['tracks']['track_count']['p5']:.0f} < 2 x absoluteMinimumTracks ({2*abs_min})")
        if report["tracks"]["inlier_ratio"]["p5"] < 0.3:
            verdict = "MARGINAL"; rule.append("5th-percentile inlier ratio < 0.3")
        if report["frames"]["recenter_rate_per_min"] > 6:
            verdict = "MARGINAL"; rule.append("more than one recenter per 10 s")
        if report["tracks"]["frames_below_absolute_minimum"] > 0:
            verdict = "MARGINAL"; rule.append("frames at the absolute minimum track count")
    report["verdict_rule"] = {"verdict": verdict, "triggers": rule,
                              "rule": "BAD if any natural hard loss or <95 % success/usable; MARGINAL if p5 tracks < 2x absolute minimum, "
                                      "p5 inlier ratio < 0.3, > 6 recenters/min, or any frame below the absolute minimum; else GOOD. "
                                      "Drift magnitude is deliberately not a criterion."}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "vo_quality.json").write_text(json.dumps(report, indent=2, default=ev._json_default) + "\n")
    fr, tr, ge = report["frames"], report["tracks"], report["gt_error"]
    print(f"{a.dataset.name}: {fr['total']} frames / {fr['duration_s']:.1f} s; success {fr['success']}/{fr['total']}; "
          f"recenters {fr['recenters']}, restarts {fr['restarts']}, hard losses {hard_losses}, dropouts {len(dropouts)}; "
          f"segments {report['metric_readout']['segments']}")
    print(f"  tracks min {tr['track_count']['p0']:.0f} p1 {tr['track_count']['p1']:.0f} p5 {tr['track_count']['p5']:.0f} "
          f"median {tr['track_count']['p50']:.0f}; inlier ratio p5 {tr['inlier_ratio']['p5']:.2f} median {tr['inlier_ratio']['p50']:.2f}; "
          f"frames < {abs_min}: {tr['frames_below_absolute_minimum']}, < {2*abs_min}: {tr['frames_below_2x_absolute_minimum']}")
    print(f"  GT error: ATE {ge['ate_rmse_m']:.2f} m, max {ge['max_m']:.2f} m, final {ge['final_m']:.2f} m over {ge['gt_path_length_m']:.0f} m "
          f"({ge['final_error_over_path_pct']:.2f} %); growth {ge['growth_rate_m_per_s']:.3f} m/s; path ratio est/GT {ge['path_length_ratio_est_over_gt']:.4f}; "
          f"closure GT {ge['gt_start_end_distance_m']:.2f} m vs est {ge['estimated_start_end_distance_m']:.2f} m")
    print("  marks: " + ", ".join(f"{m['time_s']:.0f}s={m['error_m']:.1f}" for m in marks if m["error_m"] is not None))
    print(f"  verdict rule -> {verdict} {rule}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
