"""Known-answer tests for the EXP-INT-002 analysis tools on hand-built recordings.

Every quantity has a closed form: an out-and-back ground-truth path, a VO whose error grows
linearly north, one trusted reference, one snap. `snap_longitudinal` must reproduce the vectors
``a = u + r``, ``b``, ``c = b − a`` and the exact identity ``|e_iso|² − |e_nok|² = |a|² − |b|² − 2 d·c``
on every frame; a drift that reverses after the snap must be classified B, a drift that continues
must be classified A. `gt_revisit_events` must find exactly the return leg beyond the recency
horizon; `inject_synthetic_drift` must keep the two files the replay cross-checks consistent and
write a declared manifest; `vo_quality_report` must reproduce the closed-form error curve.

Run from the repository root: ``python -m pytest evaluation/tools/int -q``.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate_int_arms as ev  # noqa: E402
import gt_revisit_events as gre  # noqa: E402
import inject_synthetic_drift as isd  # noqa: E402
import snap_longitudinal as sl  # noqa: E402
import vo_quality_report as vq  # noqa: E402
from test_evaluate_int_arms import ev_header  # noqa: E402

N = 100                 # frames 0..99 at 0.1 s
REF_FRAME = 10          # reference captured at GT east 5.0
SNAP_FRAME = 88         # query at GT east 6.0 -> u = (1, 0)
SLOPE = 0.02            # VO error grows 0.02 m / frame north before the snap


def gt_east(k: int) -> float:
    return 0.5 * k if k <= 50 else 25.0 - 0.5 * (k - 50)


def vo_err(k: int, reverse_after: bool) -> tuple[float, float]:
    """The VO's error vector e = GT − VO at frame k (north only)."""
    if k <= SNAP_FRAME or not reverse_after:
        return (0.0, SLOPE * k)
    return (0.0, SLOPE * SNAP_FRAME - 0.2 * (k - SNAP_FRAME))


def _dataset(d: Path):
    d.mkdir(parents=True)
    with (d / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_index", "timestamp_s", "image_path"])
        for k in range(N):
            w.writerow([k, f"{k * 0.1:.9f}", f"images/frame_{k:06d}.png"])
    with (d / "groundtruth.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["timestamp_s", "east_m", "north_m", "up_m", "heading_deg", "fix_quality", "valid"])
        for k in range(N):
            w.writerow([f"{k * 0.1:.9f}", gt_east(k), 0.0, 50.0, 0.0, 1, 1])


ALIGN_COLS = ("frame_index,timestamp_s,vo_success,translation_usable,local_east_m,local_north_m,heading_deg,"
              "heading_status,heading_valid,global_position_valid,global_pose_valid,global_east_m,global_north_m,"
              "segment_id,alignment_epoch_id,event,request_pending,request_cause,retry_gap_blocking,skyline_present,"
              "north_valid,west_valid").split(",")


def _align(run: Path, pos_fn, local_fn, captures=()):
    run.mkdir(parents=True, exist_ok=True)
    with (run / "alignment_frames.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(ALIGN_COLS)
        for k in range(N):
            e, n = pos_fn(k); le, ln = local_fn(k)
            cap = k in captures
            w.writerow([k, f"{k * 0.1:.1f}", "true", "true", le, ln, 0.0, "FRESH", "true", "true", "true", e, n, 0, 0,
                        "init" if k == 0 else "none", "true", "scheduled_exposure", "false", str(cap).lower(),
                        str(cap).lower(), str(cap).lower()])


def _events(run: Path, stored_ref, before, after):
    header = ev_header()
    def row(**kw):
        r = {h: "" for h in header}; r.update(kw); return r
    rows = [row(frame_index=REF_FRAME, timestamp_s=REF_FRAME * 0.1, kind="reference_inserted", reference_id=0,
                global_after_east_m=stored_ref[0], global_after_north_m=stored_ref[1]),
            row(frame_index=SNAP_FRAME, timestamp_s=SNAP_FRAME * 0.1, kind="retrieval", top_k_ids="0", top_k_scores="0.99",
                west_top_k_ids="0", west_top_k_scores="0.98", top_region_score=0.98, competing_region_score=0.5,
                region_margin=0.48, excluded_recent_ids="", database_size=1, considered_count=1, competitor_exists="true",
                north_score=0.99, west_score=0.98, agreement="true", dual_status="complete"),
            row(frame_index=SNAP_FRAME, timestamp_s=SNAP_FRAME * 0.1, kind="attempt", reason="accepted", reference_id=0,
                detail="score=0.98;margin=0.48", verdict="accept:accepted"),
            row(frame_index=SNAP_FRAME, timestamp_s=SNAP_FRAME * 0.1, kind="reanchor", reference_id=0,
                global_before_east_m=before[0], global_before_north_m=before[1], global_after_east_m=after[0],
                global_after_north_m=after[1], position_jump_m=math.dist(before, after),
                query_reference_frame_gap=SNAP_FRAME - REF_FRAME, query_reference_time_gap_s=(SNAP_FRAME - REF_FRAME) * 0.1,
                top_region_score=0.98, competing_region_score=0.5, region_margin=0.48, north_score=0.99, west_score=0.98,
                north_margin=0.5, west_margin=0.5, agreement="true", dual_verdict="agreed", fusion_rule="weakest_view",
                cause="scheduled_exposure", origin_cause="scheduled_exposure", matcher="c0_frozen_ncc", lag_samples=0,
                support_count=0, e_eff_before=SNAP_FRAME, e_eff_after=REF_FRAME, recovery_case="drift_correction",
                competitor_exists="true", top_separation_m=0.0, selected_reference_east_m=after[0],
                selected_reference_north_m=after[1], estimated_query_to_reference_m=math.dist(before, after),
                reference_age_frames=SNAP_FRAME - REF_FRAME, reference_age_s=(SNAP_FRAME - REF_FRAME) * 0.1)]
    with (run / "alignment_events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)


def _build(tmp_path: Path, reverse_after: bool):
    ds = tmp_path / "datasets" / "toy"; _dataset(ds)
    def vo_pos(k):
        e = vo_err(k, reverse_after); return (gt_east(k) - e[0], 0.0 - e[1])
    vo = tmp_path / "runs" / ("vo_rev" if reverse_after else "vo"); _align(vo, vo_pos, vo_pos, captures={REF_FRAME, SNAP_FRAME})
    stored = vo_pos(REF_FRAME)                                   # the reference stores the VO's own position then
    c = (stored[0] - vo_pos(SNAP_FRAME)[0], stored[1] - vo_pos(SNAP_FRAME)[1])
    def int_pos(k):
        p = vo_pos(k); return p if k < SNAP_FRAME else (p[0] + c[0], p[1] + c[1])
    it = tmp_path / "runs" / ("int_rev" if reverse_after else "int"); _align(it, int_pos, vo_pos, captures={REF_FRAME, SNAP_FRAME})
    _events(it, stored, vo_pos(SNAP_FRAME), stored)
    return ds, vo, it


def test_snap_longitudinal_vectors_and_identity_are_exact(tmp_path):
    ds, vo, it = _build(tmp_path, reverse_after=False)
    s = sl.analyse(ds, vo, it, tmp_path / "out", [1.0], 0.5)
    assert s["same_vo_increments"]["identical"] is True
    snap = s["snaps"][0]
    assert snap["u_vec_m"] == pytest.approx([1.0, 0.0])                       # GT(88) − GT(10)
    assert snap["r_vec_m"] == pytest.approx([0.0, SLOPE * REF_FRAME])         # stored reference's own error
    assert snap["a_vec_m"] == pytest.approx([1.0, SLOPE * REF_FRAME])         # a = u + r
    assert snap["b_vec_m"] == pytest.approx([0.0, SLOPE * SNAP_FRAME])
    assert snap["c_vec_m"] == pytest.approx([-1.0, SLOPE * (SNAP_FRAME - REF_FRAME)])   # c = b − a
    assert snap["vo_drift_since_reference_insertion_m"] == pytest.approx(SLOPE * (SNAP_FRAME - REF_FRAME))
    assert snap["delta_e_0_m"] == pytest.approx(math.hypot(1.0, 0.2) - 1.76)
    h = snap["horizons"]["1s"]
    # one second (10 frames) later: e_iso = (1, 0.2 + 0.2) ; e_nok = (0, 1.76 + 0.2)
    assert h["e_iso_m"] == pytest.approx(math.hypot(1.0, 0.4))
    assert h["e_nok_m"] == pytest.approx(1.96)
    assert h["delta_e_iso_m"] < 0
    assert snap["identity_max_abs_residual"] < 1e-9
    assert snap["classification"] == "A_persistently_beneficial"
    series = ev.read_csv(tmp_path / "out" / f"snap_{SNAP_FRAME}_series.csv")
    assert len(series) == N - SNAP_FRAME - 1


def test_snap_longitudinal_classifies_reversing_drift_as_B(tmp_path):
    ds, vo, it = _build(tmp_path, reverse_after=True)
    s = sl.analyse(ds, vo, it, tmp_path / "out", [1.0], 0.5)
    snap = s["snaps"][0]
    assert snap["delta_e_0_m"] < 0                                              # instantaneously beneficial
    h = snap["horizons"]["1s"]
    assert h["e_nok_m"] == pytest.approx(abs(1.76 - 2.0))                       # the VO's error swung through zero
    assert h["e_iso_m"] == pytest.approx(math.hypot(1.0, 0.2 - 2.0))
    assert h["delta_e_iso_m"] > 0 and h["d_dot_c"] < 0                          # drift against the correction
    assert h["compensation_index_of_b"] > 0.5                                   # b was cancelling the coming drift
    assert snap["classification"] == "B_beneficial_then_harmful"


def test_snap_longitudinal_refuses_mismatched_or_reanchored_vo_arm(tmp_path):
    ds, vo, it = _build(tmp_path, reverse_after=False)
    # (a) a VO_ONLY arm whose local increments differ from the INT arm's (here: scaled) is refused
    rows = ev.read_csv(vo / "alignment_frames.csv")
    bad = tmp_path / "runs" / "vo_scaled"; bad.mkdir(parents=True)
    with (bad / "alignment_frames.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            r = dict(r); r["local_east_m"] = str(2 * float(r["local_east_m"])); r["global_east_m"] = r["local_east_m"]; w.writerow(r)
    with pytest.raises(SystemExit):
        sl.analyse(ds, bad, it, tmp_path / "out_bad", [1.0], 0.5)
    # (b) an INT run passed as --vo-only (same locals, but it re-anchored) is refused
    with pytest.raises(SystemExit):
        sl.analyse(ds, it, it, tmp_path / "out_bad2", [1.0], 0.5)


def test_opportunity_table_reports_the_pre_query_error_on_accepted_rows(tmp_path):
    import revisit_opportunity_table as rot  # noqa: E402  (import here: the module needs the fixture's events)
    ds, vo, it = _build(tmp_path, reverse_after=False)
    events = {"events": [{"event_id": 0, "later_pass": {"time_start_s": SNAP_FRAME * 0.1, "time_end_s": SNAP_FRAME * 0.1}}]}
    ej = tmp_path / "events.json"; ej.write_text(json.dumps(events))
    rows = rot.build(ds, vo, it, ej, tmp_path / "opp", recency_s=5.0, margin_s=0.0, downstream_s=0.5)
    acc = [r for r in rows if r["accepted"]]
    assert len(acc) == 1
    # the alignment_frames row of the accept frame is post-snap; the table must still report the PRE-query error
    assert acc[0]["int_error_before_query_m"] == pytest.approx(SLOPE * SNAP_FRAME)            # |b| = 1.76
    assert acc[0]["vo_only_error_m"] == pytest.approx(SLOPE * SNAP_FRAME)
    assert acc[0]["delta_e_0_m"] == pytest.approx(math.hypot(1.0, SLOPE * REF_FRAME) - SLOPE * SNAP_FRAME)


def test_inject_synthetic_drift_refuses_a_step_on_frame_zero(tmp_path):
    ds, vo, it = _build(tmp_path, reverse_after=False)
    al = ev.read_csv(vo / "alignment_frames.csv")
    with (vo / "metric_track.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_index", "timestamp_s", "segment_index", "segment_east_m", "segment_north_m", "metric_frame_usable"])
        for r in al:
            w.writerow([r["frame_index"], r["timestamp_s"], 0, r["local_east_m"], r["local_north_m"], 1])
    with pytest.raises(SystemExit):
        isd.main(["--vo-run", str(vo), "--out", str(tmp_path / "d0"), "--model", "step", "--step", "3.0", "--step-frame", "0"])
    isd.main(["--vo-run", str(vo), "--out", str(tmp_path / "d1"), "--model", "step", "--step", "3.0", "--bearing-deg", "90", "--step-frame", "30"])
    a2 = ev.read_csv(tmp_path / "d1" / "alignment_frames.csv")
    assert float(a2[0]["global_east_m"]) == float(al[0]["global_east_m"])                     # the root is untouched
    assert float(a2[30]["local_east_m"]) - float(al[30]["local_east_m"]) == pytest.approx(3.0)


def test_gt_revisit_events_finds_the_return_leg_only(tmp_path):
    ds = tmp_path / "datasets" / "toy"; _dataset(ds)
    frames, t, xy = gre.load_positions(ds)
    d_min, g_min = gre.nearest_prior_pass(t, xy, 5.0)
    s = gre.cumulative_path(xy)
    # radius 0.5 m: frame 74 (1.0 m from frame 24, the newest frame ≥ 5 s older) stays outside, frame 75 retraces frame 25 exactly
    events, inside = gre.group_events(frames, t, xy, s, d_min, g_min, 0.5, 10, None, None)
    assert len(events) == 1
    e = events[0]
    assert e["later_pass"]["frame_start"] == 75 and e["later_pass"]["frame_end"] == 99   # elapsed ≥ 5 s from frame 75 on
    assert e["earlier_pass"]["frame"] == 100 - e["closest_approach"]["frame"]
    assert e["closest_approach"]["separation_m"] == pytest.approx(0.0)
    assert e["traversal_relation"] == "reverse"
    assert e["elapsed_s"] == pytest.approx(5.0)
    assert int(inside.sum()) == 25


def test_inject_synthetic_drift_keeps_files_consistent_and_declares_itself(tmp_path, monkeypatch):
    ds, vo, it = _build(tmp_path, reverse_after=False)
    # a metric_track.csv consistent with the alignment file, as the replay requires
    al = ev.read_csv(vo / "alignment_frames.csv")
    with (vo / "metric_track.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_index", "timestamp_s", "segment_index", "segment_east_m", "segment_north_m",
                                       "metric_east_m", "metric_north_m", "metric_frame_usable"])
        for r in al:
            w.writerow([r["frame_index"], r["timestamp_s"], 0, r["local_east_m"], r["local_north_m"], r["local_east_m"], r["local_north_m"], 1])
    (vo / "manifest.json").write_text(json.dumps({"run_id": "toy-vo"}))
    out = tmp_path / "runs" / "drifted"
    isd.main(["--vo-run", str(vo), "--out", str(out), "--model", "linear", "--rate", "0.5", "--bearing-deg", "90"])
    m2 = ev.read_csv(out / "metric_track.csv"); a2 = ev.read_csv(out / "alignment_frames.csv")
    for r_m, r_a, r_0 in zip(m2, a2, al):
        assert float(r_m["segment_east_m"]) == float(r_a["local_east_m"])            # what the replay cross-checks
        assert float(r_a["local_east_m"]) == float(r_a["global_east_m"])
        k = int(r_a["frame_index"])
        assert float(r_a["local_east_m"]) - float(r_0["local_east_m"]) == pytest.approx(0.5 * 0.1 * k)   # east drift 0.5 m/s
        assert float(r_a["local_north_m"]) - float(r_0["local_north_m"]) == pytest.approx(0.0, abs=1e-12)
    man = json.loads((out / "manifest.json").read_text())
    assert man["run_id"].endswith("-synthetic-drift-linear")
    assert man["synthetic_drift"]["final_drift_m"]["magnitude"] == pytest.approx(0.5 * 0.1 * (N - 1))
    assert "SYNTHETIC" in man["synthetic_drift"]["evidence_tier_note"]
    assert set(man["synthetic_drift"]["source_sha256"]) == {"metric_track.csv", "alignment_frames.csv"}


def test_vo_quality_report_reproduces_the_closed_form_error(tmp_path):
    ds, vo, it = _build(tmp_path, reverse_after=False)
    al = ev.read_csv(vo / "alignment_frames.csv")
    with (vo / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_index", "timestamp_s", "success", "event", "track_count", "inlier_count", "process_time_ns"])
        for r in al:
            k = int(r["frame_index"]); w.writerow([k, r["timestamp_s"], "true", "init" if k == 0 else ("recenter" if k == 50 else "none"), 500 if k != 30 else 40, 450 if k != 30 else 35, 1000000])
    with (vo / "metric_track.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_index", "timestamp_s", "segment_index", "segment_east_m", "segment_north_m", "metric_frame_usable",
                                       "unknown_translation_gap", "h_status", "yaw_nav_status", "yaw_disagreement_deg", "gsd_m_per_px"])
        for r in al:
            w.writerow([r["frame_index"], r["timestamp_s"], 0, r["local_east_m"], r["local_north_m"], 1, 0, "FRESH", "FRESH", 0.1, 0.07])
    (vo / "manifest.json").write_text(json.dumps({"estimator_config": {"absoluteMinimumTracks": 30, "maxFeatures": 300}}))
    vq.main(["--dataset", str(ds), "--run", str(vo), "--out", str(tmp_path / "q")])
    rep = json.loads((tmp_path / "q" / "vo_quality.json").read_text())
    assert rep["frames"]["success_fraction"] == pytest.approx(1.0)
    assert rep["frames"]["recenters"] == 1 and rep["frames"]["restarts"] == 0
    assert rep["gt_error"]["final_m"] == pytest.approx(SLOPE * (N - 1))
    assert rep["gt_error"]["gt_start_end_distance_m"] == pytest.approx(0.5)      # out 25 m, back to 0.5 m
    assert rep["tracks"]["track_count"]["p0"] == 40 and rep["tracks"]["frames_below_2x_absolute_minimum"] == 1
    assert rep["alignment"]["natural_hard_loss_frames"] == []
    assert rep["verdict_rule"]["verdict"] == "MARGINAL"                            # a frame below 2x the minimum
