"""Known-answer tests for the EXP-INT-001 evaluator on a hand-built recording.

The ground truth is a straight line; the VO_ONLY arm drifts linearly; the INT arm re-anchors once
onto a reference whose stored position is exact, so every quantity has a closed form. A second INT
arm re-anchors onto a reference 60 m from the query, which must be classified wrong-place and
severely harmful, and must fail the milestone criteria mechanically.

Run from the repository root: ``python -m pytest evaluation/tools/int -q``.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate_int_arms as ev  # noqa: E402

N = 101                       # frames 0..100 at 0.1 s
DRIFT_PER_FRAME = 0.2         # VO error grows 0.2 m per frame eastward
REANCHOR_FRAME = 60
EVENT_COLS = ev.read_csv  # placeholder to keep flake quiet


def _write_dataset(d: Path):
    d.mkdir(parents=True)
    with (d / "frames.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_index", "timestamp_s", "image_path"])
        for k in range(N):
            w.writerow([k, f"{k * 0.1:.9f}", f"images/frame_{k:06d}.png"])
    with (d / "groundtruth.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["timestamp_s", "east_m", "north_m", "up_m", "heading_deg", "fix_quality", "valid"])
        for k in range(N):                                   # 10 Hz grid, GT walks north at 1 m/frame from (100, 200)
            w.writerow([f"{k * 0.1:.9f}", 100.0, 200.0 + k, 50.0, 0.0, 1, 1])


def _alignment_frames(run: Path, est_fn, valid_fn=lambda k: True):
    run.mkdir(parents=True, exist_ok=True)
    cols = ("frame_index,timestamp_s,vo_success,translation_usable,local_east_m,local_north_m,heading_deg,"
            "heading_status,heading_valid,global_position_valid,global_pose_valid,global_east_m,global_north_m,"
            "segment_id,alignment_epoch_id").split(",")
    with (run / "alignment_frames.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for k in range(N):
            e, n = est_fn(k)
            v = valid_fn(k)
            w.writerow([k, f"{k * 0.1:.1f}", "true", "true", e, n, 0.0, "FRESH", "true",
                        str(v).lower(), str(v).lower(), e if v else "", n if v else "", 0, 0])


def _events(run: Path, reference_frame: int, stored_ref_xy, before_xy, after_xy):
    header = ev_header()
    rows = []
    def row(**kw):
        r = {h: "" for h in header}; r.update(kw); return r
    rows.append(row(frame_index=reference_frame, timestamp_s=reference_frame * 0.1, kind="reference_inserted",
                    reference_id=0))
    rows.append(row(frame_index=REANCHOR_FRAME, timestamp_s=REANCHOR_FRAME * 0.1, kind="retrieval",
                    top_k_ids="0", top_k_scores="0.99", top_region_score=0.99, competing_region_score=0.5,
                    region_margin=0.49, excluded_recent_ids="", database_size=1, considered_count=1,
                    competitor_exists="true", north_score=0.99, west_score=0.99, agreement="true", dual_status="agreed"))
    rows.append(row(frame_index=REANCHOR_FRAME, timestamp_s=REANCHOR_FRAME * 0.1, kind="attempt", reason="accepted",
                    detail="score=0.99;margin=0.49", verdict="accept:accepted"))
    rows.append(row(frame_index=REANCHOR_FRAME, timestamp_s=REANCHOR_FRAME * 0.1, kind="reanchor", reference_id=0,
                    global_before_east_m=before_xy[0], global_before_north_m=before_xy[1],
                    global_after_east_m=after_xy[0], global_after_north_m=after_xy[1],
                    position_jump_m=math.dist(before_xy, after_xy),
                    query_reference_frame_gap=REANCHOR_FRAME - reference_frame,
                    query_reference_time_gap_s=(REANCHOR_FRAME - reference_frame) * 0.1,
                    top_region_score=0.99, competing_region_score=0.5, region_margin=0.49,
                    north_score=0.99, west_score=0.99, north_margin=0.5, west_margin=0.5, agreement="true",
                    dual_verdict="agreed", fusion_rule="weakest_view", cause="scheduled_exposure",
                    origin_cause="scheduled_exposure", matcher="c0_frozen_ncc", lag_samples=0, support_count=0,
                    source_anchor_id="", e_eff_before=300, e_eff_after=0, recovery_case="drift_correction",
                    competitor_exists="true", top_separation_m=0.0,
                    selected_reference_east_m=stored_ref_xy[0], selected_reference_north_m=stored_ref_xy[1],
                    estimated_query_to_reference_m=0.0, reference_age_frames=REANCHOR_FRAME - reference_frame,
                    reference_age_s=(REANCHOR_FRAME - reference_frame) * 0.1))
    with (run / "alignment_events.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)


def ev_header():
    return ("frame_index,timestamp_s,kind,segment_id,alignment_epoch_id,reference_id,reason,detail,"
            "global_before_east_m,global_before_north_m,global_after_east_m,global_after_north_m,"
            "local_east_m,local_north_m,heading_deg,delta_east_m,delta_north_m,position_jump_m,"
            "query_reference_frame_gap,query_reference_time_gap_s,e_eff_before,e_eff_after,i_eff_before,i_eff_after,"
            "top_k_ids,top_k_scores,top_k_lags,west_top_k_ids,west_top_k_scores,west_top_k_lags,"
            "excluded_recent_ids,database_size,considered_count,top_region_score,competing_region_score,region_margin,"
            "north_score,north_margin,west_score,west_margin,west_lag_samples,agreement,dual_status,dual_verdict,"
            "fusion_rule,temporal_required,cause,origin_cause,origin_frame,escalation_cause,changed_cause,"
            "overrode_retry_gap,matcher,lag_samples,overlap_fraction,support_count,verdict,vo_aux_displacement_error_m,"
            "pose_schema,source_anchor_id,source_reanchor_frame,heading_status,loss_source,recovery_case,"
            "global_position_valid_before,heading_valid,competitor_exists,scored_north,scored_west,"
            "north_top_reference_id,west_top_reference_id,top_separation_m,selected_reference_east_m,"
            "selected_reference_north_m,estimated_query_to_reference_m,reference_age_frames,reference_age_s").split(",")


@pytest.fixture
def scenario(tmp_path):
    ds = tmp_path / "datasets" / "toy"
    _write_dataset(ds)
    vo = tmp_path / "runs" / "vo"
    _alignment_frames(vo, lambda k: (DRIFT_PER_FRAME * k, float(k)))           # drifts east 0.2 m/frame
    # INT good: identical drift until the re-anchor at frame 60 onto reference frame 10 (stored exactly,
    # (2.0, 10.0) — the estimate at frame 10 carried 2 m of drift; the snap moves the position onto it).
    # After the snap the alignment holds the correction: error = |drift(k) - drift(60)| ... simplest: the
    # persistent position becomes GT-relative plus the stored reference's own 2 m error.
    good = tmp_path / "runs" / "int_good"
    _alignment_frames(good, lambda k: (DRIFT_PER_FRAME * k, float(k)) if k < REANCHOR_FRAME
                      else (2.0 + DRIFT_PER_FRAME * (k - REANCHOR_FRAME) - 50 * DRIFT_PER_FRAME + 50 * DRIFT_PER_FRAME, float(k)))
    _events(good, reference_frame=10, stored_ref_xy=(2.0, 10.0),
            before_xy=(DRIFT_PER_FRAME * REANCHOR_FRAME, float(REANCHOR_FRAME)), after_xy=(2.0, float(REANCHOR_FRAME)))
    # INT bad: snaps onto a reference captured 60 frames (60 m north) away from the query's true place.
    bad = tmp_path / "runs" / "int_bad"
    _alignment_frames(bad, lambda k: (DRIFT_PER_FRAME * k, float(k)) if k < REANCHOR_FRAME
                      else (0.0, float(k) - 60.0))
    _events(bad, reference_frame=0, stored_ref_xy=(0.0, 0.0),
            before_xy=(DRIFT_PER_FRAME * REANCHOR_FRAME, float(REANCHOR_FRAME)), after_xy=(0.0, 0.0))
    return ds, vo, good, bad


def test_registration_and_error_curve_are_closed_form(scenario):
    ds, vo, good, bad = scenario
    gt = ev.load_gt(ds)
    curve = ev.error_curve(gt, ev.read_csv(vo / "alignment_frames.csv"))
    assert curve["err"][0] == pytest.approx(0.0)
    assert curve["err"][100] == pytest.approx(DRIFT_PER_FRAME * 100)
    assert curve["valid"].all()


def test_good_snap_is_beneficial_and_meets_proof_of_benefit(scenario, tmp_path):
    ds, vo, good, bad = scenario
    res = ev.compare(ds, vo, {"int_good": good}, tmp_path / "out_good", 50.0, 15.0, 20.0, 10.0)
    arm = res["arms"]["int_good"]
    re_ = ev.read_csv(tmp_path / "out_good" / "reanchors_int_good.csv")
    assert len(re_) == 1
    r = re_[0]
    assert float(r["e_before_m"]) == pytest.approx(DRIFT_PER_FRAME * REANCHOR_FRAME)   # 12 m before
    assert float(r["e_after_m"]) == pytest.approx(2.0)                                  # the reference's own 2 m
    assert float(r["delta_e_m"]) == pytest.approx(2.0 - 12.0)
    assert float(r["gt_query_to_reference_m"]) == pytest.approx(50.0)                   # 50 frames = 50 m north apart
    assert r["wrong_place"] == "False" and r["beneficial"] == "True" and r["severely_harmful"] == "False"
    v, i = arm["vo_only_on_common_frames"], arm["int_on_common_frames"]
    assert i["ate_rmse_m"] < v["ate_rmse_m"] and i["final_m"] < v["final_m"]
    assert arm["proof_of_benefit_criteria"]["3_some_drift_correction_with_negative_delta_e"] is True
    assert arm["proof_of_benefit_criteria"]["4_no_wrong_place_accept"] is True
    assert arm["verdict_mechanical"] in ("PROOF_OF_BENEFIT", "STRONG")
    assert (tmp_path / "out_good" / "metrics.json").exists()
    m = json.loads((tmp_path / "out_good" / "metrics.json").read_text())
    assert m["registration"].startswith("translation fixing frame 0")


def test_wrong_place_snap_is_classified_and_fails_the_milestone(scenario, tmp_path):
    ds, vo, good, bad = scenario
    res = ev.compare(ds, vo, {"int_bad": bad}, tmp_path / "out_bad", 50.0, 15.0, 20.0, 10.0)
    arm = res["arms"]["int_bad"]
    r = ev.read_csv(tmp_path / "out_bad" / "reanchors_int_bad.csv")[0]
    assert float(r["gt_query_to_reference_m"]) == pytest.approx(60.0)
    assert r["wrong_place"] == "True"
    assert float(r["delta_e_m"]) > 15.0 and r["severely_harmful"] == "True"
    assert arm["proof_of_benefit_criteria"]["4_no_wrong_place_accept"] is False
    assert arm["verdict_mechanical"] == "NOT_MET"


def test_events_only_pools_the_same_classification(scenario, tmp_path):
    ds, vo, good, bad = scenario
    res = ev.events_only(ds, bad, tmp_path / "eo", 50.0, 15.0, 20.0)
    assert res["reanchors"]["wrong_place"] == 1 and res["reanchors"]["severely_harmful"] == 1
    assert res["events"]["kinds"]["reanchor"] == 1
    assert res["recency_density"]["retrievals"] == 1
