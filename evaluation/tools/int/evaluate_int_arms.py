"""EXP-INT-001 evaluator: VO_ONLY versus INT on one recording, against simulator ground truth.

Both arms are `VoRunnerApp` runs over the **same** dataset, metric scaling, height channel,
authoritative heading, estimator and readout; they differ only in whether the relocalization layer
may move the persistent position. Each writes ``alignment_frames.csv`` (the persistent position per
frame, ``global_position_valid``) and the INT arm also ``alignment_events.csv``. This tool joins them
with ``datasets/<id>/groundtruth.csv`` and computes the pre-registered quantities — nothing here is
tuned, fitted or chosen after the fact:

* **Frame registration** — a translation that maps the estimate's origin onto the ground truth at
  frame 0, identically for both arms. No rotation is fitted (the heading is the authoritative
  external channel's, `DEC-VO-010`) and no scale is fitted (the scale is the height channel's,
  `DEC-VO-009`); a least-squares alignment would absorb the very drift being measured.
* **Per-frame position error** ``e(f) = |(GT(f) - GT(0)) - est(f)|`` on the frames where the arm's
  persistent position is VALID. **ATE RMSE** is the root mean square over the frames where **both**
  arms are valid (a paired comparison); the **final error** is at the last such frame.
* **Re-anchor table** — for every accepted correction: ``e_before``/``e_after``/``delta_e`` (the
  drift-correction case; ``e_before`` is left empty in the hard-loss-recovery case, where no
  navigation position existed before the query and none is fabricated), the ground-truth distance
  between the query and the selected reference's capture, the reference age, both views' scores,
  the margin, competitor existence, the trigger cause and the lineage — copied from the sidecar.
* **Classification thresholds** (pre-registered): a re-anchor onto a reference more than
  ``--wrong-place-m`` (50) from the query's true position is a **wrong-place** accept; a
  drift correction with ``delta_e > +--severe-harm-m`` (15) is **severely harmful**; a reference
  within ``--genuine-m`` (20) is a **genuine revisit** accept.
* **Milestone criteria** (pre-registered in `EXP-INT-001`) are evaluated mechanically and reported
  as booleans; the procedural criterion (no post-result tuning) is not something a script can know.

Run from the repository root::

    python evaluation/tools/int/evaluate_int_arms.py --dataset datasets/<id> --vo-only runs/<vo-only run> \\
        --int runs/<int run> --out <dir>

``--int`` may be given several times (e.g. the C0 arm and the C1-4 fallback). ``--events-only`` (with
one ``--int`` and no ``--vo-only``) tabulates re-anchors and retrievals against ground truth without
the trajectory comparison — the calibration sweep's use.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

MATERIAL_IMPROVEMENT_PCT = 10.0


# ----------------------------------------------------------------------------- loading

def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def load_gt(dataset: Path) -> dict:
    """frame -> (t, east, north) from frames.csv timestamps matched onto the ground-truth grid."""
    frames = read_csv(dataset / "frames.csv")
    gt = read_csv(dataset / "groundtruth.csv")
    t = np.array([float(r["timestamp_s"]) for r in gt])
    e = np.array([float(r["east_m"]) for r in gt])
    n = np.array([float(r["north_m"]) for r in gt])
    out = {}
    for r in frames:
        k = int(r["frame_index"]); tk = float(r["timestamp_s"])
        i = int(np.argmin(np.abs(t - tk)))
        if abs(t[i] - tk) > 1e-6:
            raise SystemExit(f"{dataset}: frame {k} at {tk} s is not on the ground-truth grid")
        out[k] = (tk, float(e[i]), float(n[i]))
    return out


def _f(s):
    s = (s or "").strip()
    return float("nan") if s == "" else float(s)


def error_curve(gt: dict, frames_rows: list[dict]) -> dict:
    """Per-frame registered error for one arm. Returns arrays keyed by frame index."""
    f0 = int(frames_rows[0]["frame_index"])
    if not (abs(_f(frames_rows[0]["global_east_m"])) < 1e-9 and abs(_f(frames_rows[0]["global_north_m"])) < 1e-9):
        raise SystemExit("the persistent position is not (0, 0) at the first frame; registration assumes the root anchor")
    t0, e0, n0 = gt[f0]
    frames, ts, valid, err = [], [], [], []
    for r in frames_rows:
        k = int(r["frame_index"])
        v = r["global_position_valid"].strip().lower() == "true"
        tk, ge, gn = gt[k]
        if v:
            ex, ey = _f(r["global_east_m"]), _f(r["global_north_m"])
            err.append(math.hypot((ge - e0) - ex, (gn - n0) - ey))
        else:
            err.append(float("nan"))
        frames.append(k); ts.append(tk); valid.append(v)
    return {"frame": np.array(frames), "t": np.array(ts), "valid": np.array(valid), "err": np.array(err)}


# ----------------------------------------------------------------------------- events

def reference_frames(events: list[dict]) -> dict:
    """reference_id -> the frame it was inserted on (from the insertion rows)."""
    out = {}
    for r in events:
        # kind = reference_inserted carries the new id; reference_rejected carries the reason.
        if r["kind"] == "reference_inserted" and r["reference_id"].strip():
            out[int(r["reference_id"])] = int(r["frame_index"])
    return out


def reanchor_table(gt: dict, events: list[dict], wrong_place_m: float, severe_harm_m: float,
                   genuine_m: float) -> list[dict]:
    f0 = min(gt)
    _, e0, n0 = gt[f0]
    ref_frames = reference_frames(events)
    rows = []
    for r in events:
        if r["kind"] != "reanchor":
            continue
        q = int(r["frame_index"])
        tq, qe, qn = gt[q]
        ref_id = int(r["reference_id"])
        gap = int(float(r["query_reference_frame_gap"]))
        ref_frame = ref_frames.get(ref_id, q - gap)
        _, re_, rn = gt[ref_frame]
        d_ref = math.hypot(qe - re_, qn - rn)
        gb = (_f(r["global_before_east_m"]), _f(r["global_before_north_m"]))
        ga = (_f(r["global_after_east_m"]), _f(r["global_after_north_m"]))
        e_after = math.hypot((qe - e0) - ga[0], (qn - n0) - ga[1])
        drift_case = r["recovery_case"] == "drift_correction" and not math.isnan(gb[0])
        e_before = math.hypot((qe - e0) - gb[0], (qn - n0) - gb[1]) if drift_case else float("nan")
        delta_e = e_after - e_before if drift_case else float("nan")
        rows.append({
            "frame": q, "time_s": tq, "recovery_case": r["recovery_case"], "reference_id": ref_id,
            "reference_frame": ref_frame, "reference_age_s": _f(r["reference_age_s"]),
            "gt_query_to_reference_m": d_ref,
            "e_before_m": e_before, "e_after_m": e_after, "delta_e_m": delta_e,
            "position_jump_m": _f(r["position_jump_m"]),
            "top_region_score": _f(r["top_region_score"]), "competing_region_score": _f(r["competing_region_score"]),
            "region_margin": _f(r["region_margin"]), "north_score": _f(r["north_score"]), "west_score": _f(r["west_score"]),
            "north_margin": _f(r["north_margin"]), "west_margin": _f(r["west_margin"]),
            "agreement": r["agreement"], "dual_verdict": r["dual_verdict"], "fusion_rule": r["fusion_rule"],
            "competitor_exists": r["competitor_exists"], "top_separation_m": _f(r["top_separation_m"]),
            "cause": r["cause"], "origin_cause": r["origin_cause"], "matcher": r["matcher"],
            "lag_samples": r["lag_samples"], "support_count": r["support_count"],
            "source_anchor_id": r["source_anchor_id"], "e_eff_before": r["e_eff_before"], "e_eff_after": r["e_eff_after"],
            "wrong_place": d_ref > wrong_place_m,
            "genuine_revisit": d_ref <= genuine_m,
            "severely_harmful": drift_case and delta_e > severe_harm_m,
            "harmful": drift_case and delta_e > 0.0,
            "beneficial": drift_case and delta_e < 0.0,
        })
    return rows


def retrieval_table(gt: dict, events: list[dict]) -> list[dict]:
    """Every retrieval with the verdict it received and the recency/density context (§10)."""
    ref_frames = reference_frames(events)
    attempts = {}
    for r in events:
        if r["kind"] == "attempt":
            attempts[int(r["frame_index"])] = r
    rows = []
    for r in events:
        if r["kind"] != "retrieval":
            continue
        q = int(r["frame_index"])
        tq, qe, qn = gt[q]
        excluded = {int(x) for x in r["excluded_recent_ids"].split(";") if x.strip()}
        eligible = [(rid, fr) for rid, fr in ref_frames.items() if fr < q and rid not in excluded]
        nearest_eligible = min((math.hypot(qe - gt[fr][1], qn - gt[fr][2]) for _, fr in eligible), default=float("nan"))
        nearest_any = min((math.hypot(qe - gt[fr][1], qn - gt[fr][2]) for fr in ref_frames.values() if fr < q),
                          default=float("nan"))
        top_ids = [int(x) for x in r["top_k_ids"].split(";") if x.strip()]
        top1_d = (math.hypot(qe - gt[ref_frames[top_ids[0]]][1], qn - gt[ref_frames[top_ids[0]]][2])
                  if top_ids and top_ids[0] in ref_frames else float("nan"))
        a = attempts.get(q, {})
        rows.append({
            "frame": q, "time_s": tq, "database_size": int(float(r["database_size"] or 0)),
            "considered_count": int(float(r["considered_count"] or 0)), "excluded_recent": len(excluded),
            "n_eligible": len(eligible), "nearest_eligible_reference_m": nearest_eligible,
            "nearest_any_reference_m": nearest_any, "top1_gt_distance_m": top1_d,
            "top_region_score": _f(r["top_region_score"]), "competing_region_score": _f(r["competing_region_score"]),
            "region_margin": _f(r["region_margin"]), "competitor_exists": r["competitor_exists"],
            "north_score": _f(r["north_score"]), "west_score": _f(r["west_score"]),
            "agreement": r["agreement"], "dual_status": r["dual_status"],
            "verdict": a.get("verdict", ""), "outcome": a.get("reason", ""),
            # the attempt's detail starts with the gate reason (weak_match, ambiguous_region, ...)
            "why": (a.get("detail", "") or "").split(";")[0],
        })
    return rows


def event_counts(events: list[dict]) -> dict:
    kinds, verdicts, insertions = {}, {}, {}
    for r in events:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        if r["kind"] == "attempt":
            v = r["verdict"] or r["reason"]
            verdicts[v] = verdicts.get(v, 0) + 1
        if r["kind"] == "reference_inserted":
            insertions["inserted"] = insertions.get("inserted", 0) + 1
        elif r["kind"] == "reference_rejected":
            k = r["reason"].strip() or "rejected"
            insertions[k] = insertions.get(k, 0) + 1
    return {"kinds": kinds, "attempt_verdicts": verdicts, "insertions": insertions}


# ----------------------------------------------------------------------------- comparison

def summarise_arm(curve: dict, mask: np.ndarray) -> dict:
    e = curve["err"][mask]
    return {"n_frames": int(mask.sum()), "ate_rmse_m": float(np.sqrt(np.mean(e ** 2))),
            "mean_m": float(np.mean(e)), "median_m": float(np.median(e)), "max_m": float(np.max(e)),
            "final_m": float(e[-1]), "final_frame": int(curve["frame"][mask][-1]),
            "frames_valid": int(curve["valid"].sum()), "frames_total": int(curve["valid"].size)}


def compare(dataset: Path, vo_run: Path, int_runs: dict, out: Path, wrong_place_m: float,
            severe_harm_m: float, genuine_m: float, material_pct: float) -> dict:
    gt = load_gt(dataset)
    vo = error_curve(gt, read_csv(vo_run / "alignment_frames.csv"))
    result = {"dataset": str(dataset), "vo_only_run": str(vo_run), "registration":
              "translation fixing frame 0 (identical for both arms); no rotation or scale fitted",
              "thresholds": {"wrong_place_m": wrong_place_m, "severe_harm_m": severe_harm_m,
                             "genuine_revisit_m": genuine_m, "material_improvement_pct": material_pct},
              "arms": {}}
    out.mkdir(parents=True, exist_ok=True)
    curves = {"vo_only": vo}
    for name, run in int_runs.items():
        frames_rows = read_csv(run / "alignment_frames.csv")
        events = read_csv(run / "alignment_events.csv")
        cur = error_curve(gt, frames_rows)
        curves[name] = cur
        both = vo["valid"] & cur["valid"]
        if both.sum() == 0:
            raise SystemExit(f"{name}: no frame where both arms have a valid persistent position")
        vo_s = summarise_arm(vo, both)
        int_s = summarise_arm(cur, both)
        re_rows = reanchor_table(gt, events, wrong_place_m, severe_harm_m, genuine_m)
        rt_rows = retrieval_table(gt, events)
        with (out / f"reanchors_{name}.csv").open("w", newline="") as f:
            if re_rows:
                w = csv.DictWriter(f, fieldnames=list(re_rows[0].keys())); w.writeheader(); w.writerows(re_rows)
            else:
                f.write("frame\n")
        with (out / f"retrievals_{name}.csv").open("w", newline="") as f:
            if rt_rows:
                w = csv.DictWriter(f, fieldnames=list(rt_rows[0].keys())); w.writeheader(); w.writerows(rt_rows)
            else:
                f.write("frame\n")
        drift = [r for r in re_rows if r["recovery_case"] == "drift_correction"]
        first_re = min((r["frame"] for r in re_rows), default=None)
        persist = None
        if first_re is not None:
            after = both & (vo["frame"] > first_re + 1)
            if after.sum() > 0:
                persist = float(np.mean(cur["err"][after] < vo["err"][after]))
        pct = 100.0 * (vo_s["ate_rmse_m"] - int_s["ate_rmse_m"]) / vo_s["ate_rmse_m"]
        proof = {
            "1_ate_rmse_lower": int_s["ate_rmse_m"] < vo_s["ate_rmse_m"],
            "2_final_error_lower": int_s["final_m"] < vo_s["final_m"],
            "3_some_drift_correction_with_negative_delta_e": any(r["beneficial"] for r in drift),
            "4_no_wrong_place_accept": not any(r["wrong_place"] for r in re_rows),
            "5_pre_registered_config_without_post_result_tuning": "procedural — see the experiment record",
        }
        strong = {
            "ate_improvement_pct": pct, "ate_improvement_material": pct >= material_pct,
            "no_severely_harmful_drift_correction": not any(r["severely_harmful"] for r in drift),
            "improvement_persists_fraction_after_first_reanchor": persist,
            "improvement_persists": (persist is not None and persist >= 0.5),
        }
        proof_ok = all(v is True for k, v in proof.items() if k[0] in "1234")
        strong_ok = proof_ok and strong["ate_improvement_material"] and strong["no_severely_harmful_drift_correction"] \
            and strong["improvement_persists"]
        result["arms"][name] = {
            "run": str(run), "vo_only_on_common_frames": vo_s, "int_on_common_frames": int_s,
            "frames_int_unknown": int((~cur["valid"]).sum()),
            "ate_change_pct_negative_is_better": -pct, "final_error_change_m": int_s["final_m"] - vo_s["final_m"],
            "reanchors": {"total": len(re_rows), "drift_correction": len(drift),
                          "hard_loss_recovery": len(re_rows) - len(drift),
                          "genuine_revisit": sum(r["genuine_revisit"] for r in re_rows),
                          "wrong_place": sum(r["wrong_place"] for r in re_rows),
                          "beneficial": sum(r["beneficial"] for r in drift), "harmful": sum(r["harmful"] for r in drift),
                          "severely_harmful": sum(r["severely_harmful"] for r in drift)},
            "events": event_counts(events),
            "recency_density": _recency_summary(rt_rows),
            "proof_of_benefit_criteria": proof, "strong_criteria": strong,
            "verdict_mechanical": ("STRONG" if strong_ok else "PROOF_OF_BENEFIT" if proof_ok else "NOT_MET"),
        }
    # error curve for all arms on the union of frames
    with (out / "error_curve.csv").open("w", newline="") as f:
        w = csv.writer(f)
        names = list(curves.keys())
        w.writerow(["frame", "time_s"] + [f"err_{n}_m" for n in names] + [f"valid_{n}" for n in names])
        for i in range(vo["frame"].size):
            w.writerow([int(vo["frame"][i]), f"{vo['t'][i]:.3f}"]
                       + ["" if math.isnan(curves[n]["err"][i]) else f"{curves[n]['err'][i]:.4f}" for n in names]
                       + [str(bool(curves[n]["valid"][i])).lower() for n in names])
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    return result


def _recency_summary(rt_rows: list[dict]) -> dict:
    if not rt_rows:
        return {"retrievals": 0}
    ne = np.array([r["nearest_eligible_reference_m"] for r in rt_rows], dtype=float)
    na = np.array([r["nearest_any_reference_m"] for r in rt_rows], dtype=float)
    t1 = np.array([r["top1_gt_distance_m"] for r in rt_rows], dtype=float)
    q = lambda v: ([float(x) for x in np.nanquantile(v, [0.05, 0.5, 0.95])] if np.isfinite(v).any() else None)
    return {"retrievals": len(rt_rows),
            "excluded_recent_per_query_median": float(np.median([r["excluded_recent"] for r in rt_rows])),
            "n_eligible_per_query_median": float(np.median([r["n_eligible"] for r in rt_rows])),
            "queries_with_no_eligible_reference": int(sum(r["n_eligible"] == 0 for r in rt_rows)),
            "nearest_eligible_reference_m_q05_50_95": q(ne), "nearest_any_reference_m_q05_50_95": q(na),
            "top1_gt_distance_m_q05_50_95": q(t1),
            "no_competing_hypothesis": int(sum("no_competing_hypothesis" in r["why"] for r in rt_rows)),
            "ambiguous_region": int(sum("ambiguous_region" in r["why"] for r in rt_rows)),
            "weak_match": int(sum("weak_match" in r["why"] for r in rt_rows)),
            "west_incomplete_or_refused": int(sum("west_" in r["why"] for r in rt_rows)),
            "accepted": int(sum(r["outcome"] == "accepted" for r in rt_rows))}


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(float(o)) else float(o)
    raise TypeError(str(type(o)))


def events_only(dataset: Path, int_run: Path, out: Path, wrong_place_m: float, severe_harm_m: float,
                genuine_m: float) -> dict:
    gt = load_gt(dataset)
    events = read_csv(int_run / "alignment_events.csv")
    re_rows = reanchor_table(gt, events, wrong_place_m, severe_harm_m, genuine_m)
    rt_rows = retrieval_table(gt, events)
    drift = [r for r in re_rows if r["recovery_case"] == "drift_correction"]
    out.mkdir(parents=True, exist_ok=True)
    res = {"dataset": str(dataset), "run": str(int_run),
           "reanchors": {"total": len(re_rows), "drift_correction": len(drift),
                         "genuine_revisit": sum(r["genuine_revisit"] for r in re_rows),
                         "wrong_place": sum(r["wrong_place"] for r in re_rows),
                         "beneficial": sum(r["beneficial"] for r in drift), "harmful": sum(r["harmful"] for r in drift),
                         "severely_harmful": sum(r["severely_harmful"] for r in drift),
                         "gt_query_to_reference_m": [r["gt_query_to_reference_m"] for r in re_rows],
                         "delta_e_m": [r["delta_e_m"] for r in drift]},
           "events": event_counts(events), "recency_density": _recency_summary(rt_rows)}
    (out / "events_metrics.json").write_text(json.dumps(res, indent=2, default=_json_default) + "\n")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--vo-only", type=Path, default=None)
    ap.add_argument("--int", dest="int_runs", action="append", default=[],
                    help="NAME=runs/<dir>, repeatable (e.g. int_c0=runs/exp-int-001-...-int-c0)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--wrong-place-m", type=float, default=50.0)
    ap.add_argument("--severe-harm-m", type=float, default=15.0)
    ap.add_argument("--genuine-m", type=float, default=20.0)
    ap.add_argument("--material-pct", type=float, default=MATERIAL_IMPROVEMENT_PCT)
    ap.add_argument("--events-only", action="store_true")
    a = ap.parse_args(argv)
    runs = {}
    for spec in a.int_runs:
        name, _, p = spec.partition("=")
        runs[name if p else Path(spec).name] = Path(p or spec)
    if a.events_only:
        if len(runs) != 1:
            raise SystemExit("--events-only takes exactly one --int")
        res = events_only(a.dataset, next(iter(runs.values())), a.out, a.wrong_place_m, a.severe_harm_m, a.genuine_m)
        print(json.dumps(res["reanchors"], default=_json_default))
        return 0
    if a.vo_only is None or not runs:
        raise SystemExit("--vo-only and at least one --int are required")
    res = compare(a.dataset, a.vo_only, runs, a.out, a.wrong_place_m, a.severe_harm_m, a.genuine_m, a.material_pct)
    for name, arm in res["arms"].items():
        v, i = arm["vo_only_on_common_frames"], arm["int_on_common_frames"]
        print(f"{name}: ATE RMSE VO {v['ate_rmse_m']:.2f} m -> INT {i['ate_rmse_m']:.2f} m "
              f"({arm['ate_change_pct_negative_is_better']:+.1f} %); final {v['final_m']:.2f} -> {i['final_m']:.2f} m; "
              f"reanchors {arm['reanchors']}; verdict(mechanical) {arm['verdict_mechanical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
