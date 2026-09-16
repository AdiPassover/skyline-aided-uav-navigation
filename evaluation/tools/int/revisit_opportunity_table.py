"""Why each ground-truth revisit did or did not produce an accepted correction — one row per skyline
capture inside a revisit event, joining the GT geometry (`gt_revisit_events.py`) with what the
layer actually did on that capture (`alignment_frames.csv`, `alignment_events.csv`).

Per capture: the query frame/time; the true separation from the prior pass and the elapsed time;
the VO_ONLY arm's GT error and the INT arm's own pre-query error at that frame; the memory as it
stood (references inserted before the frame, with their stored and true positions): nearest
reference of any age and nearest *eligible* reference (older than the recency horizon), both as true
distances; whether a request was pending, whether the retry gap was blocking, whether a retrieval
executed; the North / West top-1 ids with their true distances and scores, the fused top score, the
margin, competitor existence, dual agreement; the selected candidate and its true distance; the
gate's verdict and its first reason; whether an ACCEPT happened and its ``Δe_0``; and the downstream
effect, ``e_INT − e_VO`` ten seconds later. Captures that were never queried say why (no pending
request / retry gap / translation unusable / no valid North view).

Run from the repository root::

    python evaluation/tools/int/revisit_opportunity_table.py --dataset datasets/<id> --vo-only runs/<vo-only run> \\
        --int runs/<int run> --events <dir>/gt_revisit_events.json --out <dir> [--recency-s 30] [--margin-s 2]
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


def build(dataset: Path, vo_run: Path, int_run: Path, events_json: Path, out: Path, recency_s: float,
          margin_s: float, downstream_s: float) -> list[dict]:
    gt = ev.load_gt(dataset)
    f0 = min(gt); _, e0, n0 = gt[f0]
    gtr = {f: (v[0], v[1] - e0, v[2] - n0) for f, v in gt.items()}          # registered GT
    vo_rows = ev.read_csv(vo_run / "alignment_frames.csv")
    in_rows = ev.read_csv(int_run / "alignment_frames.csv")
    events = ev.read_csv(int_run / "alignment_events.csv")
    ev_json = json.loads(events_json.read_text())
    vo_by = {int(r["frame_index"]): r for r in vo_rows}
    in_by = {int(r["frame_index"]): r for r in in_rows}
    by_kind = {}
    for r in events:
        by_kind.setdefault(r["kind"], {}).setdefault(int(r["frame_index"]), []).append(r)
    inserted = sorted([(int(r["frame_index"]), int(r["reference_id"]), float(r["timestamp_s"]),
                        ev._f(r["global_after_east_m"]), ev._f(r["global_after_north_m"]))
                       for rr in by_kind.get("reference_inserted", {}).values() for r in rr])

    def err(row):
        if row is None or row["global_position_valid"].strip().lower() != "true":
            return float("nan")
        f = int(row["frame_index"]); _, ge, gn = gtr[f]
        return math.hypot(ge - ev._f(row["global_east_m"]), gn - ev._f(row["global_north_m"]))

    def gt_dist(f_a, f_b):
        return math.hypot(gtr[f_a][1] - gtr[f_b][1], gtr[f_a][2] - gtr[f_b][2])

    frames_sorted = sorted(gtr)
    capture_frames = sorted({int(r["frame_index"]) for r in in_rows if r["skyline_present"].strip().lower() == "true"})
    rows = []
    for e in ev_json["events"]:
        t0 = e["later_pass"]["time_start_s"] - margin_s; t1 = e["later_pass"]["time_end_s"] + margin_s
        for qf in capture_frames:
            tq = gtr[qf][0]
            if not (t0 <= tq <= t1):
                continue
            a = in_by[qf]
            # true separation from the prior pass at this frame (nearest GT frame older than the recency horizon)
            older = [f for f in frames_sorted if gtr[f][0] <= tq - recency_s]
            if older:
                d = [gt_dist(qf, f) for f in older]
                k = int(np.argmin(d)); sep, prior_f = d[k], older[k]
            else:
                sep, prior_f = float("nan"), None
            mem = [m for m in inserted if m[0] < qf]
            any_d = [gt_dist(qf, m[0]) for m in mem]
            elig = [(m, gt_dist(qf, m[0])) for m in mem if m[2] <= tq - recency_s]
            ret = by_kind.get("retrieval", {}).get(qf, [None])[0]
            att = by_kind.get("attempt", {}).get(qf, [None])[0]
            rea = by_kind.get("reanchor", {}).get(qf, [None])[0]
            ref_frame_of = {m[1]: m[0] for m in inserted}

            def top1(r, key):
                if r is None or not r.get(key, "").strip():
                    return None, float("nan"), float("nan")
                ids = [int(x) for x in r[key].split(";") if x.strip()]
                sc = [float(x) for x in r[key.replace("_ids", "_scores")].split(";") if x.strip()]
                rid = ids[0]
                return rid, (gt_dist(qf, ref_frame_of[rid]) if rid in ref_frame_of else float("nan")), sc[0]

            n_id, n_d, n_s = top1(ret, "top_k_ids")
            w_id, w_d, w_s = top1(ret, "west_top_k_ids")
            sel = None; sel_d = float("nan")
            if att is not None and att.get("reference_id", "").strip():
                sel = int(att["reference_id"]); sel_d = gt_dist(qf, ref_frame_of[sel]) if sel in ref_frame_of else float("nan")
            if ret is None:
                if a["north_valid"].strip().lower() != "true":
                    why_not = "north_invalid"
                elif a["translation_usable"].strip().lower() != "true":
                    why_not = "translation_unusable"
                elif a["request_pending"].strip().lower() != "true":
                    why_not = "no_request_pending"
                elif a["retry_gap_blocking"].strip().lower() == "true":
                    why_not = "retry_gap_blocking"
                else:
                    why_not = "not_executed_other"
            else:
                why_not = ""
            # downstream: e_INT − e_VO downstream_s later
            later = [f for f in frames_sorted if gtr[f][0] >= tq + downstream_s]
            fl = later[0] if later else frames_sorted[-1]
            e_vo_now, e_int_now = err(vo_by.get(qf)), err(in_by.get(qf))
            e_vo_l, e_int_l = err(vo_by.get(fl)), err(in_by.get(fl))
            delta_e0 = float("nan")
            if rea is not None:
                # The alignment_frames.csv row of an accept frame is written from the POST-re-anchor output,
                # so the pre-query error must come from the reanchor row's global_before (drift correction) —
                # and does not exist at all for a hard-loss recovery (no fabricated e_before).
                _, ge, gn = gtr[qf]
                if rea["recovery_case"] == "drift_correction":
                    gb = (ev._f(rea["global_before_east_m"]), ev._f(rea["global_before_north_m"]))
                    ga = (ev._f(rea["global_after_east_m"]), ev._f(rea["global_after_north_m"]))
                    e_int_now = math.hypot(ge - gb[0], gn - gb[1])
                    delta_e0 = math.hypot(ge - ga[0], gn - ga[1]) - e_int_now
                else:
                    e_int_now = float("nan")
            rows.append({
                "event_id": e["event_id"], "query_frame": qf, "query_time_s": tq,
                "true_revisit_separation_m": sep, "prior_pass_frame": prior_f,
                "elapsed_since_prior_pass_s": (tq - gtr[prior_f][0]) if prior_f is not None else float("nan"),
                "vo_only_error_m": e_vo_now, "int_error_before_query_m": e_int_now,
                "memory_size": len(mem), "nearest_reference_any_age_m": min(any_d) if any_d else float("nan"),
                "n_eligible": len(elig), "nearest_eligible_reference_m": min((d for _, d in elig), default=float("nan")),
                "nearest_eligible_reference_id": (min(elig, key=lambda x: x[1])[0][1] if elig else None),
                "request_pending": a["request_pending"], "request_cause": a["request_cause"],
                "retry_gap_blocking": a["retry_gap_blocking"], "north_valid": a["north_valid"], "west_valid": a["west_valid"],
                "translation_usable": a["translation_usable"],
                "search_executed": ret is not None, "why_not_executed": why_not,
                "excluded_recent": (len([x for x in ret["excluded_recent_ids"].split(";") if x.strip()]) if ret else None),
                "considered": (int(float(ret["considered_count"] or 0)) if ret else None),
                "north_top1_id": n_id, "north_top1_true_distance_m": n_d, "north_top1_score": n_s,
                "west_top1_id": w_id, "west_top1_true_distance_m": w_d, "west_top1_score": w_s,
                "fused_top_score": ev._f(ret["top_region_score"]) if ret else float("nan"),
                "competing_score": ev._f(ret["competing_region_score"]) if ret else float("nan"),
                "region_margin": ev._f(ret["region_margin"]) if ret else float("nan"),
                "competitor_exists": ret["competitor_exists"] if ret else "",
                "agreement": ret["agreement"] if ret else "", "dual_status": ret["dual_status"] if ret else "",
                "selected_reference_id": sel, "selected_reference_true_distance_m": sel_d,
                "verdict": att["verdict"] if att else "", "outcome": att["reason"] if att else "",
                "why": (att["detail"] or "").split(";")[0] if att else "",
                "accepted": rea is not None, "delta_e_0_m": delta_e0,
                "position_jump_m": ev._f(rea["position_jump_m"]) if rea else float("nan"),
                f"downstream_frame_{downstream_s:g}s": fl,
                f"int_minus_vo_error_{downstream_s:g}s_later_m": (e_int_l - e_vo_l) if np.isfinite(e_int_l) and np.isfinite(e_vo_l) else float("nan"),
            })
    out.mkdir(parents=True, exist_ok=True)
    with (out / "revisit_opportunities.csv").open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows:
                w.writerow({k: ("" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else v))
                            for k, v in r.items()})
        else:
            f.write("event_id\n")
    (out / "revisit_opportunities.json").write_text(json.dumps(rows, indent=2, default=ev._json_default) + "\n")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--vo-only", required=True, type=Path)
    ap.add_argument("--int", dest="int_run", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--recency-s", type=float, default=30.0)
    ap.add_argument("--margin-s", type=float, default=2.0)
    ap.add_argument("--downstream-s", type=float, default=10.0)
    a = ap.parse_args(argv)
    rows = build(a.dataset, a.vo_only, a.int_run, a.events, a.out, a.recency_s, a.margin_s, a.downstream_s)
    print(f"{len(rows)} capture rows across revisit events")
    for r in rows:
        print(f"  E{r['event_id']} f{r['query_frame']} ({r['query_time_s']:.1f}s) sep {r['true_revisit_separation_m']:.1f} m "
              f"VOerr {r['vo_only_error_m']:.1f} | mem {r['memory_size']} any {r['nearest_reference_any_age_m']:.1f} "
              f"elig {r['n_eligible']} nearest {r['nearest_eligible_reference_m']:.1f} | "
              f"{'RETR' if r['search_executed'] else 'no:' + r['why_not_executed']} "
              f"N {r['north_top1_id']}@{r['north_top1_true_distance_m']:.1f}m/{r['north_top1_score']:.3f} "
              f"W {r['west_top1_id']}@{r['west_top1_true_distance_m']:.1f}m/{r['west_top1_score']:.3f} "
              f"fused {r['fused_top_score']:.3f} margin {r['region_margin']:.3f} comp {r['competitor_exists']} agree {r['agreement']} "
              f"-> {r['verdict']} {r['why']} {'ACCEPT de0=' + format(r['delta_e_0_m'], '+.2f') if r['accepted'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
