"""Experiment summaries for the final-validation feature (Experiments 1–3 and 5–7).

Measurement only: everything here reads committed artifacts (the extended index, result records,
query sidecars, reference sets) and writes CSV/JSON tables under ``evaluations/sim-ext-exp*``.
The unmodified naveval evaluator remains the metric authority for retrieval; these tables add the
per-query joins (rank, margins, aliasing, pose-defined correctness) that the report and the
acceptance layer need, exactly as the pilot's ``final_report.rank1_rows`` did.

Population discipline: every entry point takes ``population`` and touches only that population's
levels; ``final`` callers are guarded upstream (``ext.guard_population``).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from hsreloc.simret import ext, extindex, report, sets
from hsreloc.simret.geometry import assign_bin

EXTREPORT_VERSION = "1.0.0"

CORRECT_MATCH_TOLERANCE_M = 1.0     # a candidate "is" a reference when within 1 m of its position
DISPLACEMENT_BINS_M = [0.0, 1.0, 7.5, 17.5, 37.5, 75.0, 150.0]


def _resolve(base: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def _out_dir(cfg: dict, base: Path, key: str, population: str) -> Path:
    d = _resolve(base, cfg["experiments"][key]) / population
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_rows(cfg: dict, base: Path, population: str) -> list:
    idx = extindex.load_index(_resolve(base, cfg["index_out"]))
    wanted = set(ext.levels_for(cfg, population))
    return [r for r in idx["rows"] if r["level"] in wanted]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    cols: list = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


# --------------------------------------------------------------------------------------------------
# Experiment 1 — extraction accuracy vs exact GT
# --------------------------------------------------------------------------------------------------

def exp_extraction(cfg: dict, base: Path, population: str) -> dict:
    ctx = ext.context(cfg, base)
    srcs = ext.build_sources(cfg, base, ctx)
    rows = [r for r in _index_rows(cfg, base, population) if r["kind"] == "anchor_capture"]
    out_root = _out_dir(cfg, base, "exp1_out", population)
    result = {}
    for lvl, block in sorted(ext.levels_for(cfg, population).items()):
        key = block["task_key"]
        members = [r for r in rows if r["level"] == lvl]
        ids = sorted(r["observation_id"] for r in members)
        meta = {r["observation_id"]: r for r in members}
        labels = {i: f"{meta[i]['time_of_day']}+{meta[i]['clouds']}" for i in ids}
        for skey in ("segformer", "dp"):
            acc = report.extraction_accuracy(srcs["sim_exact"], srcs[skey], ids, ctx["height"],
                                             conditions=labels)
            for row in acc["rows"]:
                m = meta[row["observation_id"]]
                gt_curve = srcs["sim_exact"].get(row["observation_id"])
                pred = srcs[skey].get(row["observation_id"])
                signed = np.asarray(pred.row_per_col) - np.asarray(gt_curve.row_per_col)
                row.update(anchor_id=m["anchor_id"], time_of_day=m["time_of_day"],
                           clouds=m["clouds"], level=lvl,
                           median_abs_frac_height=row["median_abs_px"] / float(ctx["height"]),
                           p95_abs_px=float(np.percentile(np.abs(signed), 95)),
                           bias_px=float(np.median(signed)))
            for slice_key in ("time_of_day", "clouds", "condition", "anchor_id"):
                acc[f"by_{slice_key}"] = report.summarise_by(
                    acc["rows"], slice_key,
                    ["median_abs_px", "p90_abs_px", "p95_abs_px", "catastrophic_frac", "bias_px"])
            # refusal slices: which conditions the extractor declines
            ref_by_cond: dict = {}
            for r in acc["refusals"]:
                ref_by_cond.setdefault(labels.get(r["observation_id"], "?"), 0)
                ref_by_cond[labels.get(r["observation_id"], "?")] += 1
            acc["refusals_by_condition"] = dict(sorted(ref_by_cond.items()))
            report.write_table(acc, out_root / key, f"accuracy_{skey}")
            result[f"{key}/{skey}"] = {
                "n_evaluated": acc["n_evaluated"], "n_refused": acc["n_refused"],
                "refusal_rate": acc["refusal_rate"],
                "median_abs_px": (float(np.median([r["median_abs_px"] for r in acc["rows"]]))
                                  if acc["rows"] else None)}
            print(f"[simret] exp1[{population}] {key}/{skey}: {acc['n_evaluated']} evaluated, "
                  f"{acc['n_refused']} refused, median "
                  f"{result[f'{key}/{skey}']['median_abs_px']}")
    write_json(out_root / "summary.json", {"population": population, "results": result,
                                           "extreport_version": EXTREPORT_VERSION})
    return result


# --------------------------------------------------------------------------------------------------
# Experiment 2 — same-pose 9-condition consistency
# --------------------------------------------------------------------------------------------------

def exp_consistency(cfg: dict, base: Path, population: str) -> dict:
    ctx = ext.context(cfg, base)
    srcs = ext.build_sources(cfg, base, ctx)
    out_root = _out_dir(cfg, base, "exp2_out", population)
    result = {}
    for lvl, block in sorted(ext.levels_for(cfg, population).items()):
        key = block["task_key"]
        adapted = [{
            "condition_pose_group": f"{lvl}|{r['anchor_id']}",
            "condition_time_of_day": r["time_of_day"], "condition_clouds": r["clouds"],
            "anchor_id": r["anchor_id"], "observation_id": r["observation_id"],
        } for r in _index_rows(cfg, base, population)
            if r["kind"] == "anchor_capture" and r["level"] == lvl]
        res = report.condition_consistency(adapted, srcs, ctx["height"],
                                           catastrophic_frac=float(
                                               cfg["experiments"]["catastrophic_frac_height"]))
        # pair classes: TIME-held (same clouds), CLOUD-held (same time), mixed
        for r in res["rows"]:
            a_t, b_t = r["tod_pair"].split(" vs ")
            a_c, b_c = r["clouds_pair"].split(" vs ")
            r["pair_class"] = ("time_held" if a_t == b_t else
                               "clouds_held" if a_c == b_c else "mixed")
        ok = [r for r in res["rows"] if r["status"] == "ok"]
        res["by_pair_class"] = {
            src: report.summarise_by([r for r in ok if r["source"] == src], "pair_class",
                                     ["curve_median_abs_px", "curve_median_abs_offset_removed_px",
                                      "curve_offset_px"])
            for src in srcs}
        gt_nonzero = [
            {k: r[k] for k in ("observation_a", "observation_b", "pair", "curve_median_abs_px")}
            for r in ok if r["source"] == "sim_exact" and r["curve_median_abs_px"] > 0.0]
        res["gt_nonzero_pairs"] = gt_nonzero
        report.write_table(res, out_root / key, "consistency")
        result[key] = {"n_pose_groups": res["n_pose_groups"],
                       "catastrophic_counts": res["catastrophic_counts"],
                       "n_gt_nonzero_pairs": len(gt_nonzero),
                       "refusal_counts": _refusal_counts(res["rows"])}
        if gt_nonzero:
            print(f"[simret] exp2[{population}] {key}: *** GT SAME-POSE VARIATION on "
                  f"{len(gt_nonzero)} pair(s) — investigate before any extraction conclusion "
                  f"(spec FR-005); worst {max(g['curve_median_abs_px'] for g in gt_nonzero)} px")
        else:
            print(f"[simret] exp2[{population}] {key}: GT invariance exact "
                  f"({res['n_pairs_per_source'].get('sim_exact', 0)} pairs, 0 px)")
    write_json(out_root / "summary.json", {"population": population, "results": result,
                                           "extreport_version": EXTREPORT_VERSION})
    return result


def _refusal_counts(rows: list) -> dict:
    out: dict = {}
    for r in rows:
        if r["status"] != "ok":
            out[r["source"]] = out.get(r["source"], 0) + 1
    return out


# --------------------------------------------------------------------------------------------------
# per-query rank rows (the join every retrieval summary and the acceptance layer share)
# --------------------------------------------------------------------------------------------------

def rank_rows(cfg: dict, base: Path, kind: str, task_key: str, source_key: str,
              spacing: float = None, baseline: str = "ncc") -> list:
    spacing = float(spacing if spacing is not None else cfg["tasks"]["spacings_m"][0])
    name = ext.task_name(kind, task_key)
    run_id = ext.run_id_for(cfg, name, spacing, baseline, source_key)
    record_dir = _resolve(base, cfg["runs_root"]) / run_id
    qs_dir = _resolve(base, cfg["datasets_root"]) / sets.query_set_id(name, spacing, source_key)
    rs_dir = _resolve(base, cfg["refdb_root"]) / sets.reference_set_id(name, spacing, source_key)
    return rank_rows_from_dirs(record_dir, qs_dir, rs_dir, task=name, source=source_key)


def rank_rows_from_dirs(record_dir: Path, qs_dir: Path, rs_dir: Path, task: str,
                        source: str) -> list:
    """The record/sidecar/reference join by explicit directories — also loads pilot records."""
    name, source_key, run_id = task, source, Path(record_dir).name
    refs = list(csv.DictReader((rs_dir / "references.csv").open(encoding="utf-8")))
    ref_pos = {r["reference_id"]: (float(r["east_m"]), float(r["north_m"])) for r in refs}
    holes = {h["reference_id"]
             for h in json.loads((rs_dir / "manifest.json").read_text(encoding="utf-8"))
             .get("reference_source", {}).get("database_holes", [])}

    sidecar = list(csv.DictReader((qs_dir / "skyline_queries.csv").open(encoding="utf-8")))
    record = {int(r["query_id"]): r
              for r in csv.DictReader((record_dir / "queries.csv").open(encoding="utf-8"))}
    if len(record) != len(sidecar):
        raise report.ReportError(f"{run_id}: record has {len(record)} queries, sidecar "
                                 f"{len(sidecar)} — not the same run")

    def _f(v):
        return None if v in ("", None) else float(v)

    rows = []
    for q in sidecar:
        qi = int(q["query_index"])
        rr = record[qi]
        correct = q["condition_nearest_reference_id"]
        # a hole's position is deliberately absent from references.csv; the correct reference can
        # still be a hole for this source — then no candidate can ever match it (rank None)
        cands = []
        for k in range(5):
            e, n, s = _f(rr.get(f"cand{k}_east_m")), _f(rr.get(f"cand{k}_north_m")), \
                _f(rr.get(f"cand{k}_score"))
            if e is None or s is None:
                break
            rid = min(ref_pos, key=lambda i: math.hypot(ref_pos[i][0] - e, ref_pos[i][1] - n))
            if math.hypot(ref_pos[rid][0] - e, ref_pos[rid][1] - n) > CORRECT_MATCH_TOLERANCE_M:
                rid = None
            cands.append({"reference_id": rid, "score": s})
        rank = next((k + 1 for k, c in enumerate(cands) if c["reference_id"] == correct), None)
        correct_score = next((c["score"] for c in cands if c["reference_id"] == correct), None)
        best_incorrect = next((c["score"] for c in cands if c["reference_id"] != correct), None)
        in_cov = q["in_coverage"].lower() == "true"
        attainable = in_cov and correct not in holes
        accepted = rr["outcome"] == "SUCCESS"
        top1 = cands[0]["reference_id"] if cands else None
        rows.append({
            "task": name, "source": source_key, "query_index": qi,
            "observation_id": q["condition_observation_id"], "session_id": q["traversal_id"],
            "anchor_id": q.get("condition_anchor_id") or "",
            "time_of_day": q.get("condition_time_of_day") or "",
            "clouds": q.get("condition_clouds") or "",
            "nearest_reference_id": correct,
            "nearest_reference_distance_m": _f(q["condition_translation_m"]),
            "along_m": _f(q["condition_along_m"]), "lateral_m": _f(q["condition_lateral_m"]),
            "up_diff_m": _f(q["condition_up_diff_m"]),
            "translation_bin": q["condition_translation_bin"] or None,
            "stage": q["condition_stage"], "in_coverage": in_cov, "attainable": attainable,
            "correct_is_hole": correct in holes,
            "outcome": rr["outcome"], "accepted": accepted,
            "best_score": _f(rr["best_score"]), "second_best_score": _f(rr["second_best_score"]),
            "score_margin": _f(rr["score_margin"]),
            "top1_reference_id": top1,
            "top1_correct": bool(cands) and top1 == correct,
            "rank_of_correct": rank, "correct_score": correct_score,
            "best_incorrect_score": best_incorrect,
            "margin_correct_minus_best_incorrect": (
                None if correct_score is None or best_incorrect is None
                else correct_score - best_incorrect),
        })
    return rows


def _recall_at(rows: list, k: int) -> float | None:
    att = [r for r in rows if r["attainable"]]
    if not att:
        return None
    return sum(1 for r in att if r["rank_of_correct"] is not None and r["rank_of_correct"] <= k) \
        / len(att)


def _wilson_low(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - rad) / denom)


def _retrieval_summary(rows: list, n_references: int) -> dict:
    att = [r for r in rows if r["attainable"]]
    unatt = [r for r in rows if not r["attainable"]]
    accepted = [r for r in att if r["accepted"]]
    conf_false_strict = [r for r in rows if r["accepted"]
                         and not (r["attainable"] and r["top1_correct"])]
    margins = [r["margin_correct_minus_best_incorrect"] for r in att
               if r["margin_correct_minus_best_incorrect"] is not None]
    aliases: dict = {}
    for r in att:
        if not r["top1_correct"] and r["top1_reference_id"]:
            k = f"{r['nearest_reference_id']} -> {r['top1_reference_id']}"
            aliases[k] = aliases.get(k, 0) + 1
    return {
        "n_queries": len(rows), "n_attainable": len(att), "n_unattainable": len(unatt),
        "n_extraction_failures": sum(1 for r in rows if r["outcome"] == "EXTRACTION_FAILURE"),
        "n_references": n_references,
        "chance_recall_at_1": (1.0 / n_references) if n_references else None,
        "recall_at_1": _recall_at(rows, 1), "recall_at_3": _recall_at(rows, 3),
        "recall_at_5": _recall_at(rows, 5),
        "rank_counts": {str(k): sum(1 for r in att if r["rank_of_correct"] == k)
                        for k in (1, 2, 3, 4, 5)} | {
            "none": sum(1 for r in att if r["rank_of_correct"] is None)},
        "margin_correct_minus_best_incorrect": (
            {"median": float(np.median(margins)), "p10": float(np.percentile(margins, 10)),
             "p90": float(np.percentile(margins, 90))} if margins else None),
        "n_accepted_of_attainable": len(accepted),
        "accepted_precision": (sum(1 for r in accepted if r["top1_correct"]) / len(accepted)
                               if accepted else None),
        "n_confident_false_strict": len(conf_false_strict),
        "top_aliases": dict(sorted(aliases.items(), key=lambda kv: -kv[1])[:10]),
    }


# --------------------------------------------------------------------------------------------------
# Experiment 3 — exact-pose discriminability
# --------------------------------------------------------------------------------------------------

def exp_exactpose(cfg: dict, base: Path, population: str) -> dict:
    tasks = ext.load_all_tasks(cfg, base)
    out_root = _out_dir(cfg, base, "exp3_out", population)
    result = {}
    for kind, key, skey in ext.cells_for(cfg, population):
        if kind != "exactpose":
            continue
        rows = rank_rows(cfg, base, kind, key, skey)
        n_refs = tasks[ext.task_name(kind, key)]["grids"][
            f"{float(cfg['tasks']['spacings_m'][0]):g}"]["n_references"]
        summary = _retrieval_summary(rows, n_refs)
        summary["by_condition"] = {}
        for cond in sorted({f"{r['time_of_day']}+{r['clouds']}" for r in rows}):
            members = [r for r in rows if f"{r['time_of_day']}+{r['clouds']}" == cond]
            s = _retrieval_summary(members, n_refs)
            summary["by_condition"][cond] = {
                "n_attainable": s["n_attainable"], "recall_at_1": s["recall_at_1"],
                "recall_at_1_wilson_low": (None if s["recall_at_1"] is None else
                                           _wilson_low(s["recall_at_1"], s["n_attainable"])),
                "n_confident_false_strict": s["n_confident_false_strict"]}
        write_rows_csv(out_root / f"{key}_{skey}_rank_rows.csv", rows)
        write_json(out_root / f"{key}_{skey}_summary.json", summary)
        result[f"{key}/{skey}"] = {"recall_at_1": summary["recall_at_1"],
                                   "n_attainable": summary["n_attainable"],
                                   "n_references": n_refs,
                                   "chance": summary["chance_recall_at_1"]}
        print(f"[simret] exp3[{population}] {key}/{skey}: R@1 {summary['recall_at_1']} "
              f"({summary['n_attainable']} attainable, {n_refs} refs, "
              f"chance {summary['chance_recall_at_1']:.3f})")
    write_json(out_root / "summary.json", {"population": population, "results": result,
                                           "extreport_version": EXTREPORT_VERSION})
    return result


# --------------------------------------------------------------------------------------------------
# Experiment 5 — swipe retrieval vs physical displacement
# --------------------------------------------------------------------------------------------------

def _radius(rows: list, level_key: str, threshold: float, min_n: int = 30) -> dict:
    """Largest displacement-bin upper edge whose cumulative attainable R@1 stays >= threshold.

    A radius never extends across an *empty* marginal bin: with no samples between two edges the
    cumulative recall is unchanged, and claiming the larger edge would be extrapolation over
    unsampled range — the loop stops at the last sampled edge instead.
    """
    att = [r for r in rows if r["attainable"] and r["nearest_reference_distance_m"] is not None]
    best, prev = None, DISPLACEMENT_BINS_M[0]
    for edge in DISPLACEMENT_BINS_M[1:]:
        marginal = [r for r in att if prev < r["nearest_reference_distance_m"] <= edge]
        cumulative = [r for r in att if r["nearest_reference_distance_m"] <= edge]
        if not marginal and best is not None:
            break
        prev = edge
        if len(cumulative) < min_n:
            continue
        r1 = sum(1 for m in cumulative if m["top1_correct"]) / len(cumulative)
        if r1 >= threshold:
            best = {"radius_m": edge, "recall_at_1": r1, "n": len(cumulative)}
    return best or {"radius_m": None,
                    "note": f"no bin with >= {min_n} attainable queries sustains "
                            f"R@1 >= {threshold} for {level_key}"}


def exp_swipes(cfg: dict, base: Path, population: str) -> dict:
    tasks = ext.load_all_tasks(cfg, base)
    idx = {r["observation_id"]: r for r in _index_rows(cfg, base, population)}
    out_root = _out_dir(cfg, base, "exp5_out", population)
    result = {}
    for kind, key, skey in ext.cells_for(cfg, population):
        if kind != "swipes":
            continue
        rows = rank_rows(cfg, base, kind, key, skey)
        for r in rows:
            m = idx.get(r["observation_id"], {})
            r["phase"] = m.get("phase") or ""
            r["leg_axis"] = m.get("leg_axis") or ""
            r["leg_direction"] = m.get("leg_direction") or ""
            r["hard_tag"] = bool(m.get("hard_tag"))
            r["total_displacement_m"] = m.get("total_displacement_m")
            r["displacement_bin"] = (None if m.get("total_displacement_m") is None else
                                     assign_bin(m["total_displacement_m"], DISPLACEMENT_BINS_M))
            r["nearest_ref_bin"] = (None if r["nearest_reference_distance_m"] is None else
                                    assign_bin(r["nearest_reference_distance_m"],
                                               DISPLACEMENT_BINS_M))
            r["up_diff_bin"] = (None if r["up_diff_m"] is None else
                                assign_bin(abs(r["up_diff_m"]), DISPLACEMENT_BINS_M))
        n_refs = tasks[ext.task_name(kind, key)]["grids"][
            f"{float(cfg['tasks']['spacings_m'][0]):g}"]["n_references"]
        primary = [r for r in rows if not r["hard_tag"]]
        summary = {
            "overall": _retrieval_summary(rows, n_refs),
            "primary_excluding_hard": _retrieval_summary(primary, n_refs),
            "by_nearest_ref_bin": _slice_recall(primary, "nearest_ref_bin"),
            "by_nearest_ref_bin_level_flight": _slice_recall(
                [r for r in primary if r["up_diff_m"] is not None and abs(r["up_diff_m"]) <= 7.5],
                "nearest_ref_bin"),
            "by_up_diff_bin": _slice_recall(primary, "up_diff_bin"),
            "by_displacement_bin": _slice_recall(primary, "displacement_bin"),
            "by_leg_axis": _slice_recall(primary, "leg_axis"),
            "by_leg_direction": _slice_recall(primary, "leg_direction"),
            "by_phase": _slice_recall(primary, "phase"),
            "R80": _radius(primary, key, 0.8), "R90": _radius(primary, key, 0.9),
            "out_of_coverage_note": (
                "unattainable rows (no reference within tau_pos, or the correct reference is a "
                "database hole) are excluded from every recall figure and reported as counts; "
                "HardSwipe queries far from any anchor land here by design"),
        }
        write_rows_csv(out_root / f"{key}_{skey}_rank_rows.csv", rows)
        write_json(out_root / f"{key}_{skey}_summary.json", summary)
        result[f"{key}/{skey}"] = {
            "recall_at_1": summary["primary_excluding_hard"]["recall_at_1"],
            "n_attainable": summary["primary_excluding_hard"]["n_attainable"],
            "R80": summary["R80"].get("radius_m"), "R90": summary["R90"].get("radius_m")}
        print(f"[simret] exp5[{population}] {key}/{skey}: R@1 "
              f"{summary['primary_excluding_hard']['recall_at_1']} over "
              f"{summary['primary_excluding_hard']['n_attainable']} attainable; "
              f"R80 {summary['R80'].get('radius_m')} m")
    write_json(out_root / "summary.json", {"population": population, "results": result,
                                           "extreport_version": EXTREPORT_VERSION})
    return result


def exp_deformation(cfg: dict, base: Path, population: str) -> dict:
    """Experiment 4: GT viewpoint response per swipe + outbound/return repeatability."""
    from hsreloc.matchers import build_matcher
    from hsreloc.simret import deformation

    ctx = ext.context(cfg, base)
    gt = ext.build_sources(cfg, base, ctx, keys=("sim_exact",))["sim_exact"]
    repeat_sources = ext.build_sources(cfg, base, ctx, keys=("sim_exact", "segformer"))
    matchers = {"c0_frozen_ncc": build_matcher("c0_frozen_ncc", {})}
    for variant, spec in (cfg["experiments"].get("diagnostic_matchers") or {}).items():
        matchers[variant] = build_matcher(variant, spec)
    idx_rows = _index_rows(cfg, base, population)
    out_root = _out_dir(cfg, base, "exp4_out", population)
    result = {}
    for lvl, block in sorted(ext.levels_for(cfg, population).items()):
        key = block["task_key"]
        level_rows, level_pairs, notes = [], [], {}
        for sid in block["swipe_sessions"]:
            res = deformation.swipe_deformation(sid, idx_rows, gt, ctx["height"],
                                                matchers=matchers)
            if res["center_refused"]:
                notes[sid] = f"center refused: {res['center_refused']}"
                continue
            if res["final_leg_note"]:
                notes[sid] = res["final_leg_note"]
            level_rows.extend(res["rows"])
            level_pairs.extend(deformation.repeatability(res["pairs"], repeat_sources,
                                                         ctx["height"]))
            write_rows_csv(out_root / key / f"{sid}_rows.csv", res["rows"])
        # DAY+CLOUDY swipes are condition-confounded (Experiment 7): kept, summarised separately
        clean = [r for r in level_rows if r["clouds"] == "CLEAR"]
        confounded = [r for r in level_rows if r["clouds"] != "CLEAR"]
        summaries = {}
        for axis in ("lateral", "longitudinal", "vertical"):
            members = [r for r in clean if r["leg_axis"] == axis]
            fields = (["curve_median_abs_px", "curve_median_abs_offset_removed_px",
                       "curve_offset_px", "changed_frac", "edge_changed_frac"]
                      + [f"{n}_score" for n in matchers] + [f"{n}_shift" for n in matchers]
                      + [f"{n}_scale" for n in matchers])
            summaries[axis] = report.summarise_by(members, "displacement_bin", fields)
        vertical = [r for r in clean if r["leg_axis"] == "vertical"]
        vertical_diag = {
            "n": len(vertical),
            "note": ("offset = what a constant vertical shift (and therefore mean removal) "
                     "absorbs; offset-removed = the shape change no vertical shift explains"),
            "per_bin": report.summarise_by(vertical, "displacement_bin",
                                           ["curve_offset_px", "curve_median_abs_px",
                                            "curve_median_abs_offset_removed_px"]),
        }
        write_rows_csv(out_root / key / "repeatability_pairs.csv", level_pairs)
        payload = {"population": population, "level": lvl, "n_rows": len(level_rows),
                   "n_confounded_day_cloudy_rows": len(confounded),
                   "per_direction": summaries, "vertical_diagnostic": vertical_diag,
                   "session_notes": notes,
                   "n_repeatability_pairs": len({(p['outbound_id'], p['return_id'])
                                                 for p in level_pairs}),
                   "repeatability_by_source": {
                       src: report.summarise_by(
                           [p for p in level_pairs if p["source"] == src and p["status"] == "ok"],
                           "leg_axis", ["curve_median_abs_px",
                                        "curve_median_abs_offset_removed_px", "c0_ncc"])
                       for src in repeat_sources}}
        write_json(out_root / key / "deformation_summary.json", payload)
        result[key] = {"n_rows": len(level_rows), "n_pairs": payload["n_repeatability_pairs"],
                       "notes": notes}
        print(f"[simret] exp4[{population}] {key}: {len(level_rows)} response rows "
              f"({len(confounded)} DAY+CLOUDY confounded, summarised separately), "
              f"{payload['n_repeatability_pairs']} outbound/return pairs")
    write_json(out_root / "summary.json", {"population": population, "results": result,
                                           "extreport_version": EXTREPORT_VERSION})
    return result


def exp_hard(cfg: dict, base: Path) -> dict:
    """Experiment 6: what actually makes the `hard`-tagged runs difficult (post-primary only).

    Never a verdict on the author's guess: the comparison is mechanism-by-mechanism — GT
    deformation slope, FOV content change, near-field indicators (skyline height / roughness at
    the swipe center), SegFormer error, NCC-to-correct evolution — hard vs ordinary swipes, per
    level. Requires the committed Experiment 4/5 outputs for BOTH populations (it stratifies a
    frozen primary analysis, so it refuses to run before them).
    """
    for pop in ("dev", "final"):
        marker = _resolve(base, cfg["experiments"]["exp5_out"]) / pop / "summary.json"
        if not marker.exists():
            raise report.ReportError(
                f"exp-hard runs only AFTER the primary swipe analysis is committed for every "
                f"population; missing {marker} — the hard tag must not influence primary results")
    out_root = _resolve(base, cfg["experiments"]["exp6_out"])
    out_root.mkdir(parents=True, exist_ok=True)
    ctx = ext.context(cfg, base)
    srcs = ext.build_sources(cfg, base, ctx)
    idx = extindex.load_index(_resolve(base, cfg["index_out"]))
    idx_by_id = {r["observation_id"]: r for r in idx["rows"]}

    mech_rows, retr_rows, panels = [], [], []
    for lvl, block in sorted(cfg["levels"].items()):
        key, pop = block["task_key"], block["split"]
        d4 = _resolve(base, cfg["experiments"]["exp4_out"]) / pop / key
        for f in sorted(d4.glob("Run_*_rows.csv")):
            with f.open(encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    mech_rows.append({
                        "level": key, "hard": r["hard_tag"] == "True",
                        "displacement_bin": r["displacement_bin"] or None,
                        "leg_axis": r["leg_axis"],
                        "offset_removed_px": float(r["curve_median_abs_offset_removed_px"]),
                        "changed_frac": float(r["changed_frac"]),
                        "edge_changed_frac": float(r["edge_changed_frac"]),
                        "center_roughness": float(r["curve_roughness_a"]),
                        "gt_curve_mean_row": None,
                    })
        d5 = _resolve(base, cfg["experiments"]["exp5_out"]) / pop
        for skey in ("sim_exact", "segformer"):
            p = d5 / f"{key}_{skey}_rank_rows.csv"
            if p.exists():
                for r in _load_rank_csv(p):
                    r["level"] = key
                    r["source"] = skey
                    retr_rows.append(r)

    hard = [r for r in mech_rows if r["hard"]]
    ordinary = [r for r in mech_rows if not r["hard"]]
    mech_summary = {
        "note": ("GT curves are geometry: the village `hard` run's DAY+CLOUDY condition cannot "
                 "affect its GT deformation (verified by Experiment 2 GT invariance), only its "
                 "SegFormer/retrieval rows — which are analysed under Experiment 7 instead"),
        "hard": {"n": len(hard),
                 "per_bin": report.summarise_by(hard, "displacement_bin",
                                                ["offset_removed_px", "changed_frac",
                                                 "edge_changed_frac", "center_roughness"])},
        "ordinary": {"n": len(ordinary),
                     "per_bin": report.summarise_by(ordinary, "displacement_bin",
                                                    ["offset_removed_px", "changed_frac",
                                                     "edge_changed_frac", "center_roughness"])},
    }
    hard_retr = [r for r in retr_rows if r.get("hard_tag") == "True"]
    ord_retr = [r for r in retr_rows if r.get("hard_tag") == "False"]
    retr_summary = {}
    for label, members in (("hard", hard_retr), ("ordinary", ord_retr)):
        att = [r for r in members if r["attainable"] == "True"]
        retr_summary[label] = {
            "n": len(members), "n_attainable": len(att),
            "n_out_of_coverage": len(members) - len(att),
            "recall_at_1_attainable": (sum(1 for r in att if r["top1_correct"] == "True")
                                       / len(att)) if att else None,
            "by_level": {k: {"n": sum(1 for r in members if r["level"] == k),
                             "n_attainable": sum(1 for r in att if r["level"] == k)}
                         for k in sorted({r["level"] for r in members})},
        }

    # mechanism panels — declared rule: per hard swipe, the attainable observation with the WORST
    # frozen-C0 score to its correct reference (or, if none is attainable, the observation nearest
    # any reference), rendered with all sources' curves
    for lvl, block in sorted(cfg["levels"].items()):
        hard_sessions = [s for s in block["swipe_sessions"]
                         if any(idx_by_id[o]["hard_tag"] for o in idx_by_id
                                if idx_by_id[o]["session_id"] == s)]
        for sid in hard_sessions:
            cand = [r for r in retr_rows if r["session_id"] == sid and r["source"] == "sim_exact"]
            if not cand:
                continue
            att = [r for r in cand if r["attainable"] == "True"]
            pick = (min(att, key=lambda r: float(r["best_score"] or 0.0)) if att else
                    min(cand, key=lambda r: float(r["nearest_reference_distance_m"] or 1e9)))
            rule = ("worst frozen-C0 score among attainable observations of the swipe" if att else
                    "no attainable observation — the one nearest any reference (out-of-coverage)")
            panel = _hard_panel(cfg, base, ctx, srcs, idx_by_id, pick, out_root,
                                f"panel_{block['task_key']}_{sid}.jpg", rule)
            if panel:
                panels.append(panel)

    payload = {
        "headline": (f"hard swipes: {len(hard)} mechanism rows vs {len(ordinary)} ordinary; "
                     f"retrieval: {retr_summary['hard']['n_attainable']} attainable hard queries "
                     f"of {retr_summary['hard']['n']} "
                     f"({retr_summary['hard']['n_out_of_coverage']} out-of-coverage by design)"),
        "mechanism": mech_summary, "retrieval": retr_summary,
        "panels": panels, "extreport_version": EXTREPORT_VERSION,
    }
    write_json(out_root / "hard_summary.json", payload)
    write_rows_csv(out_root / "mechanism_rows.csv", mech_rows)
    print(f"[simret] exp-hard: {len(hard)} hard rows, {len(panels)} panels -> {out_root}")
    return payload


def _load_rank_csv(path: Path) -> list:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _hard_panel(cfg, base, ctx, srcs, idx_by_id, pick, out_root, name, rule):
    from hsreloc.simret import render
    from hsreloc.simret.geometry import viewpoint_offset
    from hsreloc.observation import read_session
    oid, rid = pick["observation_id"], pick["nearest_reference_id"]
    store = ctx["store"]
    smap = ctx["smap"]
    if oid not in smap or rid not in smap:
        return None
    obs = {o.observation_id: o for s in {smap[oid], smap[rid]}
           for o in read_session(store / s)[1]}

    def curves_of(i):
        out = {}
        for skey, s in srcs.items():
            try:
                out[s.provenance] = s.get(i).row_per_col
            except Exception:
                pass
        return out

    scores = {"c0_frozen_ncc": {"score": float(pick["best_score"] or 0.0), "shift": 0.0,
                                "scale": 1.0, "warp_magnitude": 0.0,
                                "accepted": pick["accepted"] == "True"}}
    off = viewpoint_offset(obs[oid], obs[rid]).as_dict()
    render.render_pair(
        out_root / name,
        {"observation_id": oid, "image_path": store / smap[oid] / "images" / f"{oid}.png",
         "curves": curves_of(oid)},
        {"observation_id": rid, "image_path": store / smap[rid] / "images" / f"{rid}.png",
         "curves": curves_of(rid)},
        off, scores, pick["top1_correct"] == "True",
        title=f"hard-swipe mechanism: {oid} vs nearest reference {rid} "
              f"({float(pick['nearest_reference_distance_m'] or 0):.0f} m)")
    return {"file": name, "title": f"{oid} vs {rid}", "rule": rule}


def _slice_recall(rows: list, field: str) -> dict:
    out = {}
    for value in sorted({str(r[field]) for r in rows if r[field] is not None}):
        members = [r for r in rows if str(r[field]) == value]
        att = [r for r in members if r["attainable"]]
        r1 = (sum(1 for r in att if r["top1_correct"]) / len(att)) if att else None
        out[value] = {"n": len(members), "n_attainable": len(att), "recall_at_1": r1,
                      "wilson_low": None if r1 is None else _wilson_low(r1, len(att))}
    return out
