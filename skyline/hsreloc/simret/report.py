"""Measurements the simulator experiments need, kept separate from the plots (Part I).

Three questions this layer answers, none of which the evaluator answers for us because none of them is
a retrieval metric:

1. **How much does the ground-truth skyline itself deform with translation?** This is the question
   ``EXP-SKY-006`` could not ask, because ECL had no ground-truth curve. With a simulator sky mask it
   is directly measurable, and it is what decides whether a matcher variant is even the right tool:
   if the GT curves at 25 m differ by a uniform shift, C1 is the answer; if by a shift and a scale,
   C3; if by neither, no rigid alignment will help and C4 or a different representation is the only
   route. That decision must be made from *this* measurement and never from a retrieval number.
2. **How accurate is each extractor against that ground truth, per condition?** The
   ``COMP-SKY-002`` metrics, computed on the simulator's own truth — reported in their own table and
   never mixed into a retrieval number (``PROT-SKY-001`` §3.5).
3. **What does a matcher variant actually change?** Per-pair scores and recovered transform
   parameters for C0 beside any variant, so a variant's effect is always readable as "it needed this
   much shift/scale/warp", not just as a better number.

Everything here **refuses on absent input**. There is no synthetic fallback and no placeholder: an
empty table is an error, because a plausible-looking table with no data behind it is the single most
dangerous artifact this repository could produce.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from hsreloc.matchers import FrozenNccMatcher, roughness
from hsreloc.matchers.base import BaseMatcher
from hsreloc.retrieval.profile import ProfileConfig, normalize
from hsreloc.simret.geometry import assign_bin, viewpoint_offset

REPORT_VERSION = "1.0.0"


class ReportError(Exception):
    """A measurement was asked for without the data it measures."""


def _require(rows, what: str):
    if not rows:
        raise ReportError(f"no data for {what}: refusing to emit an empty table. This measurement "
                          f"needs real simulator output; nothing here fabricates one.")
    return rows


# --------------------------------------------------------------------------------------------------
# 1. ground-truth skyline deformation vs translation
# --------------------------------------------------------------------------------------------------

def curve_disagreement(a, b, height_px: int) -> dict:
    """Row-space disagreement between two full-width curves, before and after removing the offset.

    ``median_abs`` is the raw disagreement; ``median_abs_offset_removed`` is what survives after a
    constant vertical shift — the split ``EXP-SKY-004``/``005`` used, kept identical here so simulator
    numbers sit on the same axis as the ECL ones.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ReportError(f"curve shapes differ: {a.shape} vs {b.shape}")
    d = b - a
    return {"median_abs_px": float(np.median(np.abs(d))),
            "p90_abs_px": float(np.percentile(np.abs(d), 90)),
            "offset_px": float(np.median(d)),
            "median_abs_offset_removed_px": float(np.median(np.abs(d - np.median(d)))),
            "median_abs_frac_height": float(np.median(np.abs(d)) / float(height_px)),
            "roughness_a": roughness(a), "roughness_b": roughness(b)}


def deformation_vs_translation(tasks: dict, observations: dict, source, height_px: int,
                               translation_bins_m=None, matchers: dict | None = None) -> dict:
    """For every query–reference pair in a task set: the pose offset and the curve disagreement.

    With ``source`` = the ``oracle:sim_exact`` source this is **GT-vs-GT deformation**, the E3
    measurement. With an automatic source it is that source's own deformation, which is a different
    and separately reported quantity.

    ``matchers`` optionally adds each variant's score and recovered transform for the same pair, so
    "would a bounded shift have absorbed this?" is answerable from one table.
    """
    from hsreloc.retrieval.skyline_curve import CurveError

    matchers = dict(matchers or {"c0_frozen_ncc": FrozenNccMatcher()})
    profile_config = ProfileConfig()
    rows, refused = [], []
    cache: dict = {}

    def profile_of(oid):
        if oid not in cache:
            cache[oid] = normalize(source.get(oid), profile_config)
        return cache[oid]

    for q in _require(tasks["queries"], "deformation_vs_translation"):
        qid, rid = q["observation_id"], q["nearest_reference_id"]
        try:
            qc, rc_ = source.get(qid), source.get(rid)
            qp, rp = profile_of(qid), profile_of(rid)
        except CurveError as exc:
            refused.append({"query": qid, "reference": rid, "reason": str(exc)})
            continue
        offset = viewpoint_offset(observations[qid], observations[rid])
        row = {"query_id": qid, "reference_id": rid, "spacing_m": float(q["spacing_m"]),
               "stage": q["condition_stage"], **offset.as_dict(),
               "translation_bin": assign_bin(offset.translation_m, list(translation_bins_m))
               if translation_bins_m else q["condition_translation_bin"],
               **{f"curve_{k}": v for k, v in
                  curve_disagreement(rc_.row_per_col, qc.row_per_col, height_px).items()}}
        for name, m in matchers.items():
            res = m.match(qp, rp)
            row[f"{name}_score"] = res.score
            row[f"{name}_shift"] = res.shift
            row[f"{name}_scale"] = res.scale
            row[f"{name}_warp"] = res.warp_magnitude
            row[f"{name}_accepted"] = res.accepted
        rows.append(row)
    _require(rows, "deformation_vs_translation (every pair was refused by the curve source)")
    return {"rows": rows, "refused": refused,
            "summary": summarise_by(rows, "translation_bin",
                                    ["curve_median_abs_px", "curve_median_abs_offset_removed_px",
                                     "curve_offset_px"] +
                                    [f"{n}_score" for n in matchers]),
            "source": source.describe(), "report_version": REPORT_VERSION}


def summarise_by(rows: list, key: str, fields: list) -> dict:
    """Median, p10/p90 and count of each field, grouped by ``key``. Groups keep their own n."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(str(r.get(key)), []).append(r)
    out = {}
    for name, members in sorted(groups.items()):
        stats = {"n": len(members)}
        for f in fields:
            values = [m[f] for m in members if m.get(f) is not None and np.isfinite(m[f])]
            if values:
                stats[f] = {"median": float(np.median(values)),
                            "p10": float(np.percentile(values, 10)),
                            "p90": float(np.percentile(values, 90))}
        out[name] = stats
    return out


# --------------------------------------------------------------------------------------------------
# 2. extraction accuracy against the simulator's own truth — its own table, never a retrieval number
# --------------------------------------------------------------------------------------------------

def extraction_accuracy(gt_source, pred_source, observation_ids: list, height_px: int,
                        conditions: dict | None = None) -> dict:
    """``COMP-SKY-002``-style accuracy of one automatic source against the simulator ground truth.

    Reported per condition when ``conditions`` maps observation id → condition label. Refusals are
    counted, never dropped: an extractor that answers on half the frames and is accurate there is a
    different thing from one that answers everywhere.
    """
    from hsreloc.retrieval.skyline_curve import CurveError

    rows, refusals = [], []
    for oid in _require(list(observation_ids), "extraction_accuracy"):
        try:
            gt = gt_source.get(oid)
        except CurveError as exc:
            refusals.append({"observation_id": oid, "side": "ground_truth", "reason": str(exc)})
            continue
        try:
            pred = pred_source.get(oid)
        except CurveError as exc:
            refusals.append({"observation_id": oid, "side": "prediction", "reason": str(exc)})
            continue
        d = np.abs(np.asarray(pred.row_per_col) - np.asarray(gt.row_per_col))
        rows.append({"observation_id": oid,
                     "condition": (conditions or {}).get(oid, ""),
                     "median_abs_px": float(np.median(d)),
                     "p90_abs_px": float(np.percentile(d, 90)),
                     "max_abs_px": float(d.max()),
                     "catastrophic_frac": float((d > 0.05 * height_px).mean()),
                     "gt_roughness": roughness(gt.row_per_col),
                     "pred_roughness": roughness(pred.row_per_col)})
    n_total = len(rows) + len(refusals)
    return {"rows": rows, "refusals": refusals,
            "n_evaluated": len(rows), "n_refused": len(refusals),
            "refusal_rate": (len(refusals) / n_total) if n_total else None,
            "by_condition": summarise_by(rows, "condition",
                                         ["median_abs_px", "p90_abs_px", "catastrophic_frac"]),
            "gt_source": gt_source.describe(), "pred_source": pred_source.describe(),
            "note": ("extraction accuracy against simulator ground truth; reported in its own table "
                     "and never mixed into a retrieval number (PROT-SKY-001 §3.5)"),
            "report_version": REPORT_VERSION}


# --------------------------------------------------------------------------------------------------
# 2b. same-pose cross-condition consistency (Experiment B; research R7 of the viewpoint feature)
# --------------------------------------------------------------------------------------------------

def condition_consistency(index_rows: list, sources: dict, height_px: int,
                          catastrophic_frac: float = 0.05) -> dict:
    """Pairwise curve disagreement within each same-pose group, across condition pairs, per source.

    ``index_rows`` are experiment-index rows (``groups.load_index``); only pose groups with ≥ 2
    members contribute. Every unordered pair of members is compared for every source in
    ``sources`` — including pairs whose condition is *identical* (duplicate captures), which are
    kept under their own ``same-condition rerender`` label: they bound the simulator's own
    repeatability the same way the GT source bounds geometry invariance.

    A refusal on either side is a row with ``status`` naming the side, never a dropped pair — an
    extractor that answers under DAY but refuses under DUSK is exactly what this measures.
    """
    from hsreloc.retrieval.skyline_curve import CurveError

    groups: dict = {}
    for r in _require(list(index_rows), "condition_consistency"):
        groups.setdefault(r["condition_pose_group"], []).append(r)
    multi = {k: sorted(v, key=lambda r: r["observation_id"])
             for k, v in groups.items() if len(v) > 1}
    _require(list(multi), "condition_consistency (no pose group has more than one member)")

    rows = []
    for key, source in sources.items():
        cache: dict = {}

        def curve_of(oid, _s=source, _c=cache):
            if oid not in _c:
                _c[oid] = _s.get(oid)
            return _c[oid]

        for pg, members in sorted(multi.items()):
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    cond_a = f"{a['condition_time_of_day']}+{a['condition_clouds']}"
                    cond_b = f"{b['condition_time_of_day']}+{b['condition_clouds']}"
                    pair = " vs ".join(sorted((cond_a, cond_b)))
                    row = {"pose_group": pg, "anchor_id": a.get("anchor_id", ""),
                           "source": key, "provenance": source.provenance,
                           "observation_a": a["observation_id"], "observation_b": b["observation_id"],
                           "condition_a": cond_a, "condition_b": cond_b,
                           "pair": ("same-condition rerender" if cond_a == cond_b else pair),
                           "tod_pair": " vs ".join(sorted((str(a["condition_time_of_day"]),
                                                           str(b["condition_time_of_day"])))),
                           "clouds_pair": " vs ".join(sorted((str(a["condition_clouds"]),
                                                              str(b["condition_clouds"])))),
                           "status": "ok"}
                    try:
                        ca = curve_of(a["observation_id"])
                    except CurveError as exc:
                        row.update(status="refused_a", refusal=str(exc)[:200])
                        rows.append(row)
                        continue
                    try:
                        cb = curve_of(b["observation_id"])
                    except CurveError as exc:
                        row.update(status="refused_b", refusal=str(exc)[:200])
                        rows.append(row)
                        continue
                    d = curve_disagreement(ca.row_per_col, cb.row_per_col, height_px)
                    row.update({f"curve_{k}": v for k, v in d.items()})
                    row["catastrophic"] = bool(
                        d["median_abs_offset_removed_px"] > catastrophic_frac * float(height_px))
                    rows.append(row)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    by_source_pair: dict = {}
    for r in ok_rows:
        by_source_pair.setdefault(r["source"], []).append(r)
    return {
        "rows": rows,
        "n_pose_groups": len(multi),
        "n_pairs_per_source": {k: len(v) for k, v in by_source_pair.items()},
        "refusals": [{k: r[k] for k in ("source", "observation_a", "observation_b", "status")}
                     for r in rows if r["status"] != "ok"],
        "catastrophic_frac_height": float(catastrophic_frac),
        "summary": {key: summarise_by(members, "pair",
                                      ["curve_median_abs_px", "curve_median_abs_offset_removed_px",
                                       "curve_offset_px"])
                    for key, members in sorted(by_source_pair.items())},
        "catastrophic_counts": {key: sum(1 for r in members if r.get("catastrophic"))
                                for key, members in sorted(by_source_pair.items())},
        "sources": {k: s.describe() for k, s in sources.items()},
        "report_version": REPORT_VERSION,
    }


# --------------------------------------------------------------------------------------------------
# 3. matcher-variant comparison on identical pairs
# --------------------------------------------------------------------------------------------------

def variant_comparison(tasks: dict, source, matchers: dict) -> dict:
    """Every variant scored on exactly the same pairs, with its recovered transform beside the score.

    The baseline must be present: a comparison that has dropped C0 cannot show what the extra freedom
    bought, which is the only thing the comparison is for.
    """
    if not any(isinstance(m, BaseMatcher) and m.variant.startswith("c0") for m in matchers.values()):
        raise ReportError("the frozen C0 baseline must be one of the matchers being compared — "
                          "a variant's effect is only readable against it (the brief's Part H)")
    from hsreloc.retrieval.skyline_curve import CurveError

    profile_config = ProfileConfig()
    cache: dict = {}

    def profile_of(oid):
        if oid not in cache:
            cache[oid] = normalize(source.get(oid), profile_config)
        return cache[oid]

    rows, refused = [], []
    for q in _require(tasks["queries"], "variant_comparison"):
        qid, rid = q["observation_id"], q["nearest_reference_id"]
        try:
            qp, rp = profile_of(qid), profile_of(rid)
        except CurveError as exc:
            refused.append({"query": qid, "reference": rid, "reason": str(exc)})
            continue
        row = {"query_id": qid, "reference_id": rid, "spacing_m": float(q["spacing_m"]),
               "stage": q["condition_stage"],
               "translation_m": float(q["condition_translation_m"]),
               "translation_bin": q["condition_translation_bin"]}
        for name, m in matchers.items():
            res = m.match(qp, rp)
            row.update({f"{name}_score": res.score, f"{name}_shift": res.shift,
                        f"{name}_scale": res.scale, f"{name}_warp": res.warp_magnitude,
                        f"{name}_accepted": res.accepted})
        rows.append(row)
    _require(rows, "variant_comparison (every pair was refused by the curve source)")
    return {"rows": rows, "refused": refused,
            "matchers": {k: m.describe() for k, m in matchers.items()},
            "summary": summarise_by(rows, "translation_bin",
                                    [f"{n}_score" for n in matchers]),
            "report_version": REPORT_VERSION}


# --------------------------------------------------------------------------------------------------
# 3b. rank-based variant retrieval over the full reference set (research R9)
# --------------------------------------------------------------------------------------------------

def variant_retrieval(tasks: dict, source, matchers: dict) -> dict:
    """Every matcher ranking the *whole* reference database for every query.

    ``variant_comparison`` scores only the correct pair, which shows recovery but not aliasing;
    this ranks all references, so a rung's gain (correct rank improves) and its cost (an unrelated
    reference now scores high) appear in the same row. C0 must be present, ties break
    deterministically by ``(-score, reference_id)``, and a refused query is recorded, never
    imputed.
    """
    if not any(isinstance(m, BaseMatcher) and m.variant.startswith("c0") for m in matchers.values()):
        raise ReportError("the frozen C0 baseline must be one of the matchers being compared — "
                          "a variant's effect is only readable against it (the brief's Part H)")
    from hsreloc.retrieval.skyline_curve import CurveError

    profile_config = ProfileConfig()
    cache: dict = {}

    def profile_of(oid):
        if oid not in cache:
            cache[oid] = normalize(source.get(oid), profile_config)
        return cache[oid]

    grids = tasks["grids"]
    rows, refused = [], []
    holes: dict = {}
    for q in _require(tasks["queries"], "variant_retrieval"):
        spacing_key = f"{float(q['spacing_m']):g}"
        references = grids[spacing_key]["references"]
        try:
            qp = profile_of(q["observation_id"])
        except CurveError as exc:
            refused.append({"query": q["observation_id"], "reason": str(exc)})
            continue
        ref_profiles = {}
        for g in references:
            rid = g["reference_id"]
            if rid in holes:
                continue
            try:
                ref_profiles[rid] = profile_of(rid)
            except CurveError as exc:
                holes[rid] = str(exc)
        if not ref_profiles:
            raise ReportError("no reference curve could be produced at all — nothing to rank")
        correct = q["nearest_reference_id"]
        row = {"query_id": q["observation_id"], "spacing_m": float(q["spacing_m"]),
               "stage": q["condition_stage"],
               "translation_m": float(q["condition_translation_m"]),
               "translation_bin": q["condition_translation_bin"],
               "up_diff_m": q.get("condition_up_diff_m"),
               "correct_reference": correct,
               "n_references_scored": len(ref_profiles)}
        for name, m in matchers.items():
            scored = sorted(((m.match(qp, rp), rid) for rid, rp in ref_profiles.items()),
                            key=lambda t: (-t[0].score, t[1]))
            ranked_ids = [rid for _, rid in scored]
            correct_rank = (ranked_ids.index(correct) + 1) if correct in ranked_ids else None
            best_res, best_id = scored[0]
            correct_res = next((res for res, rid in scored if rid == correct), None)
            best_incorrect = next((res for res, rid in scored if rid != correct), None)
            row.update({
                f"{name}_rank": correct_rank,
                f"{name}_top1": bool(best_id == correct),
                f"{name}_best_reference": best_id,
                f"{name}_best_score": best_res.score,
                f"{name}_best_accepted": best_res.accepted,
                f"{name}_correct_score": None if correct_res is None else correct_res.score,
                f"{name}_best_incorrect_score": None if best_incorrect is None else best_incorrect.score,
                f"{name}_margin": (None if (correct_res is None or best_incorrect is None
                                            or not np.isfinite(correct_res.score)
                                            or not np.isfinite(best_incorrect.score))
                                   else float(correct_res.score - best_incorrect.score)),
                f"{name}_best_shift": best_res.shift,
                f"{name}_best_scale": best_res.scale,
            })
        rows.append(row)
    _require(rows, "variant_retrieval (every query was refused by the curve source)")
    return {"rows": rows, "refused": refused, "database_holes": holes,
            "matchers": {k: m.describe() for k, m in matchers.items()},
            "top1_counts": {name: sum(1 for r in rows if r.get(f"{name}_top1"))
                            for name in matchers},
            "n_queries": len(rows),
            "summary": summarise_by(rows, "translation_bin",
                                    [f"{n}_margin" for n in matchers]
                                    + [f"{n}_best_incorrect_score" for n in matchers]),
            "source": source.describe(), "report_version": REPORT_VERSION}


# --------------------------------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------------------------------

def write_table(result: dict, out_dir: Path, name: str) -> Path:
    """``<name>.csv`` for the rows and ``<name>_summary.json`` for everything else."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _require(result.get("rows"), f"{name} (nothing to write)")
    columns = list(rows[0])
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    meta = {k: v for k, v in result.items() if k != "rows"}
    meta["n_rows"] = len(rows)
    (out_dir / f"{name}_summary.json").write_text(json.dumps(meta, indent=2, sort_keys=True,
                                                             default=str) + "\n", encoding="utf-8")
    return csv_path
