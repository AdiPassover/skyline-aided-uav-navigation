"""Per-query metrics for DEM-based skyline relocalization (spec 004).

Pure metric functions over a parsed skyline result record + query set. No I/O: the CLI
(``evaluate_skyline.py``) does the file handling and assembles ``metrics.json``. Every
metric definition here is frozen by the spec; thresholds/parameters arrive via ``config``.

Reuses ``naveval.frames.shortest_angle_diff_deg`` for circular heading error and
``naveval.metrics.runtime_statistics`` for latency percentiles. Position is horizontal
Euclidean distance in the common ENU frame the query set's ground truth already lives in
(DEC-004), so no re-projection is needed here.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from naveval.frames import shortest_angle_diff_deg
from naveval.metrics import runtime_statistics
from naveval.skyline_record import QueryGroundTruth, QueryResult, SkylineQuerySet, SkylineResultRecord


# --- small statistics helper (median + high percentiles; never mean alone, FR-M5) ---

def distribution_stats(values) -> Optional[dict]:
    """Median / p90 / p95 / max / mean / n over a 1-D array, or None if empty."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


# --- primitive per-query quantities ---

def position_error_m(result: QueryResult, gt: QueryGroundTruth) -> Optional[float]:
    """Horizontal Euclidean error (m) between the reported position and GT, in ENU."""
    if result.est_east_m is None or result.est_north_m is None:
        return None
    if gt.east_m is None or gt.north_m is None:
        return None
    de = result.est_east_m - gt.east_m
    dn = result.est_north_m - gt.north_m
    return math.hypot(de, dn)


def heading_error_deg(estimated_deg: Optional[float], gt_deg: Optional[float]) -> Optional[float]:
    """Absolute circular error (shortest angular distance mod 360)."""
    if estimated_deg is None or gt_deg is None:
        return None
    return abs(shortest_angle_diff_deg(estimated_deg, gt_deg))


def _join(record: SkylineResultRecord, query_set: SkylineQuerySet):
    """Yield (result, gt) pairs joined by query_id -> query_index. Missing GT -> gt None."""
    for r in record.results:
        gt = None
        try:
            gt = query_set.queries.get(int(r.query_id))
        except (ValueError, TypeError):
            gt = None
        yield r, gt


# --- E1: position, heading, region-in-candidates ---

def position_metrics(record, query_set, tolerance_m: float) -> dict:
    """Per-query position errors over in-coverage SUCCESS results, plus distribution."""
    errors = []
    for r, gt in _join(record, query_set):
        if r.outcome != "SUCCESS" or gt is None or not gt.in_coverage:
            continue
        e = position_error_m(r, gt)
        if e is not None:
            errors.append(e)
    return {"errors_m": errors, "stats": distribution_stats(errors)}


def heading_metrics(record, query_set) -> dict:
    """Prior vs skyline-result circular heading error, and the paired improvement."""
    result_err, prior_err, improvement = [], [], []
    for r, gt in _join(record, query_set):
        if gt is None or gt.heading_deg is None:
            continue
        r_err = heading_error_deg(r.est_heading_deg, gt.heading_deg)
        p_err = heading_error_deg(r.compass_prior_deg, gt.heading_deg)
        if r_err is not None:
            result_err.append(r_err)
        if p_err is not None:
            prior_err.append(p_err)
        if r_err is not None and p_err is not None:
            improvement.append(p_err - r_err)  # positive => skyline improves on the prior
    return {
        "result_stats": distribution_stats(result_err),
        "prior_stats": distribution_stats(prior_err),
        "improvement_stats": distribution_stats(improvement),
    }


def region_in_candidates(record, query_set, tolerance_m: float, k: int = 1) -> dict:
    """Top-1 (and, where a shortlist is exposed, top-k) region-in-candidates rate over
    in-coverage queries. A query is a hit if the matched/best candidate (or any of the
    top-k candidates) is within tolerance of the true position."""
    in_cov = [(r, gt) for r, gt in _join(record, query_set)
              if gt is not None and gt.in_coverage and gt.east_m is not None]
    if not in_cov:
        return {"top1_rate": None, "recall_at_k_rate": None, "n": 0, "k": k}

    def within(east, north, gt) -> bool:
        return math.hypot(east - gt.east_m, north - gt.north_m) <= tolerance_m

    top1_hits, recall_hits, any_candidates = 0, 0, False
    for r, gt in in_cov:
        # top-1: the matched sample (SUCCESS) or the reported position.
        best_east = r.matched_east_m if r.matched_east_m is not None else r.est_east_m
        best_north = r.matched_north_m if r.matched_north_m is not None else r.est_north_m
        if best_east is not None and best_north is not None and within(best_east, best_north, gt):
            top1_hits += 1
            recall_hits += 1
            continue
        if r.candidates:  # top-k only where a ranked shortlist is exposed
            any_candidates = True
            if any(within(ce, cn, gt) for ce, cn, _ in r.candidates[:k]):
                recall_hits += 1

    n = len(in_cov)
    return {
        "top1_rate": top1_hits / n,
        "recall_at_k_rate": (recall_hits / n) if any_candidates else None,
        "n": n,
        "k": k,
    }


# --- E2: reliability, false / confident-false, calibration ---

def _is_correct(r: QueryResult, gt: Optional[QueryGroundTruth], tolerance_m: float) -> Optional[bool]:
    """Whether a SUCCESS is a correct localization. None if not a SUCCESS or GT missing."""
    if r.outcome != "SUCCESS" or gt is None:
        return None
    if not gt.in_coverage:
        return False  # a SUCCESS on an out-of-coverage query is wrong by construction
    e = position_error_m(r, gt)
    if e is None:
        return None
    return e <= tolerance_m


def reliability_metrics(record, query_set, tolerance_m: float, reject_threshold: float) -> dict:
    total = len(record.results)
    counts = {"SUCCESS": 0, "REJECTED": 0, "AMBIGUOUS": 0, "OUT_OF_COVERAGE": 0, "EXTRACTION_FAILURE": 0}
    false_reloc = 0
    confident_false = 0
    correct = 0
    evaluable_success = 0
    for r, gt in _join(record, query_set):
        counts[r.outcome] += 1
        verdict = _is_correct(r, gt, tolerance_m)
        if r.outcome == "SUCCESS" and verdict is not None:
            evaluable_success += 1
            if verdict:
                correct += 1
            else:
                false_reloc += 1
                if r.confidence is not None and r.confidence >= reject_threshold:
                    confident_false += 1
    return {
        "total": total,
        "success_rate": counts["SUCCESS"] / total,
        "rejection_rate": counts["REJECTED"] / total,
        "ambiguous_rate": counts["AMBIGUOUS"] / total,
        "out_of_coverage_rate": counts["OUT_OF_COVERAGE"] / total,
        "extraction_failure_rate": counts["EXTRACTION_FAILURE"] / total,
        "correct_localization_rate": (correct / total),
        "false_relocalization_rate": (false_reloc / total),
        "confident_false_relocalization_rate": (confident_false / total),
        "counts": counts,
        "evaluable_success": evaluable_success,
    }


def confidence_calibration(record, query_set, tolerance_m: float, n_bins: int = 5,
                           min_samples: int = 10) -> Optional[dict]:
    """Binned reliability curve: mean confidence vs empirical correctness per bin.
    Returns None (omit with reason upstream) when too few evaluable successes exist."""
    pairs = []
    for r, gt in _join(record, query_set):
        verdict = _is_correct(r, gt, tolerance_m)
        if r.outcome == "SUCCESS" and verdict is not None and r.confidence is not None:
            pairs.append((float(r.confidence), 1.0 if verdict else 0.0))
    if len(pairs) < min_samples:
        return None
    conf = np.array([p[0] for p in pairs])
    corr = np.array([p[1] for p in pairs])
    edges = np.linspace(conf.min(), conf.max() + 1e-9, n_bins + 1)
    bins = []
    for i in range(n_bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if not mask.any():
            continue
        bins.append({
            "bin_low": float(edges[i]),
            "bin_high": float(edges[i + 1]),
            "mean_confidence": float(conf[mask].mean()),
            "empirical_correct": float(corr[mask].mean()),
            "n": int(mask.sum()),
        })
    return {"bins": bins, "n": len(pairs)}


# --- E4: latency + RAM ---

def latency_metrics(record) -> dict:
    ns = np.array([r.process_time_ns for r in record.results], dtype=np.float64)
    stats = runtime_statistics(ns)  # reused: median/p90/p99 in ms
    env = record.manifest.environment
    return {
        "latency": stats,
        "peak_rss_mb": env.get("peak_rss_mb"),
        "is_target_hardware": record.manifest.is_target_hardware,
        "hardware": {k: env.get(k) for k in ("hostname", "os", "cpu_model")},
    }


# --- E5: reference-database footprint ---

def database_footprint(record) -> dict:
    # Additive (spec 006): a historical/generic record carries footprint fields in
    # reference_source; fall back to it when reference_db is empty (spec 004/005 unchanged).
    db = record.manifest.reference_db or record.manifest.reference_source
    total_bytes = db.get("total_bytes")
    n_refs = db.get("n_references")
    area = db.get("covered_area_km2")
    spacing = db.get("sample_spacing_m")
    bytes_per_ref = (total_bytes / n_refs) if (total_bytes and n_refs) else None
    bytes_per_km2 = (total_bytes / area) if (total_bytes and area) else None
    return {
        "total_bytes": total_bytes,
        "n_references": n_refs,
        "bytes_per_reference": bytes_per_ref,
        "bytes_per_km2": bytes_per_km2,
        "covered_area_km2": area,
        "sample_spacing_m": spacing,
        "camera_height_m": db.get("camera_height_m"),
    }


# --- E3: minimal condition slices ---

def _binned(values_and_errors, edges) -> list:
    out = []
    for i in range(len(edges) - 1):
        errs = [e for v, e in values_and_errors if edges[i] <= v < edges[i + 1] and e is not None]
        stats = distribution_stats(errs)
        out.append({"bin_low": edges[i], "bin_high": edges[i + 1],
                    "position_error_median_m": (stats["median"] if stats else None),
                    "n": (stats["n"] if stats else 0)})
    return out


def condition_slices(record, query_set, tolerance_m: float, slice_config: dict) -> dict:
    """Position-error median sliced by the minimal core factors, driven by config.
    A factor is skipped cleanly when its inputs are absent."""
    joined = list(_join(record, query_set))

    def success_error(r, gt):
        if r.outcome != "SUCCESS" or gt is None or not gt.in_coverage:
            return None
        return position_error_m(r, gt)

    slices: dict = {}

    prior_bins = slice_config.get("prior_error_bins")
    if prior_bins:
        data = []
        for r, gt in joined:
            if gt is None or gt.heading_deg is None or gt.compass_prior_deg is None:
                continue
            pe = abs(shortest_angle_diff_deg(gt.compass_prior_deg, gt.heading_deg))
            data.append((pe, success_error(r, gt)))
        if data:
            slices["prior_error_deg"] = _binned(data, prior_bins)

    roll_bins = slice_config.get("roll_bins")
    if roll_bins:
        data = []
        for r, gt in joined:
            if gt is None or gt.roll_deg is None:
                continue
            data.append((abs(gt.roll_deg), success_error(r, gt)))
        if data:
            slices["roll_magnitude_deg"] = _binned(data, roll_bins)

    # coverage position: interior vs edge (in_coverage true) -- boolean slice.
    cov = {"in_coverage": [], "out_of_coverage": []}
    for r, gt in joined:
        if gt is None:
            continue
        key = "in_coverage" if gt.in_coverage else "out_of_coverage"
        cov[key].append(success_error(r, gt))
    slices["coverage"] = {
        "in_coverage": distribution_stats(cov["in_coverage"]),
        "out_of_coverage_success_count": sum(
            1 for r, gt in joined if gt is not None and not gt.in_coverage and r.outcome == "SUCCESS"
        ),
    }
    return slices


# ---------------------------------------------------------------------------
# Spec 006 additive metrics: topological success, score-separation / aliasing,
# PR-ROC (AUC-PR + new-place ROC), and generic-axis slicing.
#
# These are matcher-agnostic: they read only the generic result record's outcomes,
# candidate coordinates, and scores (contracts/result-record.md). Grounded in the VPR-Bench
# evaluation protocol (LIT-013): AUC-PR as the primary retrieval metric, ROC for rejecting
# out-of-database ("new") places, plus score-separation (d', ROC-AUC) for perceptual aliasing.
# Existing 004 functions above are unchanged.
# ---------------------------------------------------------------------------

def topological_success(record, query_set, tolerance_m: float, k: int = 1) -> dict:
    """Headline success: is the top-1 (or top-k) retrieved reference within tolerance of the true
    place? A thin, spec-006-named view over ``region_in_candidates`` (which it delegates to, so the
    two never diverge). ``success_rate`` == top-1 topological-localization success @ tolerance."""
    region = region_in_candidates(record, query_set, tolerance_m, k)
    return {
        "success_rate": region["top1_rate"],
        "recall_at_1_rate": region["top1_rate"],
        "recall_at_k_rate": region["recall_at_k_rate"],
        "k": k,
        "n": region["n"],
    }


def _labeled_candidate_scores(record, query_set, tolerance_m: float, near_m: Optional[float]):
    """Over exposed shortlist candidates across in-coverage queries with GT, split candidate
    scores into positive (within tolerance of GT), near-positive (ignore band, within near_m),
    and negative (beyond). Returns (positives, nears, negatives)."""
    positives, nears, negatives = [], [], []
    for r, gt in _join(record, query_set):
        if gt is None or gt.east_m is None or not gt.in_coverage:
            continue
        for ce, cn, cs in r.candidates:
            if cs is None:
                continue
            d = math.hypot(ce - gt.east_m, cn - gt.north_m)
            if d <= tolerance_m:
                positives.append(cs)
            elif near_m is not None and d <= near_m:
                nears.append(cs)
            else:
                negatives.append(cs)
    return positives, nears, negatives


def _roc_auc(pos, neg) -> Optional[float]:
    """Mann-Whitney ROC-AUC = P(score_pos > score_neg), ties counted at 0.5. None if a class is
    empty. Rank-based (average ranks), so O(n log n)."""
    if not pos or not neg:
        return None
    data = sorted([(float(s), 1) for s in pos] + [(float(s), 0) for s in neg], key=lambda x: x[0])
    ranks = [0.0] * len(data)
    i = 0
    while i < len(data):
        j = i
        while j + 1 < len(data) and data[j + 1][0] == data[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank across the tie group
        for t in range(i, j + 1):
            ranks[t] = avg_rank
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    sum_pos_ranks = sum(ranks[t] for t in range(len(data)) if data[t][1] == 1)
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _d_prime(pos, neg) -> Optional[float]:
    """Sensitivity index (mean_pos - mean_neg) / sqrt(0.5*(var_pos+var_neg)). None if a class is
    empty or the pooled spread is zero (perfect separation: rely on ROC-AUC instead)."""
    if len(pos) < 1 or len(neg) < 1:
        return None
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    denom = math.sqrt(0.5 * (float(p.var()) + float(n.var())))
    if denom == 0.0:
        return None
    return float((p.mean() - n.mean()) / denom)


def score_separation(record, query_set, tolerance_m: float, near_m: Optional[float] = None) -> dict:
    """Positive-vs-negative match-score separation for perceptual-aliasing analysis (FR-M5):
    d' and ROC-AUC over the shortlist candidate scores, plus each class's distribution."""
    pos, near, neg = _labeled_candidate_scores(record, query_set, tolerance_m, near_m)
    return {
        "d_prime": _d_prime(pos, neg),
        "roc_auc": _roc_auc(pos, neg),
        "n_positive": len(pos),
        "n_near": len(near),
        "n_negative": len(neg),
        "positive_stats": distribution_stats(pos),
        "negative_stats": distribution_stats(neg),
    }


def aliasing_by_distance(record, query_set, tolerance_m: float, near_m: float) -> dict:
    """Score distributions for true matches / spatially-nearby non-matches / distant non-matches —
    the aliasing view (FR-M5). Nearby non-matches are the dangerous population."""
    pos, near, neg = _labeled_candidate_scores(record, query_set, tolerance_m, near_m)
    return {
        "true_match": distribution_stats(pos),
        "nearby_non_match": distribution_stats(near),
        "distant_non_match": distribution_stats(neg),
    }


def _per_query_detection(record, query_set, tolerance_m: float):
    """(score, label) per query for PR/ROC. score = confidence (fallback best_score); label = 1
    iff a correct consumable fix (SUCCESS within tolerance AND in-coverage), else 0. An out-of-DB
    query is a negative (label 0) — a confident SUCCESS on it hurts precision, as it should."""
    pairs = []
    for r, gt in _join(record, query_set):
        if gt is None:
            continue
        score = r.confidence if r.confidence is not None else r.best_score
        if score is None:
            continue
        correct = False
        if r.outcome == "SUCCESS" and gt.in_coverage:
            e = position_error_m(r, gt)
            correct = e is not None and e <= tolerance_m
        pairs.append((float(score), 1 if correct else 0))
    return pairs


def _auc_pr(pairs) -> Optional[float]:
    """Average precision (area under the precision-recall curve) from (score, label). None if no
    positives exist."""
    if not pairs:
        return None
    n_pos = sum(l for _, l in pairs)
    if n_pos == 0:
        return None
    ranked = sorted(pairs, key=lambda x: -x[0])
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    for _, label in ranked:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return float(ap)


def pr_roc(record, query_set, tolerance_m: float, operating_points=None) -> dict:
    """PR + ROC over per-query top-1 detections (FR-M6): AUC-PR (primary), ROC-AUC (new-place
    rejection), and precision/recall/false-positive count at each configured operating point."""
    pairs = _per_query_detection(record, query_set, tolerance_m)
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    ops = []
    for thr in (operating_points or []):
        thr = float(thr)
        tp = sum(1 for s, l in pairs if s >= thr and l == 1)
        fp = sum(1 for s, l in pairs if s >= thr and l == 0)
        fn = sum(1 for s, l in pairs if s < thr and l == 1)
        ops.append({
            "threshold": thr,
            "precision": (tp / (tp + fp)) if (tp + fp) > 0 else None,
            "recall": (tp / (tp + fn)) if (tp + fn) > 0 else None,
            "false_positive_count": fp,
        })
    return {
        "auc_pr": _auc_pr(pairs),
        "roc_auc": _roc_auc(pos, neg),
        "n": len(pairs),
        "n_positive": len(pos),
        "operating_points": ops,
    }


def generic_axis_slices(record, query_set, tolerance_m: float, axis_config: dict) -> dict:
    """Generic slicing support for later specs (FR-M10): slice top-1 success-error median by a
    per-query axis value read from the query's condition fields — reference spacing, viewpoint
    displacement, prior radius, sequence length. ``axis_config`` maps an axis name to
    ``{"field": <condition field>, "bins": [edges]}``. An axis is **skipped cleanly** when no
    query carries its field (so 006 provides the support without the later experiments existing)."""
    joined = list(_join(record, query_set))
    out: dict = {}
    for axis, spec in (axis_config or {}).items():
        field_name = spec.get("field")
        bins = spec.get("bins")
        if not field_name or not bins:
            continue
        data = []
        for r, gt in joined:
            if gt is None:
                continue
            raw = gt.conditions.get(field_name)
            if raw in (None, ""):
                continue
            try:
                v = float(raw)
            except (ValueError, TypeError):
                continue
            se = position_error_m(r, gt) if (r.outcome == "SUCCESS" and gt.in_coverage) else None
            data.append((v, se))
        if data:
            out[axis] = _binned(data, bins)
    return out
