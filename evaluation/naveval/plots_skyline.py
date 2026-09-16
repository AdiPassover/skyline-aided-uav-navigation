"""Skyline evaluation figures (spec 004).

The minimum core set from data-model.md, each stamped with the evaluation id + config
digest (reusing ``plots._stamp``) and rendering an explicit labelled empty panel
(``plots._empty_panel``) when its metric is unsupported/omitted -- never a missing file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from naveval import skyline as sky
from naveval.plots import _empty_panel, _stamp


def _cdf(ax, values, label):
    arr = np.sort(np.asarray([v for v in values if v is not None], dtype=float))
    if arr.size == 0:
        return False
    y = np.arange(1, arr.size + 1) / arr.size
    ax.plot(arr, y, marker=".", label=label)
    return True


def render_all(record, query_set, config, out_dir, pos, head, rel, calib, footprint) -> None:
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    eid = Path(out_dir).name
    digest = config.get("config_digest", "")

    # position error CDF
    fig, ax = plt.subplots(figsize=(5, 4))
    if not _cdf(ax, pos["errors_m"], "position error"):
        _empty_panel(ax, "no in-coverage position errors")
    else:
        ax.set_xlabel("horizontal position error (m)"); ax.set_ylabel("CDF"); ax.legend()
    ax.set_title("Position error CDF")
    _stamp(fig, eid, digest); fig.savefig(fig_dir / "position_error_cdf.png", dpi=110); plt.close(fig)

    # heading error CDF: prior vs skyline result
    fig, ax = plt.subplots(figsize=(5, 4))
    result_err = [sky.heading_error_deg(r.est_heading_deg, gt.heading_deg)
                  for r, gt in sky._join(record, query_set) if gt and gt.heading_deg is not None]
    prior_err = [sky.heading_error_deg(r.compass_prior_deg, gt.heading_deg)
                 for r, gt in sky._join(record, query_set) if gt and gt.heading_deg is not None]
    got = _cdf(ax, prior_err, "compass prior")
    got = _cdf(ax, result_err, "skyline result") or got
    if not got:
        _empty_panel(ax, "no heading ground truth")
    else:
        ax.set_xlabel("heading error (deg)"); ax.set_ylabel("CDF"); ax.legend()
    ax.set_title("Heading error CDF (prior vs result)")
    _stamp(fig, eid, digest); fig.savefig(fig_dir / "heading_error_cdf.png", dpi=110); plt.close(fig)

    # outcome breakdown
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = rel["counts"]
    ax.bar(list(counts.keys()), list(counts.values()), color="#4477aa")
    ax.set_ylabel("count"); ax.set_title("Outcome breakdown")
    ax.text(0.99, 0.95, f"false-reloc: {rel['false_relocalization_rate']:.3f}\n"
                        f"confident-false: {rel['confident_false_relocalization_rate']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30); lbl.set_ha("right")
    _stamp(fig, eid, digest); fig.tight_layout(); fig.savefig(fig_dir / "outcome_breakdown.png", dpi=110); plt.close(fig)

    # confidence calibration
    fig, ax = plt.subplots(figsize=(5, 4))
    if calib is None or not calib.get("bins"):
        _empty_panel(ax, "insufficient data for calibration")
    else:
        xs = [b["mean_confidence"] for b in calib["bins"]]
        ys = [b["empirical_correct"] for b in calib["bins"]]
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("mean reported confidence"); ax.set_ylabel("empirical correctness")
    ax.set_title("Confidence calibration")
    _stamp(fig, eid, digest); fig.savefig(fig_dir / "confidence_calibration.png", dpi=110); plt.close(fig)

    # error vs sample spacing (single DB point + quantization-floor line)
    fig, ax = plt.subplots(figsize=(5, 4))
    spacing = footprint.get("sample_spacing_m")
    stats = pos["stats"]
    if spacing is not None and stats is not None:
        ax.axvline(spacing, linestyle="--", color="gray", label=f"sample spacing = {spacing} m")
        ax.scatter([spacing], [stats["median"]], color="#ee6677", label="median position error")
        ax.set_xlabel("reference sample spacing (m)"); ax.set_ylabel("position error (m)"); ax.legend()
    else:
        _empty_panel(ax, "no spacing / no position errors")
    ax.set_title("Position error vs sample spacing")
    _stamp(fig, eid, digest); fig.savefig(fig_dir / "error_vs_sample_spacing.png", dpi=110); plt.close(fig)

    # database footprint
    fig, ax = plt.subplots(figsize=(5, 4))
    items = [(k, footprint.get(k)) for k in ("total_bytes", "bytes_per_reference", "bytes_per_km2")
             if footprint.get(k) is not None]
    if items:
        ax.bar([k for k, _ in items], [v for _, v in items], color="#228833")
        ax.set_yscale("log"); ax.set_ylabel("bytes (log)")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(20); lbl.set_ha("right")
    else:
        _empty_panel(ax, "no footprint metadata")
    ax.set_title("Reference database footprint")
    _stamp(fig, eid, digest); fig.tight_layout(); fig.savefig(fig_dir / "db_footprint.png", dpi=110); plt.close(fig)

    # --- spec 006 additive: candidate score-separation / aliasing (positive vs negative) ---
    tol = float(config.get("operational_tolerance_m", 0.0))
    near_cfg = config.get("near_tolerance_m")
    near = float(near_cfg) if near_cfg is not None else None
    pos_s, near_s, neg_s = sky._labeled_candidate_scores(record, query_set, tol, near)
    fig, ax = plt.subplots(figsize=(5, 4))
    if pos_s and neg_s:
        bins = np.linspace(min(pos_s + neg_s + near_s), max(pos_s + neg_s + near_s), 20)
        ax.hist(neg_s, bins=bins, alpha=0.6, label="distant non-match", color="#bbbbbb")
        if near_s:
            ax.hist(near_s, bins=bins, alpha=0.6, label="nearby non-match", color="#ee6677")
        ax.hist(pos_s, bins=bins, alpha=0.6, label="true match", color="#228833")
        sep = sky.score_separation(record, query_set, tol, near)
        ax.set_xlabel("candidate match score"); ax.set_ylabel("count"); ax.legend(fontsize=8)
        ax.text(0.02, 0.95, f"ROC-AUC: {sep['roc_auc']:.3f}" if sep["roc_auc"] is not None else "ROC-AUC: n/a",
                transform=ax.transAxes, va="top", fontsize=8)
    else:
        _empty_panel(ax, "no shortlist candidates to separate")
    ax.set_title("Score separation / perceptual aliasing")
    _stamp(fig, eid, digest); fig.savefig(fig_dir / "score_separation.png", dpi=110); plt.close(fig)

    # --- spec 006 additive: precision-recall curve (per-query top-1 detections) ---
    fig, ax = plt.subplots(figsize=(5, 4))
    pairs = sky._per_query_detection(record, query_set, tol)
    n_pos = sum(l for _, l in pairs)
    if pairs and n_pos > 0:
        ranked = sorted(pairs, key=lambda x: -x[0])
        tp = fp = 0
        precisions, recalls = [], []
        for _, label in ranked:
            if label == 1:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp)); recalls.append(tp / n_pos)
        ax.plot(recalls, precisions, marker=".")
        prroc = sky.pr_roc(record, query_set, tol)
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.text(0.02, 0.05, f"AUC-PR: {prroc['auc_pr']:.3f}" if prroc["auc_pr"] is not None else "AUC-PR: n/a",
                transform=ax.transAxes, va="bottom", fontsize=8)
    else:
        _empty_panel(ax, "no positive detections for PR curve")
    ax.set_title("Precision-Recall (retrieval)")
    _stamp(fig, eid, digest); fig.savefig(fig_dir / "precision_recall.png", dpi=110); plt.close(fig)
