"""EXP-CONF-001 P4: development-side predictive-validity analysis (DEV DATA ONLY — not evidence).

Consumes: the four confidence captures, the per-sequence label CSVs (``labels.py``), and the
committed fitting-target decision (``fitting_target.py``). Produces ``metrics.json`` plus flat
CSV tables under the output directory. Everything here is **in-sample development analysis** on
the calibration pool (plus the two overlap consistency checks); nothing is held-out evidence.

Frozen inputs it must not and does not change: the six-signal list and orientations, W/m, the
target definitions, the fitting-target selection (read from JSON, applied verbatim), the
adequacy floors, and the A5 ordinal-squash semantics (the fitted model's canonical quantity is
the linear predictor z; every metric below is rank-based and mapping-invariant).

Statistical helpers are implemented on numpy alone (the ``naveval`` dependency policy: no
scipy/sklearn/pandas — research.md R4).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# The frozen signal list, with a priori orientation: +1 means HIGHER value = MORE degraded
# (envelope/scene signals), -1 means LOWER value = more degraded (support signals). Orientations
# come from the DEC-CONF-002 mechanism table, not from data.
SIGNALS = {
    "inlier_count": -1.0,
    "relative_support": -1.0,
    "inlier_ratio": -1.0,
    "inlier_coverage": -1.0,
    "inc_flow_px": +1.0,
    "inc_log_scale_dispersion": +1.0,
}


# ----------------------------------------------------------------------------- loading


def load_frames(run_dir: Path) -> dict:
    """Reads the v1.1.0 frames.csv columns this analysis needs, absence as NaN."""
    cols = ["track_count", "inlier_count", "inlier_coverage", "relative_support",
            "inc_flow_px", "inc_log_scale", "inc_log_scale_dispersion",
            "confidence_score", "confidence_time_ns", "process_time_ns"]
    out: dict = {c: [] for c in cols}
    out["event"] = []
    out["success"] = []
    with (run_dir / "frames.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for c in cols:
                v = r.get(c, "")
                out[c].append(float(v) if v not in ("", None) else np.nan)
            out["event"].append(r["event"])
            out["success"].append(r["success"] == "true")
    res = {c: np.array(out[c]) for c in cols}
    res["event"] = np.array(out["event"])
    res["success"] = np.array(out["success"])
    with np.errstate(invalid="ignore", divide="ignore"):
        res["inlier_ratio"] = res["inlier_count"] / res["track_count"]
    return res


def load_labels(path: Path) -> dict:
    g_a, t_a, g_b, t_b, g_c, t_c, t_e = [], [], [], [], [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g_a.append(r["gradable_a"] == "1")
            t_a.append(float(r["t_a_deg"]) if r["gradable_a"] == "1" else np.nan)
            g_b.append(r["gradable_b"] == "1")
            t_b.append(float(r["t_b_deg"]) if r["gradable_b"] == "1" else np.nan)
            g_c.append(r["gradable_c"] == "1")
            t_c.append(float(r["t_c_frac"]) if r["gradable_c"] == "1" else np.nan)
            t_e.append(r["t_e"] == "1")
    return {"gradable_a": np.array(g_a), "t_a": np.array(t_a),
            "gradable_b": np.array(g_b), "t_b": np.array(t_b),
            "gradable_c": np.array(g_c), "t_c": np.array(t_c),
            "t_e": np.array(t_e)}


# ----------------------------------------------------------------------------- statistics


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), 1-based — the standard Spearman ranking."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), n
    rx, ry = rankdata(x[m]), rankdata(y[m])
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan"), n
    return float((rx * ry).sum() / denom), n


def average_precision(score: np.ndarray, label: np.ndarray) -> tuple[float, int, int]:
    """AUC-PR as average precision (step interpolation), higher score = positive-leaning.

    Ties are handled by processing equal scores as one block (threshold semantics)."""
    m = np.isfinite(score)
    score, label = score[m], label[m].astype(bool)
    n_pos = int(label.sum())
    if n_pos == 0:
        return float("nan"), n_pos, int(m.sum())
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], label[order]
    tp = fp = 0
    ap = 0.0
    prev_recall = 0.0
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        tp += int(y[i:j + 1].sum())
        fp += int((~y[i:j + 1]).sum())
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j + 1
    return float(ap), n_pos, int(m.sum())


def pr_curve(score: np.ndarray, label: np.ndarray) -> list[dict]:
    """Operating points at each distinct threshold (decreasing score)."""
    m = np.isfinite(score)
    score, label = score[m], label[m].astype(bool)
    n_pos = int(label.sum())
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], label[order]
    pts = []
    tp = fp = 0
    i = 0
    n = len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        tp += int(y[i:j + 1].sum())
        fp += int((~y[i:j + 1]).sum())
        pts.append({"threshold": float(s[i]),
                    "recall": tp / n_pos if n_pos else float("nan"),
                    "precision": tp / (tp + fp),
                    "fp": fp, "tp": tp})
        i = j + 1
    return pts


def fit_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 200) -> dict:
    """Plain MLE logistic regression via IRLS on internally standardised predictors; raw-scale
    coefficients are recovered exactly. numpy only."""
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    w = np.zeros(Z1.shape[1])
    converged = False
    for _ in range(max_iter):
        eta = Z1 @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        W = np.clip(p * (1 - p), 1e-10, None)
        grad = Z1.T @ (y - p)
        H = (Z1 * W[:, None]).T @ Z1
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    eta = Z1 @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
    ll = float((y * np.log(np.clip(p, 1e-300, None))
                + (1 - y) * np.log(np.clip(1 - p, 1e-300, None))).sum())
    beta = w[1:] / sd
    intercept = float(w[0] - (w[1:] * mu / sd).sum())
    return {"intercept": intercept, "coefficients": beta.tolist(),
            "converged": bool(converged), "log_likelihood": ll, "n": int(len(y)),
            "n_pos": int(y.sum())}


def pav_decreasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic fit of P(y=1 | x) constrained NON-INCREASING in x (support signal semantics).
    Returns (block_upper_edges_x, block_means). Implemented as increasing-PAV on -x."""
    order = np.argsort(-x, kind="mergesort")   # descending x -> increasing constraint on -x
    ys = y[order].astype(float)
    # classic PAV with weights
    means = list(ys)
    weights = [1.0] * len(ys)
    starts = list(range(len(ys)))
    i = 0
    while i < len(means) - 1:
        if means[i] > means[i + 1] + 1e-15:
            m = (means[i] * weights[i] + means[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1])
            means[i] = m
            weights[i] += weights[i + 1]
            del means[i + 1], weights[i + 1], starts[i + 1]
            while i > 0 and means[i - 1] > means[i] + 1e-15:
                m = (means[i - 1] * weights[i - 1] + means[i] * weights[i]) / (weights[i - 1] + weights[i])
                means[i - 1] = m
                weights[i - 1] += weights[i]
                del means[i], weights[i], starts[i]
                i -= 1
        else:
            i += 1
    xs = x[order]
    edges = []
    k = 0
    for b, st in enumerate(starts):
        end = starts[b + 1] if b + 1 < len(starts) else len(xs)
        edges.append(float(xs[end - 1]))  # smallest x in this block (descending order)
        k = end
    return np.array(edges), np.array(means)


# ----------------------------------------------------------------------------- analysis


def degraded_labels(t_vals: np.ndarray, gradable: np.ndarray, threshold: float) -> np.ndarray:
    lab = np.full(len(t_vals), np.nan)
    lab[gradable] = (t_vals[gradable] > threshold).astype(float)
    return lab


def fa_per_minute_curve(score: np.ndarray, label: np.ndarray, frame_rate_hz: float) -> list[dict]:
    """Recall vs false alarms per minute over the evaluated frames."""
    m = np.isfinite(score) & np.isfinite(label)
    minutes = m.sum() / frame_rate_hz / 60.0
    pts = pr_curve(score[m], label[m])
    return [{"threshold": p["threshold"], "recall": p["recall"],
             "false_alarms_per_min": p["fp"] / minutes, "precision": p["precision"]}
            for p in pts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", required=True, help="seq=rundir:labels.csv (calibration pool)")
    ap.add_argument("--checks", nargs="*", default=[], help="seq=rundir:labels.csv (consistency checks)")
    ap.add_argument("--decision", required=True, help="fitting_target.py output JSON")
    ap.add_argument("--frame-rate", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    decision = json.loads(Path(a.decision).read_text(encoding="utf-8"))
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    def parse(spec):
        seq, rest = spec.split("=", 1)
        run_dir, labels_path = rest.rsplit(":", 1)
        return seq, load_frames(Path(run_dir)), load_labels(Path(labels_path))

    pool = [parse(s) for s in a.pool]
    checks = [parse(s) for s in a.checks]
    metrics: dict = {"decision": decision["selection"],
                     "note": "DEVELOPMENT ANALYSIS - in-sample on the calibration pool; "
                             "not held-out evidence"}

    # ---------------- Question A: Spearman per signal x graded target, per sequence ----------
    spearman_rows = []
    for seq, fr, lb in pool + checks:
        role = "pool" if any(s == seq for s, _, _ in pool) else "check"
        for sig in SIGNALS:
            for tgt, gkey in (("t_a", "gradable_a"), ("t_b", "gradable_b"), ("t_c", "gradable_c")):
                mask = lb[gkey]
                rho, n = spearman(fr[sig][mask], lb[tgt][mask])
                spearman_rows.append({"sequence": seq, "role": role, "signal": sig,
                                      "target": tgt, "spearman_rho": rho, "n": n})
    metrics["spearman"] = spearman_rows

    # ---------------- T-D detection ----------------------------------------------------------
    sel = decision["selection"]
    members = decision["t_a"]["members"]
    fit_member = sel["fitting_target"]

    # pooled arrays over the calibration pool
    def pooled(key, source="frames"):
        if source == "frames":
            return np.concatenate([fr[key] for _, fr, _ in pool])
        return np.concatenate([lb[key] for _, _, lb in pool])

    pool_grad = pooled("gradable_a", "labels")
    pool_ta = pooled("t_a", "labels")

    detection: dict = {}
    for mname, minfo in members.items():
        lab = degraded_labels(pool_ta, pool_grad, minfo["threshold"])
        per_signal = {}
        for sig, orient in SIGNALS.items():
            s = orient * pooled(sig)
            apv, n_pos, n_eval = average_precision(s[pool_grad], lab[pool_grad])
            per_signal[sig] = {"auc_pr": apv, "n_pos": n_pos, "n_eval": n_eval}
        base_rate = float(np.nanmean(lab[pool_grad]))
        detection[mname] = {"threshold": minfo["threshold"], "episodes": minfo["episodes_total"],
                            "base_rate": base_rate, "per_signal": per_signal}
    metrics["t_d_detection_single_signals"] = detection

    # ---------------- the selected model -----------------------------------------------------
    model_block: dict = {"branch": sel["branch"]}
    fitted_scores_pool = None
    if fit_member is not None:
        thr = members[fit_member]["threshold"]
        lab = degraded_labels(pool_ta, pool_grad, thr)
        X_cols = list(SIGNALS)
        Xp = np.column_stack([pooled(c) for c in X_cols])
        eval_mask = pool_grad & np.isfinite(lab) & np.all(np.isfinite(Xp), axis=1)

        if sel["branch"] == "six_signal_logistic":
            fit = fit_logistic(Xp[eval_mask], lab[eval_mask])
            z = fit["intercept"] + Xp @ np.array(fit["coefficients"])
            fitted_scores_pool = np.where(np.all(np.isfinite(Xp), axis=1), z, np.nan)
            model_block.update({"signals": X_cols, "fit": fit,
                                "score_semantics": "canonical linear predictor z "
                                                   "(higher = more degraded)"})
        else:  # single_index_fallback: isotonic over relative_support
            rs = pooled("relative_support")
            eval_mask = pool_grad & np.isfinite(lab) & np.isfinite(rs)
            edges, means = pav_decreasing(rs[eval_mask], lab[eval_mask].astype(int))
            # degradation score = PAV block mean at the frame's relative_support
            def iso_score(v):
                out = np.full(len(v), np.nan)
                fin = np.isfinite(v)
                # edges are the lower x-edge of each block, blocks ordered by descending x
                idx = np.searchsorted(-edges, -v[fin], side="left")
                idx = np.clip(idx, 0, len(means) - 1)
                out[fin] = means[idx]
                return out
            fitted_scores_pool = iso_score(rs)
            model_block.update({"signal": "relative_support",
                                "isotonic_blocks": [{"x_lower_edge": float(e), "mean": float(m)}
                                                    for e, m in zip(edges, means)],
                                "score_semantics": "isotonic block mean of the in-sample "
                                                   "degraded fraction (ORDINAL, not a probability)"})

        apv, n_pos, n_eval = average_precision(fitted_scores_pool[eval_mask], lab[eval_mask])
        model_block["auc_pr_fit_member"] = {"member": fit_member, "auc_pr": apv,
                                            "n_pos": n_pos, "n_eval": n_eval}

        # sensitivity: same fitted scores against the other members' labels (no refit)
        sens = {}
        for mname, minfo in members.items():
            if mname == fit_member:
                continue
            lab2 = degraded_labels(pool_ta, pool_grad, minfo["threshold"])
            m2 = pool_grad & np.isfinite(lab2) & np.isfinite(fitted_scores_pool)
            apv2, np2, ne2 = average_precision(fitted_scores_pool[m2], lab2[m2])
            sens[mname] = {"auc_pr": apv2, "n_pos": np2, "n_eval": ne2}
        model_block["sensitivity_members"] = sens

        # Brown-Lowe refit baseline: logistic over (track_count, inlier_count)
        bl_X = np.column_stack([pooled("track_count"), pooled("inlier_count")])
        bl_mask = pool_grad & np.isfinite(lab) & np.all(np.isfinite(bl_X), axis=1)
        bl_fit = fit_logistic(bl_X[bl_mask], lab[bl_mask])
        bl_z = bl_fit["intercept"] + bl_X @ np.array(bl_fit["coefficients"])
        bl_ap, bl_np, bl_ne = average_precision(bl_z[bl_mask], lab[bl_mask])
        model_block["brown_lowe_refit"] = {
            "form": "logistic margin over (track_count, inlier_count) - the refitted "
                    "n_i > alpha + beta*n_f family (LIT-CONF-002 S2)",
            "fit": bl_fit, "auc_pr": bl_ap, "n_pos": bl_np, "n_eval": bl_ne,
            "adequacy_note": "pre-registered baseline; reported regardless of the 10*k floor "
                             "(k=2 would require 20 episodes)"}

        # recall vs false alarms per minute (selected model + best single indicator)
        model_block["fa_curve_model"] = fa_per_minute_curve(
            fitted_scores_pool[pool_grad], lab[pool_grad], a.frame_rate)[:400]
        best_single = max(detection[fit_member]["per_signal"],
                          key=lambda s: (detection[fit_member]["per_signal"][s]["auc_pr"]
                                         if np.isfinite(detection[fit_member]["per_signal"][s]["auc_pr"]) else -1))
        s_best = SIGNALS[best_single] * pooled(best_single)
        model_block["fa_curve_best_single"] = {
            "signal": best_single,
            "points": fa_per_minute_curve(s_best[pool_grad], lab[pool_grad], a.frame_rate)[:400]}

        # decile-conditioned graded error (score deciles over evaluated frames)
        sc = fitted_scores_pool
        m3 = pool_grad & np.isfinite(sc) & np.isfinite(pool_ta)
        qs = np.percentile(sc[m3], np.arange(0, 101, 10))
        deciles = []
        for d in range(10):
            lo, hi = qs[d], qs[d + 1]
            sel_m = m3 & (sc >= lo) & (sc <= hi if d == 9 else sc < hi)
            if sel_m.sum() > 0:
                deciles.append({"decile": d + 1, "n": int(sel_m.sum()),
                                "t_a_mean": float(np.nanmean(pool_ta[sel_m])),
                                "t_a_median": float(np.nanmedian(pool_ta[sel_m])),
                                "t_a_p95": float(np.nanpercentile(pool_ta[sel_m], 95))})
        model_block["t_a_by_score_decile"] = deciles

        # consistency checks: same fitted model evaluated on -a / -c
        check_out = {}
        for seq, fr, lb in checks:
            Xc = np.column_stack([fr[c] for c in X_cols])
            if sel["branch"] == "six_signal_logistic":
                zc = model_block["fit"]["intercept"] + Xc @ np.array(model_block["fit"]["coefficients"])
                zc = np.where(np.all(np.isfinite(Xc), axis=1), zc, np.nan)
            else:
                zc = iso_score(fr["relative_support"])
            labc = degraded_labels(lb["t_a"], lb["gradable_a"], thr)
            mc = lb["gradable_a"] & np.isfinite(labc) & np.isfinite(zc)
            apc, npc, nec = average_precision(zc[mc], labc[mc])
            check_out[seq] = {"auc_pr": apc, "n_pos": npc, "n_eval": nec}
        model_block["consistency_checks"] = check_out
    metrics["model"] = model_block

    # ---------------- T-E (operational, partially circular; reported separately) -------------
    # Per-sequence model score for the lead-time analysis.
    def model_score_for(fr):
        if fit_member is None:
            return None
        Xc = np.column_stack([fr[c] for c in list(SIGNALS)])
        if sel["branch"] == "six_signal_logistic":
            zc = model_block["fit"]["intercept"] + Xc @ np.array(model_block["fit"]["coefficients"])
            return np.where(np.all(np.isfinite(Xc), axis=1), zc, np.nan)
        return iso_score(fr["relative_support"])

    # Swept operating points: model-score thresholds achieving in-pool T-D recall levels.
    te_thresholds = {}
    if fitted_scores_pool is not None:
        thr_lab = degraded_labels(pool_ta, pool_grad,
                                  members[fit_member]["threshold"])
        curve = pr_curve(fitted_scores_pool[pool_grad & np.isfinite(fitted_scores_pool)
                                            & np.isfinite(thr_lab)],
                         thr_lab[pool_grad & np.isfinite(fitted_scores_pool)
                                 & np.isfinite(thr_lab)])
        for target_recall in (0.5, 0.8, 0.9):
            pt = next((p_ for p_ in curve if p_["recall"] >= target_recall), None)
            if pt is not None:
                te_thresholds[f"recall_{target_recall}"] = pt["threshold"]

    te_block = {"partial_circularity_note":
                "restarts are fired by the estimator's own support logic; this analysis is "
                "OPERATIONAL (A2) and is never pooled with T-D results",
                "operating_points": te_thresholds, "sequences": []}
    for seq, fr, lb in pool + checks:
        te = lb["t_e"]
        restarts = np.flatnonzero(fr["event"] == "restart")
        row = {"sequence": seq, "restarts": int(len(restarts)),
               "t_e_frames": int(te.sum()), "events": []}
        zc = model_score_for(fr)
        if zc is not None:
            for ridx in restarts:
                window = np.arange(max(0, ridx - 10), ridx)
                traj = [float(zc[i]) if np.isfinite(zc[i]) else None
                        for i in range(max(0, ridx - 15), ridx)]
                ev = {"restart_frame": int(ridx), "score_last15": traj, "lead_frames": {}}
                for name, thr_ in te_thresholds.items():
                    crossed = [int(ridx - i) for i in window if np.isfinite(zc[i]) and zc[i] > thr_]
                    ev["lead_frames"][name] = max(crossed) if crossed else 0
                row["events"].append(ev)
            # separation: AUC-PR of the model score for T-E frames (where a score exists)
            m_te = np.isfinite(zc)
            ap_te, np_te, ne_te = average_precision(zc[m_te], te[m_te])
            row["auc_pr_t_e_scored_frames"] = {"auc_pr": ap_te, "n_pos": np_te, "n_eval": ne_te,
                                               "note": "restart rows themselves carry no score "
                                                       "and drop out; positives are the "
                                                       "pre-restart window frames that do"}
        te_block["sequences"].append(row)
    metrics["t_e"] = te_block

    # ---------------- runtime cost ------------------------------------------------------------
    rt = []
    for seq, fr, lb in pool + checks:
        ct = fr["confidence_time_ns"]
        pt = fr["process_time_ns"]
        fin = np.isfinite(ct)
        rt.append({"sequence": seq,
                   "confidence_time_us_median": float(np.nanmedian(ct) / 1e3),
                   "confidence_time_us_p99": float(np.nanpercentile(ct[fin], 99) / 1e3),
                   "process_time_ms_median": float(np.nanmedian(pt) / 1e6),
                   "confidence_over_process_median": float(np.nanmedian(ct) / np.nanmedian(pt))})
    metrics["runtime"] = rt

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n",
                                         encoding="utf-8")
    with (outdir / "spearman.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "role", "signal", "target",
                                          "spearman_rho", "n"], lineterminator="\n")
        w.writeheader()
        w.writerows(spearman_rows)
    print(json.dumps({"decision": sel, "auc_pr_model": model_block.get("auc_pr_fit_member"),
                      "runtime": rt}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
