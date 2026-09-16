"""EXP-CONF-005: frozen held-out analyses of delta_rot_refit on HKairport03 / AMtown03.

Everything here implements exactly the front half of the EXP-CONF-005 pre-registration,
frozen before the sequences were downloaded: identity gate, primary metrics A-E (circular moving-block bootstrap,
L = 100 frames, B = 500, seed 20260831, fixed tail labels), restart classification (never
from delta_rot_refit itself), secondary fused-attitude check with its continuity gate,
common-mode audit, and cost. Per-sequence first; pooled only afterwards and flagged as such.
No fitting, no thresholds, no probability.

Output: evaluations/exp-conf-005/analysis.json
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
REPO = TOOL_DIR.parents[2]
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_001"))
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_002"))
from p4_analysis import average_precision, load_frames, spearman                 # noqa: E402
from panel_analysis import tail_flags                                            # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "exp_conf_003_analysis", TOOL_DIR.parent / "exp_conf_003" / "analysis.py")
_a3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a3)
read_cols, unwrap_interp_deg = _a3.read_cols, _a3.unwrap_interp_deg

SEQS = ("hkairport03", "amtown03")
OFF_RUN = {s: f"{s}-homography-rigid-conf-off-v1" for s in SEQS}
DIAG_RUN = {s: f"{s}-homography-rigid-diagrefit-v1" for s in SEQS}
TAIL_FRAC = 0.02
BOOT_L = 100
BOOT_B = 500
BOOT_SEED = 20260831
LEAD = 15
SUSTAINED_MIN = 10
SUSTAINED_PCT = 0.80


def qt(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(np.max(x))}


def ap_of(score: np.ndarray, flags: np.ndarray, mask: np.ndarray) -> float:
    ap, _, _ = average_precision(score[mask], flags[mask])
    return float(ap)


def pct_rank(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    m = np.isfinite(x)
    if m.sum():
        out[m] = np.argsort(np.argsort(x[m])) / max(1, m.sum() - 1)
    return out


def block_bootstrap_ci(score: np.ndarray, target: np.ndarray, flags: np.ndarray,
                       rng: np.random.Generator) -> dict:
    """Circular moving-block bootstrap CIs for Spearman(score, target) and AP(score->flags).
    Tail labels are fixed from the full sequence; blocks resample (score, target, flag)
    triples jointly along the frame axis."""
    m = np.isfinite(score) & np.isfinite(target)
    idx = np.where(m)[0]
    n = len(idx)
    if n < 500:
        return {"n": n, "note": "too few samples for bootstrap"}
    sp, ap = [], []
    n_blocks = int(np.ceil(n / BOOT_L))
    for _ in range(BOOT_B):
        starts = rng.integers(0, n, size=n_blocks)
        take = (starts[:, None] + np.arange(BOOT_L)[None, :]).ravel() % n
        take = take[:n]
        ii = idx[take]
        rho, _ = spearman(score[ii], target[ii])
        sp.append(rho)
        if flags[ii].sum() >= 3:
            a, _, _ = average_precision(score[ii], flags[ii])
            ap.append(a)
    def ci(v):
        v = np.array([x for x in v if np.isfinite(x)])
        return {"lo2.5": float(np.percentile(v, 2.5)), "hi97.5": float(np.percentile(v, 97.5)),
                "n_resamples": int(len(v))} if len(v) else {"n_resamples": 0}
    return {"n": n, "block_len": BOOT_L, "B": BOOT_B,
            "spearman_ci": ci(sp), "ap_ci": ci(ap)}


def identity_check(seq: str) -> dict:
    off = REPO / "runs" / OFF_RUN[seq]
    diag = REPO / "runs" / DIAG_RUN[seq]
    out: dict = {}
    out["logical_transform_byte_identical"] = (
        (off / "logical_transform.csv").read_bytes()
        == (diag / "logical_transform.csv").read_bytes())
    with (off / "frames.csv").open(newline="", encoding="utf-8") as f:
        fr = list(csv.DictReader(f))
    with (diag / "frames.csv").open(newline="", encoding="utf-8") as f:
        dr = list(csv.DictReader(f))
    out["frame_count_equal"] = len(fr) == len(dr)
    skip = {"process_time_ns", "confidence_time_ns"}
    mism = 0
    first = None
    for a, b in zip(fr, dr):
        for c in a:
            if c in skip:
                continue
            if a[c] != b.get(c, ""):
                mism += 1
                if first is None:
                    first = {"frame": a["frame_index"], "col": c, "off": a[c],
                             "diag": b.get(c, "")}
    out["frames_mismatches_excl_timing"] = mism
    if first:
        out["first_mismatch"] = first
    out["passed"] = (out["logical_transform_byte_identical"] and out["frame_count_equal"]
                     and mism == 0)
    return out


def analyse_sequence(seq: str) -> dict:
    s: dict = {"identity": identity_check(seq)}
    if not s["identity"]["passed"]:
        return s

    off_dir = REPO / "runs" / OFF_RUN[seq]
    cheap = load_frames(off_dir)
    n = len(cheap["event"])
    events = cheap["event"]
    restarts = [int(i) for i in np.where(events == "restart")[0]]
    s["counts"] = {"frames": n,
                   "accepted_increments": int(np.sum((events == "none")
                                                     | (events == "recenter"))),
                   "recenters": int(np.sum(events == "recenter")),
                   "restarts": restarts}

    # diagnostic sidecar
    cols = ["inliers", "refit_ok", "delta_rotation_deg", "refit_time_ns", "readout_time_ns"]
    d = {c: np.full(n, np.nan) for c in cols}
    with (REPO / "runs" / DIAG_RUN[seq] / "diagnostic_refit.csv").open(
            newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r["frame_index"])
            for c in cols:
                v = r.get(c, "")
                if v not in ("", None):
                    d[c][i] = float(v)
    drot = d["delta_rotation_deg"]
    s["coverage"] = {"diag_rows_with_refit": int(np.isfinite(drot).sum()),
                     "refit_failures": int(np.nansum(d["refit_ok"] == 0))}
    s["signal_drot"] = qt(drot)

    # independent reference (phase_a machinery output)
    ref_cols = ("rot_dis_deg", "n_fb_valid")
    pa = {c: np.full(n, np.nan) for c in ref_cols}
    with (REPO / f"evaluations/exp-conf-005/reference/{seq}.csv").open(
            newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r["frame_index"])
            for c in ref_cols:
                v = r.get(c, "")
                if v not in ("", None):
                    pa[c][i] = float(v)
    rot_dis = pa["rot_dis_deg"]
    s["coverage"]["reference_pairs"] = int(np.isfinite(rot_dis).sum())

    rng = np.random.default_rng(BOOT_SEED)

    # ---- primary metrics --------------------------------------------------------------
    m = np.isfinite(drot) & np.isfinite(rot_dis)
    flags_ref = tail_flags(np.where(m, rot_dis, np.nan))
    rho, nn = spearman(drot, rot_dis)
    p: dict = {"spearman": {"rho": rho, "n": nn}}
    if m.sum() > 500:
        dec = np.percentile(drot[m], np.arange(0, 101, 10))
        p["rot_dis_by_drot_decile"] = [
            {"median": float(np.median(rot_dis[mm])),
             "p90": float(np.percentile(rot_dis[mm], 90))}
            for k in range(10)
            for mm in [m & (drot >= dec[k]) & (drot <= dec[k + 1])]]
    p["ap_vs_ref_tail"] = {"ap": ap_of(drot, flags_ref, m),
                           "base_rate": float(flags_ref[m].mean()),
                           "n_pos": int(flags_ref[m].sum())}
    fa = tail_flags(np.where(m, drot, np.nan))
    inter = int(np.sum(fa & flags_ref))
    p["tail"] = {"jaccard": inter / max(1, np.sum(fa | flags_ref)),
                 "precision": inter / max(1, fa.sum()),
                 "recall": inter / max(1, flags_ref.sum()),
                 "n_a": int(fa.sum()), "n_b": int(flags_ref.sum())}
    p["bootstrap"] = block_bootstrap_ci(drot, rot_dis, flags_ref, rng)
    s["primary"] = p

    # ---- restart analysis --------------------------------------------------------------
    drot_pct = pct_rank(drot)
    rot_dis_pct = pct_rank(rot_dis)
    # the frozen off-runs are v1.0 records (no confidence block), so inc_flow_px comes from
    # the logical sidecar, where the production readout always writes it
    flow_side = read_cols(off_dir / "logical_transform.csv", ["inc_flow_px"])["inc_flow_px"]
    flow = np.where((events != "init") & (events != "restart"), flow_side, np.nan)
    flow_pct = pct_rank(flow)
    ev_list = []
    for r in restarts:
        lead = np.arange(max(1, r - LEAD), r)
        scored = lead[np.isfinite(rot_dis[lead])]
        e: dict = {"restart": int(r), "n_scored_lead": int(len(scored))}
        if len(scored) < 5:
            e["classification"] = "insufficient evidence"
        else:
            med_ref = float(np.median(rot_dis_pct[scored]))
            med_fb = float(np.median(pa["n_fb_valid"][scored]))
            med_flow = float(np.median(flow_pct[lead][np.isfinite(flow_pct[lead])])) \
                if np.isfinite(flow_pct[lead]).any() else float("nan")
            if med_ref >= 0.90 and med_fb >= 300:
                e["classification"] = "estimator-internal"
            elif med_ref < 0.90 and (med_fb < 300 or med_flow >= 0.95):
                e["classification"] = "regime"
            else:
                e["classification"] = "ambiguous"
            e["evidence"] = {"median_lead_rot_dis_pct": med_ref,
                             "median_lead_n_fb_valid": med_fb,
                             "median_lead_flow_pct": med_flow}
        prof = drot_pct[max(1, r - LEAD):r + 1]
        e["drot_pct_profile_t-15..t"] = [round(float(x), 3) if np.isfinite(x) else None
                                         for x in prof]
        fin = prof[np.isfinite(prof)]
        e["sustained_rise"] = bool(np.sum(fin >= SUSTAINED_PCT) >= SUSTAINED_MIN)
        ev_list.append(e)
    s["restart_events"] = ev_list

    # ---- secondary: fused attitude -----------------------------------------------------
    att_path = REPO / f"datasets/{seq}/attitude.csv"
    sec: dict = {}
    if att_path.exists():
        att = read_cols(att_path, ["timestamp_s", "yaw_compass_deg"])
        ta = att["timestamp_s"]
        gaps = np.diff(ta)
        with (REPO / f"datasets/{seq}/frames.csv").open(newline="", encoding="utf-8") as f:
            t_cam = np.array([float(r["timestamp_s"]) for r in csv.DictReader(f)])
        gate = {"rate_hz": float(1.0 / np.median(gaps)), "max_gap_s": float(np.max(gaps)),
                "covers": bool(ta[0] <= t_cam[0] and ta[-1] >= t_cam[-1])}
        gate["passed"] = (90.0 <= gate["rate_hz"] <= 110.0 and gate["max_gap_s"] <= 0.5
                          and gate["covers"])
        sec["continuity"] = gate
        if gate["passed"]:
            att_cam = unwrap_interp_deg(t_cam, ta, att["yaw_compass_deg"])
            d_att = np.full(n, np.nan)
            d_att[1:] = np.diff(att_cam)
            side = read_cols(off_dir / "logical_transform.csv", ["inc_rotation_deg"])
            has_inc = (events != "init") & (events != "restart")
            mm = has_inc & np.isfinite(d_att)
            meds = {sgn: float(np.median(np.abs(
                sgn * side["inc_rotation_deg"][mm] - d_att[mm]))) for sgn in (1.0, -1.0)}
            sgn = 1.0 if meds[1.0] <= meds[-1.0] else -1.0
            sec["sign_convention"] = {"chosen": sgn, "median_abs_err": meds}
            err_att = np.full(n, np.nan)
            err_att[mm] = np.abs(sgn * side["inc_rotation_deg"][mm] - d_att[mm])
            sec["err_att"] = qt(err_att)
            rho, nn = spearman(drot, err_att)
            sec["spearman"] = {"rho": rho, "n": nn}
            m2 = np.isfinite(drot) & np.isfinite(err_att)
            if m2.sum() > 500:
                dec = np.percentile(drot[m2], np.arange(0, 101, 10))
                sec["err_att_by_drot_decile"] = [
                    {"median": float(np.median(err_att[b])),
                     "p90": float(np.percentile(err_att[b], 90))}
                    for k in range(10)
                    for b in [m2 & (drot >= dec[k]) & (drot <= dec[k + 1])]]
            flags_att = tail_flags(np.where(m2, err_att, np.nan))
            sec["ap_vs_att_tail"] = {"ap": ap_of(drot, flags_att, m2),
                                     "base_rate": float(flags_att[m2].mean()),
                                     "n_pos": int(flags_att[m2].sum())}
            sec["bootstrap"] = block_bootstrap_ci(drot, err_att, flags_att,
                                                  np.random.default_rng(BOOT_SEED + 1))
            budgets = {}
            for b_ in (0.02, 0.05, 0.10):
                thr = np.nanquantile(drot, 1 - b_)
                caught = int(np.sum((drot > thr) & flags_att))
                budgets[f"top{int(b_*100)}pct"] = {
                    "caught": caught, "recall": caught / max(1, flags_att.sum())}
            sec["review_budget_recall"] = budgets
            med = np.nanmedian(drot)
            below = flags_att & (drot < med)
            sec["tail_below_median_drot"] = {
                "count": int(below.sum()),
                "fraction_of_tail": float(below.sum() / max(1, flags_att.sum()))}
            # ---- common-mode audit (frozen classification) ---------------------------
            cm_frames = np.where(below)[0]
            cls = {"internal_missed_instability": 0, "no_image_evidence_instability": 0,
                   "ambiguous": 0}
            for i in cm_frames:
                rp = rot_dis_pct[i]
                if np.isfinite(rp) and rp >= 0.9:
                    cls["internal_missed_instability"] += 1
                elif np.isfinite(rp) and rp <= 0.5:
                    cls["no_image_evidence_instability"] += 1
                else:
                    cls["ambiguous"] += 1
            sec["common_mode_classification"] = cls
            conv = tail_flags(np.where(np.isfinite(drot), drot, np.nan)) & \
                np.where(np.isfinite(err_att), err_att <= np.nanmedian(err_att), False)
            hi_ref = int(np.sum(conv & np.where(np.isfinite(rot_dis_pct),
                                                rot_dis_pct >= 0.9, False)))
            sec["converse_drot_high_err_low"] = {
                "count": int(conv.sum()),
                "with_independent_instability": hi_ref}
    else:
        sec["continuity"] = None
    s["secondary_attitude"] = sec

    # ---- cost --------------------------------------------------------------------------
    c: dict = {"refit_time_us": qt(d["refit_time_ns"] / 1000.0),
               "readout_time_us": qt(d["readout_time_ns"] / 1000.0),
               "total_overhead_us": qt((d["refit_time_ns"] + d["readout_time_ns"]) / 1000.0)}
    rho, nn = spearman(d["inliers"], d["refit_time_ns"])
    c["spearman_inliers_vs_refit_time"] = {"rho": rho, "n": nn}
    timing_csv = REPO / f"evaluations/exp-conf-005/timing/{seq}_motion_timing.csv"
    if timing_csv.exists():
        with timing_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        key = next((k for k in rows[0] if "motion" in k.lower() and "ns" in k.lower()), None)
        if key:
            mt = np.array([float(r[key]) for r in rows if r.get(key, "") != ""])
            c["baseline_motion_stage_ms"] = qt(mt / 1e6)
            c["overhead_pct_of_motion_stage_median"] = float(
                np.nanmedian(d["refit_time_ns"] + d["readout_time_ns"])
                / np.median(mt) * 100.0)
    s["cost"] = c
    return s


def main() -> int:
    report: dict = {"tail_frac": TAIL_FRAC,
                    "bootstrap": {"block_len": BOOT_L, "B": BOOT_B, "seed": BOOT_SEED}}
    seqs_done = {}
    for seq in SEQS:
        if not (REPO / "runs" / DIAG_RUN[seq]).exists():
            print(f"[e5] {seq}: runs not present yet — skipping")
            continue
        seqs_done[seq] = analyse_sequence(seq)
        print(f"[e5] {seq} done", flush=True)
    report["sequences"] = seqs_done

    # pooled ONLY after both sequences (flagged)
    if len(seqs_done) == 2 and all(s.get("identity", {}).get("passed") for s in
                                   seqs_done.values()):
        report["pooled_note"] = ("pooled numbers are secondary to the per-sequence results "
                                 "above, per the frozen reporting rule")
    out = REPO / "evaluations/exp-conf-005/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"[e5] written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
