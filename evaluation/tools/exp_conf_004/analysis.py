"""EXP-CONF-004 Phases 1/3/4/5: identity verification and validation of the production-side
diagnostic-refit signal against the EXP-CONF-003 reference quantity and the fused-attitude
cross-check. No model is fitted; no thresholds are selected; tails are the descriptive
rank-based worst-2% convention; results are reported per sequence, never pooled-only.

Output: evaluations/exp-conf-004/analysis.json
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
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_003"))
from p4_analysis import average_precision, load_frames, spearman                 # noqa: E402
from panel_analysis import tail_flags                                            # noqa: E402

# exp_conf_003's driver is also named analysis.py, so load it by path (the exp_vo_010 pattern)
_spec = importlib.util.spec_from_file_location(
    "exp_conf_003_analysis", TOOL_DIR.parent / "exp_conf_003" / "analysis.py")
_a3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_a3)
ATTITUDE_PATH, read_cols, unwrap_interp_deg = (_a3.ATTITUDE_PATH, _a3.read_cols,
                                               _a3.unwrap_interp_deg)

SEQS = ("hkairport01-b", "amtown01-d")
DIAG_RUN = {s: f"{s}-homography-rigid-diagrefit-v1" for s in SEQS}
FROZEN_RUN = {s: f"{s}-homography-rigid-conf-v1" for s in SEQS}
RESTARTS = {"hkairport01-b": (922,), "amtown01-d": (10698, 11793)}
H4_CASES = {"hkairport01-b": (4305, 3195), "amtown01-d": (1605,)}
TAIL_FRAC = 0.02

# frames.csv columns that must match the committed frozen run exactly (v1.0 core; timing and
# the confidence block are per-run/config-dependent and excluded by construction)
IDENTITY_COLS = (["frame_index", "timestamp_s", "est_x", "est_y", "est_yaw_deg", "success",
                  "event", "reference_id"]
                 + [f"h{i}{j}" for i in range(3) for j in range(3)]
                 + ["track_count", "inlier_count"])


def qt(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(np.max(x))}


def ap_of(score: np.ndarray, target: np.ndarray) -> dict:
    m = np.isfinite(score) & np.isfinite(target)
    flags = tail_flags(np.where(m, target, np.nan))
    if flags.sum() < 3:
        return {"n": int(m.sum()), "n_pos": int(flags.sum())}
    ap, _, _ = average_precision(score[m], flags[m])
    return {"n": int(m.sum()), "n_pos": int(flags[m].sum()),
            "base_rate": float(flags[m].mean()), "ap": float(ap)}


def identity_check(seq: str) -> dict:
    froz = REPO / "runs" / FROZEN_RUN[seq]
    diag = REPO / "runs" / DIAG_RUN[seq]
    out: dict = {}
    out["logical_transform_byte_identical"] = (
        (froz / "logical_transform.csv").read_bytes()
        == (diag / "logical_transform.csv").read_bytes())
    with (froz / "frames.csv").open(newline="", encoding="utf-8") as f:
        fr = list(csv.DictReader(f))
    with (diag / "frames.csv").open(newline="", encoding="utf-8") as f:
        dr = list(csv.DictReader(f))
    out["frame_count_equal"] = len(fr) == len(dr)
    mismatches = 0
    first = None
    for a, b in zip(fr, dr):
        for c in IDENTITY_COLS:
            if a[c] != b[c]:
                mismatches += 1
                if first is None:
                    first = {"frame": a["frame_index"], "col": c,
                             "frozen": a[c], "diag": b[c]}
    out["frames_core_mismatches"] = mismatches
    if first:
        out["first_mismatch"] = first
    out["passed"] = (out["logical_transform_byte_identical"] and out["frame_count_equal"]
                     and mismatches == 0)
    return out


def main() -> int:
    report: dict = {"tail_frac": TAIL_FRAC, "sequences": {}}
    for seq in SEQS:
        s: dict = {}
        s["identity"] = identity_check(seq)
        print(f"[e4] {seq} identity: {s['identity']}", flush=True)
        if not s["identity"]["passed"]:
            print(f"[e4] {seq} IDENTITY FAILED — stopping per the frozen front half")
            report["sequences"][seq] = s
            continue

        # ---- load the diagnostic sidecar --------------------------------------------------
        frozen_dir = REPO / "runs" / FROZEN_RUN[seq]
        cheap = load_frames(frozen_dir)
        n = len(cheap["event"])
        cols = ["inliers", "refit_ok", "delta_rotation_deg", "delta_center_flow_px",
                "delta_flow_direction_deg", "delta_log_scale", "delta_perspective",
                "refit_time_ns", "readout_time_ns"]
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
        s["coverage"] = {"rows_with_refit": int(np.isfinite(drot).sum()),
                         "refit_failures": int(np.nansum(d["refit_ok"] == 0)),
                         "frames": n}
        s["signal"] = {c: qt(d[c]) for c in ("delta_rotation_deg", "delta_center_flow_px",
                                             "delta_flow_direction_deg", "delta_log_scale",
                                             "delta_perspective")}

        # ---- Phase 3A: vs EXP-CONF-003 rot_dis --------------------------------------------
        pa = {c: np.full(n, np.nan) for c in ("rot_dis_deg",)}
        with (REPO / f"evaluations/exp-conf-003/phase_a/{seq}.csv").open(
                newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = r.get("rot_dis_deg", "")
                if v not in ("", None):
                    pa["rot_dis_deg"][int(r["frame_index"])] = float(v)
        rot_dis = pa["rot_dis_deg"]
        rho, m = spearman(drot, rot_dis)
        e: dict = {"spearman": {"rho": rho, "n": m}}
        e["ap_vs_rot_dis_tail"] = ap_of(drot, rot_dis)
        fa, fb = tail_flags(drot), tail_flags(rot_dis)
        inter = int(np.sum(fa & fb))
        u = int(np.sum(fa | fb))
        e["tail_overlap"] = {"intersection": inter, "union": u,
                             "jaccard": inter / u if u else None,
                             "expected_if_independent": float(
                                 TAIL_FRAC ** 2 * np.isfinite(drot).sum())}
        mm = np.isfinite(drot) & np.isfinite(rot_dis)
        if mm.sum() > 500:
            dec = np.percentile(drot[mm], np.arange(0, 101, 10))
            e["rot_dis_median_by_drot_decile"] = [
                float(np.median(rot_dis[mm & (drot >= dec[k]) & (drot <= dec[k + 1])]))
                for k in range(10)]
        # representative disagreements
        rk_a = np.full(n, np.nan)
        rk_b = np.full(n, np.nan)
        rk_a[mm] = np.argsort(np.argsort(drot[mm])) / mm.sum()
        rk_b[mm] = np.argsort(np.argsort(rot_dis[mm])) / mm.sum()
        gap = rk_b - rk_a
        idx = np.where(mm)[0]
        top_miss = idx[np.argsort(-gap[idx])][:5]       # reference says bad, internal says fine
        top_fa = idx[np.argsort(gap[idx])][:5]          # internal says bad, reference fine
        e["rot_dis_high_but_drot_low"] = [
            {"frame": int(i), "drot": float(drot[i]), "rot_dis": float(rot_dis[i])}
            for i in top_miss]
        e["drot_high_but_rot_dis_low"] = [
            {"frame": int(i), "drot": float(drot[i]), "rot_dis": float(rot_dis[i])}
            for i in top_fa]
        s["vs_rot_dis"] = e

        # ---- Phase 3B: vs fused-attitude rotation error -----------------------------------
        att = read_cols(ATTITUDE_PATH[seq], ["timestamp_s", "yaw_compass_deg"])
        with (REPO / f"datasets/{seq}/frames.csv").open(newline="", encoding="utf-8") as f:
            t_cam = np.array([float(r["timestamp_s"]) for r in csv.DictReader(f)])
        lmeta = json.load((REPO / f"evaluations/exp-conf-001-dev/labels/{seq}.json")
                          .open(encoding="utf-8"))
        sign = float(lmeta["t_a_sign_convention"])
        att_cam = unwrap_interp_deg(t_cam, att["timestamp_s"], att["yaw_compass_deg"])
        d_att = np.full(n, np.nan)
        d_att[1:] = np.diff(att_cam)
        side = read_cols(frozen_dir / "logical_transform.csv", ["inc_rotation_deg"])
        has_inc = (cheap["event"] != "init") & (cheap["event"] != "restart")
        err_att = np.full(n, np.nan)
        mmm = has_inc & np.isfinite(d_att)
        err_att[mmm] = np.abs(sign * side["inc_rotation_deg"][mmm] - d_att[mmm])
        rho, m = spearman(drot, err_att)
        b: dict = {"spearman": {"rho": rho, "n": m}}
        b["ap_vs_err_att_tail"] = ap_of(drot, err_att)
        mm = np.isfinite(drot) & np.isfinite(err_att)
        if mm.sum() > 500:
            dec = np.percentile(drot[mm], np.arange(0, 101, 10))
            b["err_att_median_by_drot_decile"] = [
                float(np.median(err_att[mm & (drot >= dec[k]) & (drot <= dec[k + 1])]))
                for k in range(10)]
        s["vs_err_att"] = b

        # ---- Phase 4: common-mode blind spot ----------------------------------------------
        tail = tail_flags(err_att)
        cm: dict = {"n_tail": int(tail.sum())}
        for budget in (0.02, 0.05, 0.10):
            thr = np.nanquantile(drot, 1 - budget)
            flagged = np.where(np.isfinite(drot), drot > thr, False)
            caught = int(np.sum(flagged & tail))
            cm[f"recall_at_top{int(budget*100)}pct"] = {
                "caught": caught, "recall": caught / max(1, tail.sum())}
        med = np.nanmedian(drot)
        cm["tail_frames_with_drot_below_median"] = int(np.sum(tail & (drot < med)))
        cm["h4_cases"] = {}
        for i in H4_CASES[seq]:
            pct = (float(np.mean(drot[np.isfinite(drot)] <= drot[i]))
                   if np.isfinite(drot[i]) else None)
            cm["h4_cases"][str(i)] = {
                "drot": float(drot[i]) if np.isfinite(drot[i]) else None,
                "drot_percentile": pct,
                "err_att": float(err_att[i]) if np.isfinite(err_att[i]) else None}
        s["common_mode"] = cm

        # ---- restart-lead behaviour --------------------------------------------------------
        fin = np.isfinite(drot)
        pct_rank = np.full(n, np.nan)
        pct_rank[fin] = np.argsort(np.argsort(drot[fin])) / max(1, fin.sum() - 1)
        s["restart_leads"] = {}
        for r in RESTARTS[seq]:
            lead = pct_rank[max(1, r - 15):r]
            s["restart_leads"][str(r)] = [round(float(x), 3) if np.isfinite(x) else None
                                          for x in lead]

        # ---- Phase 5: cost -----------------------------------------------------------------
        c: dict = {"refit_time_us": qt(d["refit_time_ns"] / 1000.0),
                   "readout_time_us": qt(d["readout_time_ns"] / 1000.0)}
        tot = d["refit_time_ns"] + d["readout_time_ns"]
        c["total_overhead_us"] = qt(tot / 1000.0)
        rho, m = spearman(d["inliers"], d["refit_time_ns"])
        c["spearman_inliers_vs_refit_time"] = {"rho": rho, "n": m}
        timing_csv = REPO / f"evaluations/exp-conf-004/timing/{seq}_motion_timing.csv"
        if timing_csv.exists():
            with timing_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            key = next((k for k in rows[0] if "motion" in k.lower() and "ns" in k.lower()),
                       None)
            if key:
                mt = np.array([float(r[key]) for r in rows if r.get(key, "") != ""])
                c["baseline_motion_stage_ms"] = qt(mt / 1e6)
                c["overhead_pct_of_motion_stage_median"] = float(
                    np.nanmedian(tot) / np.median(mt) * 100.0)
            else:
                c["baseline_motion_timing_columns"] = list(rows[0].keys())
        s["cost"] = c

        report["sequences"][seq] = s
        print(f"[e4] {seq} done", flush=True)

    out = REPO / "evaluations/exp-conf-004/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"[e4] written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
