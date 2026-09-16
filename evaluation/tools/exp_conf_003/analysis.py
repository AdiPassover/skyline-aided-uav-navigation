"""EXP-CONF-003 Phases B/C/D: frozen analyses over the Phase A decomposition.

No model is fitted; no signals are combined; no thresholds are optimised. Tails are the
descriptive rank-based worst-2% convention throughout (no physical threshold is invented).

Inputs: Phase A CSVs, Phase C counterfactual score CSVs, committed run records/labels/panel
outputs, fused-attitude streams (hkairport01-b: dataset file; amtown01-d: locally-ingested file,
provenance caveat recorded), RTK
ground-truth heading at the audit-supported 1.0 s window scale only.

Output: evaluations/exp-conf-003/analysis.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
REPO = TOOL_DIR.parents[2]
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_002"))
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_001"))
from p4_analysis import SIGNALS, average_precision, load_frames, load_labels, spearman  # noqa: E402
from panel_analysis import PANEL_ALL, load_panel, tail_flags                            # noqa: E402

SEQS = ("hkairport01-b", "amtown01-d")
ATTITUDE_PATH = {
    "hkairport01-b": REPO / "datasets/hkairport01-b/attitude.csv",
    "amtown01-d": REPO / "datasets/amtown01-d/attitude.csv",
}
ATTITUDE_PROVENANCE = {
    "hkairport01-b": "committed dataset descriptor",
    "amtown01-d": "local ingest (untracked); continuity-checked below",
}
REFINE_RUN = {
    "hkairport01-b": "hkairport01-b-homography-rigid-refine-v1",
    "amtown01-d": "amtown01-d-homography-rigid-refine-v1",
}
PHASE_A_DIR = REPO / "evaluations/exp-conf-003/phase_a"
PHASE_C_DIR = REPO / "evaluations/exp-conf-003/phase_c"
LABELS_DIR = REPO / "evaluations/exp-conf-001-dev/labels"
CASES_JSON = REPO / "evaluations/exp-conf-002/hypothesis_audit/cases.json"

QUANT = ["rot_dis_deg", "flowvec_dis_px", "flowmag_dis_px", "dir_dis_deg", "logscale_dis",
         "d_sim_px", "d_proj_px", "d_tot_px", "d_tot_misread_px"]
TAIL_FRAC = 0.02
WINDOW_S = 1.0


def qt(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x))}


def read_cols(path: Path, cols: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, list] = {c: [] for c in cols}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for c in cols:
                v = r.get(c, "")
                out[c].append(float(v) if v not in ("", None) else np.nan)
    return {c: np.array(v) for c, v in out.items()}


def load_phase(path: Path, n: int, cols: list[str]) -> dict[str, np.ndarray]:
    out = {c: np.full(n, np.nan) for c in cols}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r["frame_index"])
            for c in cols:
                v = r.get(c, "")
                if v not in ("", None):
                    out[c][i] = float(v)
    return out


def unwrap_interp_deg(t_out: np.ndarray, t_in: np.ndarray, deg: np.ndarray) -> np.ndarray:
    return np.degrees(np.interp(t_out, t_in, np.unwrap(np.radians(deg))))


def ap_vs_tail(score: np.ndarray, target: np.ndarray) -> dict:
    """AP of `score` ranking the worst-TAIL_FRAC of `target` (both higher = worse)."""
    m = np.isfinite(score) & np.isfinite(target)
    if m.sum() < 100:
        return {"n": int(m.sum())}
    flags = tail_flags(np.where(m, target, np.nan))
    if flags.sum() < 3:
        return {"n": int(m.sum()), "n_pos": int(flags.sum())}
    ap, _, _ = average_precision(score[m], flags[m])
    return {"n": int(m.sum()), "n_pos": int(flags[m].sum()),
            "base_rate": float(flags[m].mean()), "ap": float(ap)}


def main() -> int:
    report: dict = {"tail_frac": TAIL_FRAC, "window_s": WINDOW_S, "sequences": {}}

    with CASES_JSON.open(encoding="utf-8") as f:
        audit_cases = [(c["id"], c["seq"], int(c["frame"])) for c in json.load(f)["cases"]]

    for seq in SEQS:
        s: dict = {}
        run_dir = REPO / f"runs/{seq}-homography-rigid-conf-v1"
        cheap = load_frames(run_dir)
        n = len(cheap["event"])
        events = cheap["event"]
        labels = load_labels(LABELS_DIR / f"{seq}.csv")
        lmeta = json.load((LABELS_DIR / f"{seq}.json").open(encoding="utf-8"))
        sign = float(lmeta["t_a_sign_convention"])
        panel = load_panel(seq, n)
        pa = load_phase(PHASE_A_DIR / f"{seq}.csv", n,
                        QUANT + ["rot_a_deg", "rot_b_deg", "flow_a_px", "n_inl",
                                 "t_klt_ms", "t_magsac_ms", "t_refit_ms"])
        sidecar = read_cols(run_dir / "logical_transform.csv",
                            ["inc_rotation_deg", "inc_perspective", "inc_flow_px"])
        with (REPO / f"datasets/{seq}/frames.csv").open(newline="", encoding="utf-8") as f:
            t_cam = np.array([float(r["timestamp_s"]) for r in csv.DictReader(f)])

        has_inc = (events != "init") & (events != "restart")

        # ---- consistency: analysis readout == navigator readout ------------------------
        m = has_inc & np.isfinite(pa["rot_a_deg"])
        s["readout_identity"] = {
            "n": int(m.sum()),
            "max_abs_rot_diff_deg": float(np.max(np.abs(
                pa["rot_a_deg"][m] - sidecar["inc_rotation_deg"][m]))),
        }

        # ---- Phase A distributions ------------------------------------------------------
        s["phase_a"] = {q: qt(pa[q]) for q in QUANT}
        s["phase_a"]["published_d_vo_median_px"] = qt(panel["d_vo_median_px"])
        # artifact share of the published convention
        both = np.isfinite(pa["d_tot_px"]) & np.isfinite(pa["d_tot_misread_px"])
        s["phase_a"]["misread_minus_corrected_px"] = qt(
            pa["d_tot_misread_px"][both] - pa["d_tot_px"][both])
        rho, nn = spearman(pa["d_tot_misread_px"], cheap["inc_flow_px"])
        s["phase_a"]["spearman_misread_vs_flow"] = {"rho": rho, "n": nn}
        rho, nn = spearman(pa["d_tot_px"], cheap["inc_flow_px"])
        s["phase_a"]["spearman_corrected_vs_flow"] = {"rho": rho, "n": nn}
        rho, nn = spearman(pa["d_tot_misread_px"], panel["d_vo_median_px"])
        s["phase_a"]["spearman_misread_vs_published_dvo"] = {"rho": rho, "n": nn}
        # decomposition shares
        fin = np.isfinite(pa["d_sim_px"]) & np.isfinite(pa["d_proj_px"]) & \
            np.isfinite(pa["d_tot_px"]) & (pa["d_tot_px"] > 0)
        s["phase_a"]["share_sim"] = qt(pa["d_sim_px"][fin] / pa["d_tot_px"][fin])
        s["phase_a"]["share_proj"] = qt(pa["d_proj_px"][fin] / pa["d_tot_px"][fin])
        # pairwise structure among corrected quantities
        s["phase_a"]["pairwise_spearman"] = {}
        for i, qa in enumerate(QUANT):
            for qb in QUANT[i + 1:]:
                rho, nn = spearman(pa[qa], pa[qb])
                s["phase_a"]["pairwise_spearman"][f"{qa}|{qb}"] = {"rho": rho, "n": nn}
        s["phase_a"]["timing_ms"] = {k: qt(pa[k]) for k in
                                     ("t_klt_ms", "t_magsac_ms", "t_refit_ms")}

        # ---- Phase B1: fused-attitude per-interval rotation -----------------------------
        att_path = ATTITUDE_PATH[seq]
        err_att = np.full(n, np.nan)
        if att_path.exists():
            att = read_cols(att_path, ["timestamp_s", "yaw_compass_deg"])
            ta, ya = att["timestamp_s"], att["yaw_compass_deg"]
            gaps = np.diff(ta)
            att_info = {"provenance": ATTITUDE_PROVENANCE[seq],
                        "rate_hz": float(1.0 / np.median(gaps)),
                        "max_gap_s": float(np.max(gaps)),
                        "covers_camera_span": bool(ta[0] <= t_cam[0] and ta[-1] >= t_cam[-1])}
            att_cam = unwrap_interp_deg(t_cam, ta, ya)
            d_att = np.full(n, np.nan)
            d_att[1:] = np.diff(att_cam)
            m = has_inc & np.isfinite(d_att)
            err_att[m] = np.abs(sign * sidecar["inc_rotation_deg"][m] - d_att[m])
            err_ref = np.full(n, np.nan)
            mr = m & np.isfinite(pa["rot_b_deg"])
            err_ref[mr] = np.abs(sign * pa["rot_b_deg"][mr] - d_att[mr])
            b1: dict = {"attitude": att_info,
                        "err_att_vo": qt(err_att), "err_att_ref": qt(err_ref)}
            mm = np.isfinite(err_att) & np.isfinite(err_ref)
            b1["matched_median_vo_minus_ref_deg"] = float(
                np.median(err_att[mm] - err_ref[mm])) if mm.sum() else None
            b1["spearman_vs_err_att_vo"] = {}
            for q in QUANT:
                rho, nn = spearman(pa[q], err_att)
                b1["spearman_vs_err_att_vo"][q] = {"rho": rho, "n": nn}
            rho, nn = spearman(labels["t_a"], err_att)
            b1["spearman_t_a_vs_err_att_vo"] = {"rho": rho, "n": nn}
            b1["ap_vs_err_att_tail"] = {q: ap_vs_tail(pa[q], err_att) for q in QUANT}
            b1["ap_vs_err_att_tail"]["t_a"] = ap_vs_tail(labels["t_a"], err_att)
            b1["ap_vs_err_att_tail"]["published_d_vo"] = ap_vs_tail(
                panel["d_vo_median_px"], err_att)
            # the frozen T-D p99.5 binary as a classifier of the same descriptive tail
            dec = json.load((REPO / "evaluations/exp-conf-001-dev/"
                             "fitting_target_decision.json").open(encoding="utf-8"))
            thr = dec["t_a"]["thresholds"]["p99.5"]
            td = np.where(labels["gradable_a"], labels["t_a"] > thr, False)
            tail = tail_flags(err_att)
            mm = np.isfinite(err_att) & labels["gradable_a"]
            tp = int(np.sum(td & tail & mm))
            b1["td_p995_vs_err_att_tail"] = {
                "n_td_pos": int(np.sum(td & mm)), "n_tail": int(np.sum(tail & mm)),
                "true_pos": tp,
                "precision": float(tp / max(1, np.sum(td & mm))),
                "recall": float(tp / max(1, np.sum(tail & mm)))}
            # conditioned deciles: err_att by decile of rot_dis
            mq = np.isfinite(pa["rot_dis_deg"]) & np.isfinite(err_att)
            if mq.sum() > 500:
                dec = np.percentile(pa["rot_dis_deg"][mq], np.arange(0, 101, 10))
                med = []
                for k in range(10):
                    inb = mq & (pa["rot_dis_deg"] >= dec[k]) & (pa["rot_dis_deg"] <= dec[k + 1])
                    med.append(float(np.median(err_att[inb])) if inb.sum() else None)
                b1["err_att_median_by_rot_dis_decile"] = med
            s["phase_b1"] = b1
        else:
            s["phase_b1"] = {"attitude": None}

        # ---- Phase B2: RTK at the 1.0 s window scale ------------------------------------
        gt = read_cols(REPO / f"datasets/{seq}/groundtruth.csv",
                       ["timestamp_s", "heading_deg"])
        h_cam = unwrap_interp_deg(t_cam, gt["timestamp_s"], gt["heading_deg"])
        inc_vo = np.where(has_inc, sign * sidecar["inc_rotation_deg"], 0.0)
        inc_ref = np.where(has_inc & np.isfinite(pa["rot_b_deg"]),
                           sign * np.nan_to_num(pa["rot_b_deg"]), 0.0)
        bad = (~(has_inc & np.isfinite(pa["rot_b_deg"]))).astype(int)
        cum_vo, cum_ref = np.cumsum(inc_vo), np.cumsum(inc_ref)
        cum_bad = np.cumsum(bad)
        cum_dis = np.cumsum(np.where(np.isfinite(pa["rot_dis_deg"]),
                                     np.nan_to_num(pa["rot_dis_deg"]), 0.0))
        j = np.searchsorted(t_cam, t_cam + WINDOW_S, side="left")
        k = np.arange(n)
        ok = (j < n)
        kk, jj = k[ok], j[ok]
        clean = (cum_bad[jj] - cum_bad[kk]) == 0
        ew_vo = np.abs((cum_vo[jj] - cum_vo[kk]) - (h_cam[jj] - h_cam[kk]))[clean]
        ew_ref = np.abs((cum_ref[jj] - cum_ref[kk]) - (h_cam[jj] - h_cam[kk]))[clean]
        w_dis = (cum_dis[jj] - cum_dis[kk])[clean]
        rho_w = spearman(w_dis, ew_vo)
        s["phase_b2"] = {"n_clean_windows": int(clean.sum()),
                         "err_1s_vo_deg": qt(ew_vo), "err_1s_ref_deg": qt(ew_ref),
                         "median_vo_minus_ref_deg": float(np.median(ew_vo - ew_ref)),
                         "spearman_windowsum_rot_dis_vs_err_vo":
                             {"rho": rho_w[0], "n": rho_w[1]}}

        # ---- Phase B3: T-B / T-C --------------------------------------------------------
        s["phase_b3"] = {}
        for tname in ("t_b", "t_c"):
            e = {}
            for q in QUANT:
                rho, nn = spearman(pa[q], labels[tname])
                e[q] = {"rho": rho, "n": nn}
            s["phase_b3"][tname] = e

        # ---- Phase C: counterfactual scoring --------------------------------------------
        cf_path = PHASE_C_DIR / f"{REFINE_RUN[seq]}.csv"
        if cf_path.exists():
            cf = load_phase(cf_path, n, ["rot_a_deg", "d_tot_px", "d_sim_px", "d_proj_px",
                                         "rot_dis_deg", "flowvec_dis_px"])
            c: dict = {"run": REFINE_RUN[seq]}
            for q in ("d_tot_px", "d_sim_px", "d_proj_px", "rot_dis_deg", "flowvec_dis_px"):
                mm = np.isfinite(pa[q]) & np.isfinite(cf[q])
                c[q] = {"frozen": qt(pa[q][mm]), "counterfactual": qt(cf[q][mm]),
                        "matched_median_delta": float(np.median(cf[q][mm] - pa[q][mm]))}
            # attitude error of the counterfactual run's own production readout
            if att_path.exists():
                cf_side = read_cols(REPO / f"runs/{REFINE_RUN[seq]}/logical_transform.csv",
                                    ["inc_rotation_deg"])
                cf_ev = load_frames(REPO / f"runs/{REFINE_RUN[seq]}")["event"]
                cfi = (cf_ev != "init") & (cf_ev != "restart") & np.isfinite(d_att)
                e_cf = np.full(n, np.nan)
                e_cf[cfi] = np.abs(sign * cf_side["inc_rotation_deg"][cfi] - d_att[cfi])
                mm = np.isfinite(err_att) & np.isfinite(e_cf)
                c["err_att"] = {"frozen": qt(err_att[mm]), "counterfactual": qt(e_cf[mm]),
                                "matched_median_delta_deg": float(
                                    np.median(e_cf[mm] - err_att[mm]))}
            fr_ev = events
            c["restarts"] = {"frozen": [int(i) for i in np.where(fr_ev == "restart")[0]],
                             "counterfactual": [int(i) for i in np.where(
                                 load_frames(REPO / f"runs/{REFINE_RUN[seq]}")["event"]
                                 == "restart")[0]]}
            s["phase_c"] = c

        # ---- Phase D: cheap signals against everything ----------------------------------
        d: dict = {}
        targets = {"A1_fb": panel["fb_median_px"], "A1_ho_ste": panel["ho_ste_median_px"],
                   "A1_boot": panel["boot_rot_sd_deg"],
                   "A2img_corrected_d_tot": pa["d_tot_px"],
                   "A2img_published_d_vo": panel["d_vo_median_px"],
                   "A2img_misread": pa["d_tot_misread_px"],
                   "nav_rot_dis": pa["rot_dis_deg"], "nav_flowvec_dis": pa["flowvec_dis_px"],
                   "nav_dir_dis": pa["dir_dis_deg"], "nav_logscale_dis": pa["logscale_dis"],
                   "nav_d_sim": pa["d_sim_px"], "proj_d_proj": pa["d_proj_px"]}
        if np.isfinite(err_att).sum() > 100:
            targets["ext_err_att"] = err_att
        for sig, orient in SIGNALS.items():
            sv = orient * cheap[sig]
            e = {}
            for tname, tv in targets.items():
                rho, nn = spearman(sv, tv)
                e[tname] = {"rho": rho, "ap": ap_vs_tail(sv, tv).get("ap")}
            d[sig] = e
        s["phase_d"] = d

        # ---- audited cases under the corrected convention -------------------------------
        s["audit_cases"] = {}
        for cid, cseq, ci in audit_cases:
            if cseq != seq:
                continue
            row = {q: (float(pa[q][ci]) if np.isfinite(pa[q][ci]) else None) for q in QUANT}
            row["published_d_vo"] = (float(panel["d_vo_median_px"][ci])
                                     if np.isfinite(panel["d_vo_median_px"][ci]) else None)
            row["inc_perspective"] = float(sidecar["inc_perspective"][ci])
            row["err_att_deg"] = (float(err_att[ci]) if np.isfinite(err_att[ci]) else None)
            s["audit_cases"][cid] = row

        report["sequences"][seq] = s
        print(f"[analysis] {seq} done", flush=True)

    out = REPO / "evaluations/exp-conf-003/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"[analysis] written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
