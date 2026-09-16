"""EXP-CONF-002 Phase 3: frozen analyses over the reference diagnostic panel outputs.

Implements exactly the analysis list frozen in the EXP-CONF-002 front half. No learned
combination of members is fitted; agreement between members is evidence, not ground truth.
numpy only; statistics reused from EXP-CONF-001's committed p4_analysis (rankdata/spearman/
average_precision), unchanged.

Outputs: evaluations/exp-conf-002/panel/analysis.json
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
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_001"))
from p4_analysis import (SIGNALS, average_precision, load_frames, load_labels,   # noqa: E402
                         rankdata, spearman)

SEQS = ("hkairport01-b", "amtown01-d")
LABELS_DIR = REPO / "evaluations/exp-conf-001-dev/labels"
PANEL_DIR = REPO / "evaluations/exp-conf-002/panel"
DECISION = REPO / "evaluations/exp-conf-001-dev/fitting_target_decision.json"

# Panel primary quantities, all oriented higher = worse (frozen); d_vo reported alongside R2.
PANEL_Q = ("fb_median_px", "ho_ste_median_px", "boot_rot_sd_deg", "xchk_grid_px")
PANEL_ALL = PANEL_Q + ("d_vo_median_px",)
TAIL_FRAC = 0.02          # frozen rank-based tail convention
RESTART_WINDOW = 15


def load_panel(seq: str, n_frames: int) -> dict[str, np.ndarray]:
    out = {q: np.full(n_frames, np.nan) for q in PANEL_ALL}
    out["wall_ms"] = np.full(n_frames, np.nan)
    with (PANEL_DIR / f"{seq}.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r["frame_index"])
            for q in PANEL_ALL + ("wall_ms",):
                v = r.get(q, "")
                if v not in ("", None):
                    out[q][i] = float(v)
    return out


def tail_flags(x: np.ndarray) -> np.ndarray:
    """Worst-TAIL_FRAC flags among finite values (per sequence, rank-based)."""
    flags = np.zeros(len(x), dtype=bool)
    m = np.isfinite(x)
    if m.sum() < 50:
        return flags
    thr = np.quantile(x[m], 1.0 - TAIL_FRAC)
    flags[m] = x[m] > thr
    return flags


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.sum(a | b)
    return float(np.sum(a & b) / u) if u else float("nan")


def main() -> int:
    decision = json.load(DECISION.open(encoding="utf-8"))
    thresholds = decision["t_a"]["thresholds"]

    report: dict = {"tail_frac": TAIL_FRAC, "sequences": {}, "pooled": {}}
    pooled: dict[str, list] = {}

    for seq in SEQS:
        run_dir = REPO / f"runs/{seq}-homography-rigid-conf-v1"
        frames = load_frames(run_dir)
        labels = load_labels(LABELS_DIR / f"{seq}.csv")
        n = len(frames["event"])
        panel = load_panel(seq, n)
        events = frames["event"]

        s: dict = {"n_frames": n}
        s["coverage"] = {q: int(np.sum(np.isfinite(panel[q]))) for q in PANEL_ALL}
        s["wall_ms"] = {
            "median": float(np.nanmedian(panel["wall_ms"])),
            "p95": float(np.nanpercentile(panel["wall_ms"], 95)),
        }

        # -- 1. pairwise Spearman among panel quantities --------------------------------
        s["pairwise_spearman"] = {}
        for i, qa in enumerate(PANEL_ALL):
            for qb in PANEL_ALL[i + 1:]:
                rho, m = spearman(panel[qa], panel[qb])
                s["pairwise_spearman"][f"{qa}|{qb}"] = {"rho": rho, "n": m}

        # -- 2. tail agreement ------------------------------------------------------------
        flags = {q: tail_flags(panel[q]) for q in PANEL_ALL}
        s["tail_agreement"] = {}
        for i, qa in enumerate(PANEL_ALL):
            for qb in PANEL_ALL[i + 1:]:
                inter = int(np.sum(flags[qa] & flags[qb]))
                exp = TAIL_FRAC**2 * min(np.isfinite(panel[qa]).sum(),
                                         np.isfinite(panel[qb]).sum())
                s["tail_agreement"][f"{qa}|{qb}"] = {
                    "jaccard": jaccard(flags[qa], flags[qb]),
                    "intersection": inter, "expected_if_independent": float(exp),
                }
        member_flags = np.stack([flags[q] for q in PANEL_Q])
        n_tails = member_flags.sum(axis=0)
        s["consensus_counts"] = {str(k): int(np.sum(n_tails == k)) for k in range(5)}

        # -- 3-4. panel vs old targets ------------------------------------------------------
        s["vs_targets"] = {}
        for q in PANEL_ALL:
            e: dict = {}
            for tname, tvals in (("t_a", labels["t_a"]), ("t_b", labels["t_b"]),
                                 ("t_c", labels["t_c"])):
                rho, m = spearman(panel[q], tvals)
                e[f"spearman_{tname}"] = {"rho": rho, "n": m}
            for sev, thr in thresholds.items():
                lab = np.where(labels["gradable_a"], (labels["t_a"] > thr), np.nan)
                mm = np.isfinite(panel[q]) & np.isfinite(lab)
                ap, n_pos, n_eval = average_precision(panel[q][mm], lab[mm].astype(bool))
                e[f"aucpr_td_{sev}"] = {"ap": ap, "n_pos": n_pos, "n_eval": n_eval,
                                        "base": n_pos / n_eval if n_eval else None}
            s["vs_targets"][q] = e

        # T-D agreement/disagreement matrix vs the >=3-member consensus (frozen convention)
        td = np.where(labels["gradable_a"], labels["t_a"] > thresholds["p99.5"], False)
        consensus = n_tails >= 3
        any_tail = n_tails >= 1
        s["td_vs_consensus"] = {
            "td_and_consensus": int(np.sum(td & consensus)),
            "td_and_any_tail": int(np.sum(td & any_tail)),
            "td_and_zero_tails": int(np.sum(td & (n_tails == 0))),
            "consensus_and_not_td": int(np.sum(consensus & ~td & labels["gradable_a"])),
            "n_td": int(np.sum(td)), "n_consensus": int(np.sum(consensus)),
        }

        # -- 5. behaviour around restarts ---------------------------------------------------
        s["restarts"] = []
        pct = {q: np.full(n, np.nan) for q in PANEL_ALL}
        for q in PANEL_ALL:
            m = np.isfinite(panel[q])
            if m.sum():
                pct[q][m] = rankdata(panel[q][m]) / m.sum()
        for r in np.where(events == "restart")[0]:
            lead = {}
            for q in PANEL_ALL:
                lo = max(0, r - RESTART_WINDOW)
                lead[q] = [None if not np.isfinite(pct[q][i]) else round(float(pct[q][i]), 4)
                           for i in range(lo, r + 1)]
            s["restarts"].append({"frame": int(r), "percentile_profile": lead})

        # -- 6. cheap signals vs panel ------------------------------------------------------
        s["cheap_vs_panel"] = {}
        for sig, orient in SIGNALS.items():
            sc = orient * frames[sig]
            e = {}
            for q in PANEL_ALL:
                rho, m = spearman(sc, panel[q])
                e[f"spearman_{q}"] = {"rho": rho, "n": m}
            # detection of each panel member's tail
            for q in PANEL_ALL:
                mm = np.isfinite(sc) & np.isfinite(panel[q])
                ap, n_pos, n_eval = average_precision(sc[mm], flags[q][mm])
                e[f"aucpr_tail_{q}"] = {"ap": ap, "n_pos": n_pos,
                                        "base": n_pos / n_eval if n_eval else None}
            # detection of old T-D p99.5 on the same evaluable frames (comparison basis)
            lab = np.where(labels["gradable_a"], labels["t_a"] > thresholds["p99.5"], np.nan)
            mm = np.isfinite(sc) & np.isfinite(lab)
            ap, n_pos, n_eval = average_precision(sc[mm], lab[mm].astype(bool))
            e["aucpr_td_p99.5"] = {"ap": ap, "n_pos": n_pos,
                                   "base": n_pos / n_eval if n_eval else None}
            s["cheap_vs_panel"][sig] = e

        report["sequences"][seq] = s

        # pool raw columns for pooled Spearman (rank within sequence is not poolable;
        # pooled uses raw values with sequences concatenated, stated as such)
        for q in PANEL_ALL:
            pooled.setdefault(q, []).append(panel[q])
        pooled.setdefault("t_a", []).append(labels["t_a"])
        pooled.setdefault("gradable_a", []).append(labels["gradable_a"])
        for sig in SIGNALS:
            pooled.setdefault(sig, []).append(frames[sig])

    # -- pooled panel-vs-T-A and cheap-vs-panel (concatenated raw values) ------------------
    cat = {k: np.concatenate(v) for k, v in pooled.items()}
    pl: dict = {"note": "raw values concatenated across sequences (not rank-pooled)"}
    for q in PANEL_ALL:
        rho, m = spearman(cat[q], cat["t_a"])
        pl[f"{q}_vs_t_a"] = {"rho": rho, "n": m}
    for sig, orient in SIGNALS.items():
        rho, m = spearman(orient * cat[sig], cat["d_vo_median_px"])
        pl[f"{sig}_vs_d_vo"] = {"rho": rho, "n": m}
    report["pooled"] = pl

    out = PANEL_DIR / "analysis.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"[analysis] written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
