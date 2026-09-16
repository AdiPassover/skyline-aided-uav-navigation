"""Diagnostic figures for one VO_ONLY-versus-INT comparison (`EXP-INT-002` §16), from run artifacts and
ground truth only. Axes are never rescaled to flatter the INT arm: both arms share one axis, the
error axis starts at zero, and the XY plot is equal-aspect.

1. ``xy_trajectories.png``  — GT / VO_ONLY / INT in the registered ENU frame, re-anchor markers,
   trusted-reference positions (stored, and their true capture positions), revisit events.
2. ``error_vs_time.png``    — position error of both arms against time; re-anchor frames marked;
   hard-loss / UNKNOWN stretches shaded.
3. ``snap_windows.png``     — ±30 s around each accepted re-anchor: both arms' error, the isolated
   post-snap error and the no-snap counterfactual from `snap_longitudinal.py` when present.
4. ``snap_geometry.png``    — true query-to-reference distance versus pre-snap error, one point per
   accepted correction (drift-correction case), coloured by the longitudinal class.
5. ``snap_table.csv`` / ``.md`` — frame | trigger | ref | q-ref distance | e_before | e_after | Δe_0 |
   post-10 s benefit | post-30 s benefit.

    python evaluation/tools/int/plot_int_arms.py --dataset datasets/<id> --vo-only runs/<vo> --int runs/<int> \\
        --out <dir> [--label INT_C0] [--longitudinal <dir with snap_longitudinal.json>] [--events <gt_revisit_events.json>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import evaluate_int_arms as ev  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--vo-only", required=True, type=Path)
    ap.add_argument("--int", dest="int_run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default="INT")
    ap.add_argument("--longitudinal", type=Path, default=None)
    ap.add_argument("--events", type=Path, default=None)
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    gt = ev.load_gt(a.dataset)
    vo_rows = ev.read_csv(a.vo_only / "alignment_frames.csv")
    in_rows = ev.read_csv(a.int_run / "alignment_frames.csv")
    events = ev.read_csv(a.int_run / "alignment_events.csv")
    vo = ev.error_curve(gt, vo_rows); it = ev.error_curve(gt, in_rows)
    f0 = int(vo["frame"][0]); _, e0, n0 = gt[f0]
    G = np.array([[gt[int(f)][1] - e0, gt[int(f)][2] - n0] for f in vo["frame"]])
    V = np.array([[ev._f(r["global_east_m"]), ev._f(r["global_north_m"])] for r in vo_rows])
    I = np.array([[ev._f(r["global_east_m"]), ev._f(r["global_north_m"])] for r in in_rows])
    I[~it["valid"]] = np.nan; V[~vo["valid"]] = np.nan
    idx = {int(f): i for i, f in enumerate(vo["frame"])}
    re_rows = ev.reanchor_table(gt, events, 50.0, 15.0, 20.0)
    refs = [(int(r["frame_index"]), ev._f(r["global_after_east_m"]), ev._f(r["global_after_north_m"]))
            for r in events if r["kind"] == "reference_inserted"]
    losses = [int(r["frame_index"]) for r in events if r["kind"] == "hard_loss"]
    long = json.loads((a.longitudinal / "snap_longitudinal.json").read_text()) if a.longitudinal and (a.longitudinal / "snap_longitudinal.json").exists() else None
    ev_json = json.loads(a.events.read_text()) if a.events and a.events.exists() else None
    title = a.title or f"{a.dataset.name}: VO_ONLY vs {a.label}"

    # 1. XY
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(G[:, 0], G[:, 1], "-", color="0.3", lw=1.2, label="ground truth")
    ax.plot(V[:, 0], V[:, 1], "-", color="tab:blue", lw=1.0, label="VO_ONLY")
    ax.plot(I[:, 0], I[:, 1], "-", color="tab:red", lw=1.0, label=a.label)
    if refs:
        ax.plot([r[1] for r in refs], [r[2] for r in refs], "s", ms=4, mfc="none", color="tab:green", label="trusted reference (stored)")
        ax.plot([G[idx[r[0]], 0] for r in refs if r[0] in idx], [G[idx[r[0]], 1] for r in refs if r[0] in idx], ".", ms=3, color="tab:green", label="reference true capture")
    for r in re_rows:
        i = idx[r["frame"]]
        ax.plot(G[i, 0], G[i, 1], "*", ms=12, color="tab:orange", mec="k", label="accepted re-anchor (GT)" if r is re_rows[0] else None)
    if ev_json:
        for e in ev_json["events"]:
            c = e["closest_approach"]; ax.plot(c["east_m"] - e0, c["north_m"] - n0, "o", ms=9, mfc="none", color="tab:purple", label="GT revisit event" if e is ev_json["events"][0] else None)
            ax.annotate(f"E{e['event_id']}", (c["east_m"] - e0, c["north_m"] - n0), textcoords="offset points", xytext=(6, 6), color="tab:purple")
    ax.plot(G[0, 0], G[0, 1], "k^", ms=8, label="start")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_title(title); ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(a.out / "xy_trajectories.png", dpi=150); plt.close(fig)

    # 2. error vs time
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(vo["t"], vo["err"], color="tab:blue", lw=1.2, label="VO_ONLY")
    ax.plot(it["t"], it["err"], color="tab:red", lw=1.2, label=a.label)
    ymax = np.nanmax(np.concatenate([vo["err"], it["err"]])) * 1.1
    for r in re_rows:
        ax.axvline(r["time_s"], color="tab:orange", lw=1, ls="--")
        ax.annotate(f"snap f{r['frame']}\nΔe₀ {r['delta_e_m']:+.1f} m" if not math.isnan(r["delta_e_m"]) else f"recovery f{r['frame']}",
                    (r["time_s"], ymax * 0.9), fontsize=7, ha="left", color="tab:orange")
    for lf in losses:
        ax.axvline(vo["t"][idx[lf]], color="k", lw=1, ls=":"); ax.annotate("hard loss", (vo["t"][idx[lf]], ymax * 0.97), fontsize=7)
    unknown = ~it["valid"]
    if unknown.any():
        ax.fill_between(it["t"], 0, ymax, where=unknown, color="0.85", label=f"{a.label} position UNKNOWN")
    ax.set_ylim(0, ymax); ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)"); ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title(title)
    fig.tight_layout(); fig.savefig(a.out / "error_vs_time.png", dpi=150); plt.close(fig)

    # 3. snap windows
    if re_rows:
        n = len(re_rows)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        for k, r in enumerate(re_rows):
            ax = axes[0, k]; t_s = r["time_s"]
            m = (vo["t"] >= t_s - 30) & (vo["t"] <= t_s + 30)
            ax.plot(vo["t"][m] - t_s, vo["err"][m], color="tab:blue", label="VO_ONLY")
            ax.plot(it["t"][m] - t_s, it["err"][m], color="tab:red", label=a.label)
            if long:
                s = next((s for s in long["snaps"] if s["frame"] == r["frame"]), None)
                series_p = a.longitudinal / f"snap_{r['frame']}_series.csv"
                if s and series_p.exists():
                    ser = ev.read_csv(series_p)
                    ts = np.array([float(x["since_snap_s"]) for x in ser]); ei = np.array([float(x["e_iso_m"]) for x in ser]); en = np.array([ev._f(x["e_nok_m"]) for x in ser])
                    mm = ts <= 30
                    ax.plot(ts[mm], ei[mm], "--", color="tab:red", lw=0.8, label="isolated snap (no later snap)")
                    if np.isfinite(en).any():
                        ax.plot(ts[mm], en[mm], "--", color="tab:blue", lw=0.8, label="counterfactual: no snap")
            ax.axvline(0, color="tab:orange", ls="--")
            ax.set_ylim(bottom=0); ax.set_xlabel("time since snap (s)"); ax.set_ylabel("error (m)"); ax.grid(alpha=0.3)
            ax.set_title(f"f{r['frame']} ({r['recovery_case']}): q→ref {r['gt_query_to_reference_m']:.1f} m, Δe₀ {r['delta_e_m']:+.1f} m", fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle(title); fig.tight_layout(); fig.savefig(a.out / "snap_windows.png", dpi=150); plt.close(fig)

    # 4. geometry scatter
    drift = [r for r in re_rows if r["recovery_case"] == "drift_correction"]
    cls = {s["frame"]: s["classification"] for s in long["snaps"]} if long else {}
    if drift:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        colors = {"A_persistently_beneficial": "tab:green", "B_beneficial_then_harmful": "tab:orange", "C_immediately_harmful": "tab:red", "D_neutral": "0.5"}
        for r in drift:
            c = colors.get(cls.get(r["frame"]), "k")
            ax.plot(r["gt_query_to_reference_m"], r["e_before_m"], "o", color=c, ms=8)
            ax.annotate(f"f{r['frame']}", (r["gt_query_to_reference_m"], r["e_before_m"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        lim = max([r["e_before_m"] for r in drift] + [r["gt_query_to_reference_m"] for r in drift]) * 1.2
        ax.plot([0, lim], [0, lim], "k:", lw=0.8, label="e_before = q→ref distance")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("true query→reference distance u (m)"); ax.set_ylabel("pre-snap error e_before (m)")
        for k, c in colors.items():
            ax.plot([], [], "o", color=c, label=k)
        ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_title(title, fontsize=9)
        fig.tight_layout(); fig.savefig(a.out / "snap_geometry.png", dpi=150); plt.close(fig)

    # 5. table
    rows = []
    for r in re_rows:
        s = next((s for s in (long["snaps"] if long else []) if s["frame"] == r["frame"]), None)
        h10 = s["horizons"].get("10s") if s else None; h30 = s["horizons"].get("30s") if s else None
        rows.append({"frame": r["frame"], "time_s": f"{r['time_s']:.1f}", "trigger": r["cause"], "case": r["recovery_case"],
                     "reference_id": r["reference_id"], "q_ref_distance_m": f"{r['gt_query_to_reference_m']:.2f}",
                     "e_before_m": "" if math.isnan(r["e_before_m"]) else f"{r['e_before_m']:.2f}", "e_after_m": f"{r['e_after_m']:.2f}",
                     "delta_e_0_m": "" if math.isnan(r["delta_e_m"]) else f"{r['delta_e_m']:+.2f}",
                     "post_10s_benefit_m": "" if not h10 or h10.get("delta_e_iso_m") is None else f"{-h10['delta_e_iso_m']:+.2f}",
                     "post_30s_benefit_m": "" if not h30 or h30.get("delta_e_iso_m") is None else f"{-h30['delta_e_iso_m']:+.2f}",
                     "class": cls.get(r["frame"], "") if long else ""})
    with (a.out / "snap_table.csv").open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        else:
            f.write("frame\n")
    with (a.out / "snap_table.md").open("w") as f:
        if rows:
            f.write("| " + " | ".join(rows[0].keys()) + " |\n|" + "---|" * len(rows[0]) + "\n")
            for r in rows:
                f.write("| " + " | ".join(str(v) for v in r.values()) + " |\n")
        else:
            f.write("(no accepted re-anchor)\n")
    print(f"figures written to {a.out}: {len(re_rows)} re-anchors, {len(refs)} references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
