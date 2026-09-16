"""`EXP-VO-010` figures: does refitting the RANSAC model over its inlier set help?

Ten panels across five figures, matching the record's required-figure list:

    fig1  minimal-sample vs refined per-frame rotation-error distributions
    fig2  cumulative heading error vs distance; aligned heading error vs distance
    fig3  GT + refine=false + refine=true trajectories; position error vs distance
    fig4  RPE comparison; accumulated visual scale; inlier count vs refinement-induced
          rotation change; runtime distributions
    fig5  the summary chart -- refine=false vs refine=true across every window and model

Every figure is stamped with the run ids it was built from, so a figure that reaches the thesis can
be traced to exact run records without reading this file.

Usage::

    python evaluation/tools/exp_vo_010/plot_exp_vo_010.py --out evaluations/exp-vo-010/figures
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(EVALUATION_DIR / "tools" / "exp_vo_008"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_a7 = _load("exp_vo_007_analyse", EVALUATION_DIR / "tools" / "exp_vo_007" / "analyse.py")
_a10 = _load("exp_vo_010_analyse", Path(__file__).resolve().parent / "analyse.py")
_hm = _load("exp_vo_010_heading_metrics", Path(__file__).resolve().parent / "heading_metrics.py")

import recompose as rc                                                   # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.runrecord import load_run_record                            # noqa: E402
from rotation_diagnostic import gt_heading_increments                    # noqa: E402

WINDOWS = ("hkairport01-a", "hkairport01-b", "amtown01-c")
MODELS = ("affine", "homography")
RPE_LENGTHS = [10, 25, 50, 100, 200, 400]

COLOUR = {("affine", False): "tab:orange", ("affine", True): "tab:red",
          ("homography", False): "tab:cyan", ("homography", True): "tab:blue"}
LABEL = {("affine", False): "affine, minimal sample", ("affine", True): "affine, refined",
         ("homography", False): "homography, minimal sample",
         ("homography", True): "homography, refined"}


def stamp(fig, text: str) -> None:
    fig.text(0.005, 0.004, text, fontsize=5.5, color="0.35", ha="left", va="bottom")


def per_frame_rotation(window: str, model: str, refine: bool, width=1224, height=1024):
    """Per-frame rotation error against RTK heading, from the run's own sidecar."""
    rd = _a10.run_dir(window, model, refine)
    if not (rd / "logical_transform.csv").exists():
        return None
    ds = load_dataset(REPO / _a10.WINDOWS[window])
    rec = rc.load_recording(rd)
    rr = load_run_record(rd)
    cx, cy = width / 2.0, height / 2.0
    Ginv = np.linalg.inv(rec.G)
    inc = np.zeros(len(rec))
    for k in range(1, len(rec)):
        inc[k] = rc.polar_rotation(rc._jacobian(rec.G[k - 1] @ Ginv[k], cx, cy))
    dgt = gt_heading_increments(ds, rr.timestamps_s)
    sign = 1.0 if np.corrcoef(inc[1:], dgt[1:])[0, 1] > 0 else -1.0
    return _hm.wrap180(np.degrees(sign * inc[1:] - dgt[1:]))


def refinement_csv(window: str, model: str):
    p = _a10.run_dir(window, model, True) / "refinement.csv"
    if not p.exists():
        return None
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {"inliers": np.array([int(r["inliers"]) for r in rows], dtype=float),
            "d_rot": np.array([float(r["d_rot_deg"]) for r in rows]),
            "d_trans": np.array([float(r["d_trans_px"]) for r in rows])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", default="evaluations/exp-vo-010/comparison.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    comp = json.loads((REPO / a.comparison).read_text())["windows"]
    windows = [w for w in WINDOWS if w in comp]

    # ------------------------------------------------------- fig1: per-frame rotation, the H1 test
    fig, axes = plt.subplots(1, len(windows), figsize=(5.2 * len(windows), 4.2), squeeze=False)
    for j, w in enumerate(windows):
        ax = axes[0][j]
        for model in MODELS:
            for refine in (False, True):
                err = per_frame_rotation(w, model, refine)
                if err is None:
                    continue
                ax.hist(err, bins=200, range=(-1.0, 1.0), histtype="step", density=True,
                        color=COLOUR[(model, refine)],
                        ls="-" if refine else "--", lw=1.3 if refine else 1.0,
                        label=f"{LABEL[(model, refine)]}  sd={err.std(ddof=1):.4f}°")
        ax.set_title(f"{w} — per-frame rotation error")
        ax.set_xlabel("estimated − RTK heading increment (°/frame)")
        ax.set_ylabel("density")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.suptitle("EXP-VO-010 H1: refitting the model over its inlier set, per-frame rotation",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    stamp(fig, "runs/<window>-<model>-rigid-{,refine-}v1 · rotation read as the polar rotation of "
               "the incremental Jacobian at the image centre, exactly as RigidNavigationState does "
               "· GT = RTK heading, used to measure only")
    fig.savefig(out / "fig1_per_frame_rotation.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- fig2: heading over distance
    fig, axes = plt.subplots(2, len(windows), figsize=(5.2 * len(windows), 7.6), squeeze=False)
    for j, w in enumerate(windows):
        ds = load_dataset(REPO / _a10.WINDOWS[w])
        ax_cum, ax_al = axes[0][j], axes[1][j]
        for model in MODELS:
            for refine in (False, True):
                err = per_frame_rotation(w, model, refine)
                key = f"{model}-refine{'On' if refine else 'Off'}"
                if err is None or key not in comp[w]:
                    continue
                cum = np.cumsum(err)
                ax_cum.plot(np.arange(cum.size), cum, color=COLOUR[(model, refine)],
                            ls="-" if refine else "--", lw=1.1,
                            label=f"{LABEL[(model, refine)]}  end={cum[-1]:+.1f}°")
                # Aligned heading error about its own constant offset -- the corrected metric.
                h = comp[w][key]["heading"]
                ev = _a7.evaluate(_a10.run_dir(w, model, refine), ds)
                gt = ev["_gt"]
                dist = np.concatenate([[0.0],
                                       np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
                ax_al.plot(dist, np.full(dist.size, h["aligned_heading_rms_deg"]),
                           color=COLOUR[(model, refine)], ls="-" if refine else "--", lw=1.1,
                           label=f"{LABEL[(model, refine)]}  rms={h['aligned_heading_rms_deg']:.2f}°")
        ax_cum.axhline(0, color="0.4", lw=0.8)
        ax_cum.set_title(f"{w} — cumulative heading error")
        ax_cum.set_xlabel("frame")
        ax_cum.set_ylabel("accumulated error (°)")
        ax_cum.legend(fontsize=6)
        ax_cum.grid(alpha=0.3)
        ax_al.set_title(f"{w} — offset-aligned heading RMS")
        ax_al.set_xlabel("ground-truth distance travelled (m)")
        ax_al.set_ylabel("aligned heading RMS (°)")
        ax_al.legend(fontsize=6)
        ax_al.grid(alpha=0.3)
    fig.suptitle("EXP-VO-010 H2: heading, with the constant frame offset removed (EXP-VO-009 R2)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.96))
    stamp(fig, "Raw yaw RMSE is NOT plotted: EXP-VO-009 R2 measured it to be ~95 % a constant "
               "frame offset. The lower panel is the tracking component underneath it.")
    fig.savefig(out / "fig2_heading.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- fig3: trajectories + error
    fig, axes = plt.subplots(2, len(windows), figsize=(5.2 * len(windows), 8.2), squeeze=False)
    for j, w in enumerate(windows):
        ds = load_dataset(REPO / _a10.WINDOWS[w])
        ax_tr, ax_er = axes[0][j], axes[1][j]
        drawn_gt = False
        for model in MODELS:
            for refine in (False, True):
                key = f"{model}-refine{'On' if refine else 'Off'}"
                if key not in comp[w]:
                    continue
                ev = _a7.evaluate(_a10.run_dir(w, model, refine), ds)
                al, gt = ev["_aligned"], ev["_gt"]
                if not drawn_gt:
                    ax_tr.plot(gt[:, 0], gt[:, 1], color="0.25", lw=1.6, label="RTK ground truth")
                    ax_tr.plot(gt[0, 0], gt[0, 1], "o", color="0.25", ms=5)
                    drawn_gt = True
                ax_tr.plot(al[:, 0], al[:, 1], color=COLOUR[(model, refine)],
                           ls="-" if refine else "--", lw=1.0, alpha=0.9,
                           label=f"{LABEL[(model, refine)]}  {100 * ev['ate_rmse_normalised']:.3f} %")
                dist = np.concatenate([[0.0],
                                       np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
                ax_er.plot(dist, np.linalg.norm(al - gt, axis=1), color=COLOUR[(model, refine)],
                           ls="-" if refine else "--", lw=1.0, label=LABEL[(model, refine)])
        ax_tr.set_title(f"{w} — RIGID_MOTION, refine off vs on")
        ax_tr.set_xlabel("east (m)")
        ax_tr.set_ylabel("north (m)")
        ax_tr.set_aspect("equal", adjustable="datalim")
        ax_tr.legend(fontsize=6)
        ax_tr.grid(alpha=0.3)
        ax_er.set_title(f"{w} — position error against distance")
        ax_er.set_xlabel("ground-truth distance travelled (m)")
        ax_er.set_ylabel("aligned position error (m)")
        ax_er.legend(fontsize=6)
        ax_er.grid(alpha=0.3)
    fig.suptitle("EXP-VO-010 H3: trajectory and error, one global Sim(2) alignment (DEC-003)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.96))
    stamp(fig, "naveval unmodified · refine=false arms are the committed EXP-VO-006/007 baselines")
    fig.savefig(out / "fig3_trajectories.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- fig4: RPE, scale, delta-vs-inliers, cost
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
    ax = axes[0][0]
    xs = np.arange(len(RPE_LENGTHS))
    for j, w in enumerate(windows):
        for model in MODELS:
            for refine in (False, True):
                key = f"{model}-refine{'On' if refine else 'Off'}"
                if key not in comp[w]:
                    continue
                vals = [comp[w][key].get(f"rpe_{L}m") or np.nan for L in RPE_LENGTHS]
                ax.plot(xs, vals, marker="os^"[j % 3], color=COLOUR[(model, refine)],
                        ls="-" if refine else "--", lw=1.0, ms=3.5, alpha=0.85,
                        label=f"{model[:4]}·{'on' if refine else 'off'}·{w.split('-')[-1]}")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(L) for L in RPE_LENGTHS])
    ax.set_xlabel("sub-trajectory length (m)")
    ax.set_ylabel("RPE RMSE (m)")
    ax.set_title("Relative pose error")
    ax.legend(fontsize=5, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    for w in windows:
        for model in MODELS:
            for refine in (False, True):
                p = _a10.run_dir(w, model, refine) / "logical_transform.csv"
                if not p.exists():
                    continue
                with p.open(newline="") as f:
                    acc = np.array([float(r["rigid_accum_scale"]) for r in csv.DictReader(f)])
                ax.plot(acc, color=COLOUR[(model, refine)], ls="-" if refine else "--", lw=0.9,
                        alpha=0.85, label=f"{model[:4]}·{'on' if refine else 'off'}·"
                                          f"{w.split('-')[-1]}  {acc[-1]:.2f}×")
    ax.axhspan(0.94, 1.01, color="0.85", zorder=0, label="physical envelope (LIT-VO-003 §4)")
    ax.set_yscale("log")
    ax.set_xlabel("frame")
    ax.set_ylabel("accumulated visual scale (observed, never applied)")
    ax.set_title("H5: does refinement fix scale?")
    ax.legend(fontsize=5, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    for w in windows:
        for model in MODELS:
            d = refinement_csv(w, model)
            if d is None:
                continue
            ax.scatter(d["inliers"], np.abs(d["d_rot"]), s=1.5, alpha=0.15,
                       color=COLOUR[(model, True)],
                       label=f"{model} · {w.split('-')[-1]}")
    ax.set_xlabel("inlier count the refiner was given")
    ax.set_ylabel("|rotation(refined) − rotation(minimal)| (°)")
    ax.set_yscale("log")
    ax.set_title("Phase 4: how far refinement moves the model, against support")
    leg = ax.legend(fontsize=6, markerscale=6)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    labels, vals, colours = [], [], []
    for w in windows:
        for model in MODELS:
            for refine in (False, True):
                key = f"{model}-refine{'On' if refine else 'Off'}"
                v = comp[w].get(key, {}).get("ms_per_frame_median")
                if v is None:
                    continue
                labels.append(f"{model[:4]}·{'on' if refine else 'off'}\n{w.split('-')[-1]}")
                vals.append(v)
                colours.append(COLOUR[(model, refine)])
    ax.bar(np.arange(len(vals)), vals, color=colours)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylabel("median ms/frame (end to end)")
    ax.set_title("Runtime — development laptop only, no target-hardware claim")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("EXP-VO-010: RPE, scale, parameter change and cost", fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.965))
    stamp(fig, "Run-record ms/frame is cross-session and carries EXP-VO-002's confound; "
               "timing.json is the matched-conditions measurement and is what the record quotes.")
    fig.savefig(out / "fig4_rpe_scale_delta_runtime.png", dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- fig5: the summary chart
    metrics = [("normalised ATE (%)", lambda r: 100 * r["ate_rmse_normalised"]),
               ("per-frame rotation SD (°)", lambda r: r["heading"].get("per_frame_sd_deg")),
               ("aligned heading RMS (°)", lambda r: r["heading"]["aligned_heading_rms_deg"]),
               ("|cumulative drift| (°)",
                lambda r: abs(r["heading"].get("cumulative_drift_deg", np.nan)))]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.6))
    for ax, (name, get) in zip(axes, metrics):
        keys, off, on, cols = [], [], [], []
        for w in windows:
            for model in MODELS:
                ro = comp[w].get(f"{model}-refineOff")
                rn = comp[w].get(f"{model}-refineOn")
                if not ro or not rn:
                    continue
                keys.append(f"{model[:4]}\n{w.split('-')[-1]}")
                off.append(get(ro))
                on.append(get(rn))
                cols.append(COLOUR[(model, True)])
        x = np.arange(len(keys))
        ax.bar(x - 0.2, off, 0.4, color="0.75", label="minimal sample (refine off)")
        ax.bar(x + 0.2, on, 0.4, color=cols, label="refined (refine on)")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=6)
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=6)
    fig.suptitle("EXP-VO-010 summary: refine=false vs refine=true, every window and model",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    stamp(fig, "Lower is better in every panel. Aligned heading RMS and cumulative drift are the "
               "corrected heading quantities (EXP-VO-009 R2); raw yaw RMSE is deliberately absent.")
    fig.savefig(out / "fig5_summary.png", dpi=140)
    plt.close(fig)

    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
