"""`EXP-VO-009` figures: does direct Sim(2) fitting produce a cleaner rotation?

Nine panels across four figures, matching the record's required-figure list:

    fig1  per-frame yaw-error distribution by model; cumulative heading error against distance
    fig2  GT + rigid homography + rigid affine + rigid similarity trajectories; position error
          against distance
    fig3  RPE by sub-trajectory length; accumulated visual scale; track and inlier support;
          runtime distributions
    fig4  AMtown01 only: yaw error against lateral position error, the coupling EXP-VO-008
          measured at 83 %

Every figure is stamped with the run ids it was built from, so a figure that reaches the thesis can
be traced to exact run records without reading this file.

Usage::

    python evaluation/tools/exp_vo_009/plot_exp_vo_009.py --out evaluations/exp-vo-009/figures
"""
from __future__ import annotations

import argparse
import csv
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

# Both experiments name their driver `analyse`, so they are loaded by path rather than by a
# `sys.path` race that would silently bind whichever landed first.
import importlib.util                                                    # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_a7 = _load("exp_vo_007_analyse", EVALUATION_DIR / "tools" / "exp_vo_007" / "analyse.py")
_a9 = _load("exp_vo_009_analyse", Path(__file__).resolve().parent / "analyse.py")
evaluate7 = _a7.evaluate
WINDOWS, resolve = _a9.WINDOWS, _a9.resolve

from naveval.dataset import load_dataset                                 # noqa: E402

MODEL_COLOUR = {"homography": "tab:red", "affine": "tab:orange", "similarity": "tab:blue"}
MODEL_LABEL = {"homography": "homography (8 DoF)", "affine": "affine (6 DoF)",
               "similarity": "similarity (4 DoF)"}
MODELS = ("homography", "affine", "similarity")
RPE_LENGTHS = [10, 25, 50, 100, 200, 400]


def sidecar(window: str, model: str) -> dict:
    p = resolve(window, model, "rigid") / "logical_transform.csv"
    if not p.exists():
        return {}
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in rows])                # noqa: E731
    return {"acc": col("rigid_accum_scale"), "aniso": col("inc_anisotropy")[1:]}


def stamp(fig, text: str) -> None:
    fig.text(0.005, 0.004, text, fontsize=5.5, color="0.35", ha="left", va="bottom")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rotation", default="evaluations/exp-vo-009",
                    help="dir holding rotation_<window>.npz / .json")
    ap.add_argument("--comparison", default="evaluations/exp-vo-009/comparison.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rotdir = REPO / a.rotation
    comp = json.loads((REPO / a.comparison).read_text())["windows"]
    windows = [w for w in ("hkairport01-a", "hkairport01-b", "amtown01-c") if w in comp]

    # ----------------------------------------------------------------- fig1: rotation, the H1 test
    fig, axes = plt.subplots(2, len(windows), figsize=(5.0 * len(windows), 7.4), squeeze=False)
    for j, w in enumerate(windows):
        npz = rotdir / f"rotation_{w}.npz"
        ax_hist, ax_cum = axes[0][j], axes[1][j]
        if not npz.exists():
            ax_hist.text(0.5, 0.5, "no rotation data", ha="center", transform=ax_hist.transAxes)
            continue
        d = np.load(npz)
        dgt = d["dgt"]
        for m in MODELS:
            if m not in d:
                continue
            err = np.degrees(d[m][1:] - dgt[1:])
            ax_hist.hist(err, bins=160, range=(-1.0, 1.0), histtype="step", density=True,
                         color=MODEL_COLOUR[m],
                         label=f"{MODEL_LABEL[m]}  sd={err.std(ddof=1):.4f}°")
            ax_cum.plot(np.arange(err.size), np.cumsum(err), color=MODEL_COLOUR[m], lw=1.1,
                        label=f"{MODEL_LABEL[m]}  end={err.sum():+.1f}°")
        ax_hist.set_title(f"{w} — per-frame rotation error")
        ax_hist.set_xlabel("estimated − RTK heading increment (°/frame)")
        ax_hist.set_ylabel("density")
        ax_hist.legend(fontsize=7)
        ax_hist.grid(alpha=0.3)
        ax_cum.axhline(0, color="0.4", lw=0.8)
        ax_cum.set_title(f"{w} — cumulative heading error")
        ax_cum.set_xlabel("frame")
        ax_cum.set_ylabel("accumulated error (°)")
        ax_cum.legend(fontsize=7)
        ax_cum.grid(alpha=0.3)
    fig.suptitle("EXP-VO-009 H1: per-frame rotation, three motion models, identical pipeline",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.012, 1, 0.97))
    stamp(fig, "runs/<window>-{homography,affine,similarity}-rigid-v1 · rotation read as the polar "
               "rotation of the incremental Jacobian at the image centre, exactly as "
               "RigidNavigationState does · GT = RTK heading, used to measure only")
    fig.savefig(out / "fig1_rotation.png", dpi=140)
    plt.close(fig)

    # ----------------------------------------------------------------- fig2: trajectories + error
    fig, axes = plt.subplots(2, len(windows), figsize=(5.0 * len(windows), 8.0), squeeze=False)
    for j, w in enumerate(windows):
        ds = load_dataset(REPO / WINDOWS[w]["dataset"])
        ax_tr, ax_er = axes[0][j], axes[1][j]
        drawn_gt = False
        for m in MODELS:
            key = f"{m}-rigid"
            if key not in comp[w]:
                continue
            ev = evaluate7(resolve(w, m, "rigid"), ds)
            al, gt = ev["_aligned"], ev["_gt"]
            if not drawn_gt:
                ax_tr.plot(gt[:, 0], gt[:, 1], color="0.25", lw=1.6, label="RTK ground truth")
                ax_tr.plot(gt[0, 0], gt[0, 1], "o", color="0.25", ms=5)
                drawn_gt = True
            ax_tr.plot(al[:, 0], al[:, 1], color=MODEL_COLOUR[m], lw=1.0, alpha=0.9,
                       label=f"{MODEL_LABEL[m]}  {100 * ev['ate_rmse_normalised']:.3f} %")
            dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
            ax_er.plot(dist, np.linalg.norm(al - gt, axis=1), color=MODEL_COLOUR[m], lw=1.0,
                       label=MODEL_LABEL[m])
        ax_tr.set_title(f"{w} — RIGID_MOTION, by motion model")
        ax_tr.set_xlabel("east (m)")
        ax_tr.set_ylabel("north (m)")
        ax_tr.set_aspect("equal", adjustable="datalim")
        ax_tr.legend(fontsize=7)
        ax_tr.grid(alpha=0.3)
        ax_er.set_title(f"{w} — position error against distance travelled")
        ax_er.set_xlabel("ground-truth distance travelled (m)")
        ax_er.set_ylabel("aligned position error (m)")
        ax_er.legend(fontsize=7)
        ax_er.grid(alpha=0.3)
    fig.suptitle("EXP-VO-009 H2: trajectory and error, one global Sim(2) alignment (DEC-003)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.012, 1, 0.97))
    stamp(fig, "runs/<window>-{homography,affine,similarity}-rigid-v1 · naveval unmodified")
    fig.savefig(out / "fig2_trajectories.png", dpi=140)
    plt.close(fig)

    # ----------------------------------------------------------------- fig3: RPE, scale, support
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    ax = axes[0][0]
    width = 0.26
    xs = np.arange(len(RPE_LENGTHS))
    for i, m in enumerate(MODELS):
        for j, w in enumerate(windows):
            key = f"{m}-rigid"
            if key not in comp[w]:
                continue
            vals = [comp[w][key].get(f"rpe_{L}m") or np.nan for L in RPE_LENGTHS]
            ax.plot(xs, vals, marker="os^"[j % 3], color=MODEL_COLOUR[m], lw=1.1, ms=4,
                    alpha=0.55 + 0.2 * j, label=f"{m} · {w}")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(L) for L in RPE_LENGTHS])
    ax.set_xlabel("sub-trajectory length (m)")
    ax.set_ylabel("RPE RMSE (m)")
    ax.set_title("Relative pose error")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    for j, w in enumerate(windows):
        for m in MODELS:
            sc = sidecar(w, m)
            if not sc:
                continue
            ax.plot(sc["acc"], color=MODEL_COLOUR[m], lw=1.0, alpha=0.55 + 0.2 * j,
                    label=f"{m} · {w}  end={sc['acc'][-1]:.2f}×")
    ax.axhspan(0.94, 1.01, color="0.85", zorder=0, label="physical envelope (LIT-VO-003 §4)")
    ax.set_yscale("log")
    ax.set_xlabel("frame")
    ax.set_ylabel("accumulated visual scale (observed, never applied)")
    ax.set_title("H5: scale still drifts — direct Sim(2) is not a scale fix")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    labels, tracks, inliers, colours = [], [], [], []
    for w in windows:
        for m in MODELS:
            key = f"{m}-rigid"
            if key not in comp[w]:
                continue
            labels.append(f"{m[:4]}\n{w.split('-')[-1]}")
            tracks.append(comp[w][key].get("track_median", np.nan))
            inliers.append(comp[w][key].get("inlier_median", np.nan))
            colours.append(MODEL_COLOUR[m])
    xs = np.arange(len(labels))
    ax.bar(xs - 0.19, tracks, 0.38, color=colours, alpha=0.45, label="tracks (median)")
    ax.bar(xs + 0.19, inliers, 0.38, color=colours, alpha=0.95, label="inliers (median)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("count")
    ax.set_title("H6: track and inlier support")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    for w in windows:
        for m in MODELS:
            key = f"{m}-rigid"
            if key not in comp[w]:
                continue
            v = comp[w][key].get("ms_per_frame_median")
            if v is None:
                continue
            ax.bar(f"{m[:4]}·{w.split('-')[-1]}", v, color=MODEL_COLOUR[m], alpha=0.85)
    ax.set_ylabel("median ms/frame (end to end)")
    ax.set_title("Runtime — development laptop only, no target-hardware claim")
    ax.tick_params(axis="x", labelsize=6, rotation=60)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("EXP-VO-009: RPE, scale, support and cost", fontsize=11)
    fig.tight_layout(rect=(0, 0.012, 1, 0.965))
    stamp(fig, "Runtime carries EXP-VO-002's confound: a model that tracks better does more work, "
               "so cost is only interpretable alongside the support panel to its left.")
    fig.savefig(out / "fig3_rpe_scale_support_runtime.png", dpi=140)
    plt.close(fig)

    # ----------------------------------------------------------------- fig4: yaw -> XY coupling
    if "amtown01-c" in comp:
        w = "amtown01-c"
        ds = load_dataset(REPO / WINDOWS[w]["dataset"])
        npz = rotdir / f"rotation_{w}.npz"
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
        if npz.exists():
            d = np.load(npz)
            dgt = d["dgt"]
            for m in MODELS:
                key = f"{m}-rigid"
                if m not in d or key not in comp[w]:
                    continue
                ev = evaluate7(resolve(w, m, "rigid"), ds)
                al, gt, idx = ev["_aligned"], ev["_gt"], ev["_idx"]
                cum = np.degrees(np.cumsum(d[m][1:] - dgt[1:]))
                cum = np.concatenate([[0.0], cum])[idx]
                perr = np.linalg.norm(al - gt, axis=1)
                axes[0].scatter(np.abs(cum), perr, s=2, alpha=0.25, color=MODEL_COLOUR[m],
                                label=f"{MODEL_LABEL[m]}  r={np.corrcoef(np.abs(cum), perr)[0,1]:.3f}")
                dist = np.concatenate([[0.0],
                                       np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
                axes[1].plot(dist, cum, color=MODEL_COLOUR[m], lw=1.1, label=MODEL_LABEL[m])
        axes[0].set_xlabel("|accumulated heading error| (°)")
        axes[0].set_ylabel("aligned position error (m)")
        axes[0].set_title("AMtown01: position error against heading error")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)
        axes[1].axhline(0, color="0.4", lw=0.8)
        axes[1].set_xlabel("ground-truth distance travelled (m)")
        axes[1].set_ylabel("accumulated heading error (°)")
        axes[1].set_title("AMtown01: heading drift against distance")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)
        fig.suptitle("EXP-VO-009 H3: AMtown01's long open legs make it heading-sensitive "
                     "(EXP-VO-008: GT heading removes 83 % of its error)", fontsize=10)
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        stamp(fig, "Association only. A correlation between two quantities that both grow with "
                   "distance travelled is not evidence of causation; EXP-VO-008's GT-heading "
                   "substitution is the controlled test, and it is reported alongside.")
        fig.savefig(out / "fig4_amtown01_yaw_coupling.png", dpi=140)
        plt.close(fig)

    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
