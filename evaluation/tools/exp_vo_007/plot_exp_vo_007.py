"""EXP-VO-007 figures: the independent-sequence validation of `RIGID_MOTION` on `AMtown01`.

Eight panels across three figures, matching the record's *Outputs* list:

    fig1  ground truth vs legacy vs rigid trajectory; homography vs affine
    fig2  error over distance; yaw over time; accumulated visual scale (observed, never applied)
    fig3  per-frame anisotropy and projective magnitude against this sequence's own tilt ceiling;
          track support; runtime distributions

Every figure is stamped with the run ids it was built from, so a figure in the thesis can be traced
back to the exact run records without consulting this file.

Usage::

    python evaluation/tools/exp_vo_007/plot_exp_vo_007.py --out evaluations/exp-vo-007/figures
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse import DATASET, MODELS, READOUTS, evaluate, run_dir        # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402

COLOURS = {"legacy": "tab:orange", "logical": "tab:red", "rigid": "tab:blue"}
LABELS = {"legacy": "MOSAIC_LEGACY", "logical": "LOGICAL_FRAME", "rigid": "RIGID_MOTION"}


def sidecar(model: str) -> dict:
    p = REPO / run_dir(model, "rigid") / "logical_transform.csv"
    if not p.exists():
        return {}
    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in rows])                # noqa: E731
    return {"acc": col("rigid_accum_scale"), "aniso": col("inc_anisotropy")[1:],
            "persp": col("inc_perspective")[1:]}


def tilt_ceiling(root: Path) -> float | None:
    """`1/cos(theta_max)` from this sequence's OWN attitude data (`LIT-VO-003` section 4).

    `EXP-VO-006` R6 used 1.04, from `HKairport01`'s 15.54 deg maximum tilt. Reusing that number
    here would import the development sequence's geometry into the validation, so it is recomputed.
    """
    p = root / "attitude.csv"
    if not p.exists():
        return None
    with p.open(newline="") as f:
        tilt = np.array([float(r["tilt_deg"]) for r in csv.DictReader(f)])
    return float(1.0 / np.cos(np.radians(np.percentile(tilt, 99.9))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=DATASET)
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    root = REPO / a.dataset
    ds = load_dataset(root)

    ev, used = {}, []
    for model in MODELS:
        for readout in READOUTS:
            d = REPO / run_dir(model, readout)
            if (d / "frames.csv").exists():
                ev[(model, readout)] = evaluate(d, ds)
                used.append(d.name)
    if not ev:
        raise SystemExit("No run records found")
    stamp = "runs: " + ", ".join(sorted(used))

    # --- fig1: trajectories -------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    for i, model in enumerate(MODELS):
        base = next((ev[(model, r)] for r in READOUTS if (model, r) in ev), None)
        if base is None:
            continue
        ax[i].plot(base["_gt"][:, 0], base["_gt"][:, 1], "k-", lw=1.4, label="ground truth")
        for readout in READOUTS:
            if (model, readout) in ev:
                e = ev[(model, readout)]
                ax[i].plot(e["_aligned"][:, 0], e["_aligned"][:, 1], lw=0.9,
                           color=COLOURS[readout],
                           label=f"{LABELS[readout]} ({100 * e['ate_rmse_normalised']:.2f} %)")
        ax[i].set_aspect("equal")
        ax[i].set_title(f"{model}")
        ax[i].set_xlabel("east (m)"); ax[i].set_ylabel("north (m)")
        ax[i].legend(fontsize=8)
    fig.suptitle(f"EXP-VO-007 AMtown01: trajectory after one global Sim(2) alignment\n{stamp}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_trajectories.png", dpi=140)
    plt.close(fig)

    # --- fig2: error over distance, yaw, accumulated scale -------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for model in MODELS:
        ls = "-" if model == "homography" else "--"
        for readout in READOUTS:
            if (model, readout) not in ev:
                continue
            e = ev[(model, readout)]
            gt = e["_gt"]
            dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
            err = np.linalg.norm(e["_aligned"] - gt, axis=1)
            ax[0].plot(dist, err, ls, lw=0.7, color=COLOURS[readout],
                       label=f"{model[:4]} {LABELS[readout]}")
    ax[0].set_xlabel("distance travelled (m)"); ax[0].set_ylabel("position error (m)")
    ax[0].set_title("error over distance"); ax[0].legend(fontsize=7)

    if ds.has_heading:
        for model in MODELS:
            ls = "-" if model == "homography" else "--"
            for readout in READOUTS:
                if (model, readout) not in ev:
                    continue
                e = ev[(model, readout)]
                y = e.get("yaw_rmse_deg")
                if y is not None:
                    ax[1].bar(f"{model[:4]}\n{readout}", y, color=COLOURS[readout])
        ax[1].set_ylabel("yaw RMSE (deg)"); ax[1].set_title("heading error")
    else:
        ax[1].text(0.5, 0.5, "no heading ground truth", ha="center")

    for model in MODELS:
        sc = sidecar(model)
        if sc:
            ax[2].plot(sc["acc"], lw=0.8, label=f"{model}: end {sc['acc'][-1]:.3f}")
    ax[2].axhspan(0.90, 1.10, color="grey", alpha=0.2, label="[0.90, 1.10]")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("frame"); ax[2].set_ylabel("accumulated visual scale")
    ax[2].set_title("scale is observed and never applied to XY"); ax[2].legend(fontsize=8)
    fig.suptitle(f"EXP-VO-007 AMtown01\n{stamp}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig2_error_yaw_scale.png", dpi=140)
    plt.close(fig)

    # --- fig3: per-frame diagnostics, track support, runtime -----------------------------------
    ceiling = tilt_ceiling(root)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for model in MODELS:
        sc = sidecar(model)
        if not sc:
            continue
        x = np.sort(sc["aniso"])
        ax[0].plot(x, np.linspace(0, 1, x.size), lw=1.1, label=model)
    if ceiling:
        ax[0].axvline(ceiling, color="k", ls="--", lw=1.0,
                      label=f"tilt ceiling {ceiling:.3f} (this flight)")
    ax[0].set_xlim(1.0, 1.08)
    ax[0].set_xlabel("per-frame anisotropy s1/s2"); ax[0].set_ylabel("cumulative fraction")
    ax[0].set_title("per-frame deformation vs what tilt explains"); ax[0].legend(fontsize=8)

    for model in MODELS:
        for readout in READOUTS:
            if (model, readout) not in ev:
                continue
            e = ev[(model, readout)]
            if e["track_count_median"] is not None:
                ax[1].bar(f"{model[:4]}\n{readout}", e["track_count_median"],
                          color=COLOURS[readout])
    ax[1].set_ylabel("median track count"); ax[1].set_title("track support")

    for model in MODELS:
        for readout in READOUTS:
            if (model, readout) not in ev:
                continue
            rt = ev[(model, readout)]["runtime"]
            med = rt.get("median_ms") or rt.get("median")
            if med is not None:
                ax[2].bar(f"{model[:4]}\n{readout}", med, color=COLOURS[readout])
    ax[2].set_ylabel("median per-frame runtime (ms)")
    ax[2].set_title("runtime (development laptop, not target hardware)")
    fig.suptitle(f"EXP-VO-007 AMtown01: diagnostics observed but never integrated\n{stamp}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig3_diagnostics.png", dpi=140)
    plt.close(fig)

    (out / "figure_provenance.json").write_text(json.dumps(
        {"runs": sorted(used), "dataset": a.dataset,
         "tilt_ceiling_from_this_sequence": ceiling}, indent=2))
    print(f"wrote 3 figures to {out}")


if __name__ == "__main__":
    main()
