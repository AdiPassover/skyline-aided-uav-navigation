"""Thesis Figure 4.1 -- estimated trajectories on both real-flight windows.

Two panels, HK-B (development) and AM-C (independent validation), each showing RTK ground
truth against the two navigation-state representations under the projective model, after
one global Sim(2) alignment. The alignment reproduces `naveval.evaluate`'s pipeline exactly
(same synchronize -> fit_sim2 path), and the per-arm normalised ATE printed on stdout is
checked against evaluations/vo-closure/canonical_results.json by hand.

Reads only committed artifacts: runs/<id>/frames.csv and datasets/<id>/groundtruth.csv.

    cd evaluation && python tools/thesis_figures/trajectory_comparison.py
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from naveval.alignment import fit_sim2                      # noqa: E402
from naveval.dataset import load_dataset                    # noqa: E402
from naveval.metrics import absolute_trajectory_error       # noqa: E402
from naveval.runrecord import load_run_record               # noqa: E402
from naveval.sync import synchronize                        # noqa: E402

OUT = os.path.join(ROOT, "figures", "vo", "trajectory_comparison.png")

WINDOWS = [
    ("hkairport01-b", r"(a) HK-B $-$ development", [
        ("full",   "hkairport01-b-homography-logical-v1"),
        ("rigid",  "hkairport01-b-homography-rigid-v1"),
    ]),
    ("amtown01-c", r"(b) AM-C $-$ independent validation", [
        ("full",   "amtown01-c-homography-logical-v1"),
        ("rigid",  "amtown01-c-homography-rigid-v1"),
    ]),
]

STYLE = {
    "gt":     ("#000000", 1.6, 1.00, "ground truth (RTK)"),
    "full":   ("#c81e1e", 1.1, 1.00, "full composition"),
    "rigid":  ("#1f6fb4", 0.6, 0.90, "rigid integration"),
}


def arm(dataset, run_id):
    """Sim(2)-aligned estimate, ground truth, and normalised ATE for one run."""
    rr = load_run_record(os.path.join(ROOT, "runs", run_id))
    sr = synchronize(
        rr.timestamps_s, dataset.gt_timestamps, dataset.gt_east, dataset.gt_north,
        dataset.gt_heading if dataset.has_heading else None, dataset.gt_valid,
        clock_offset_s=dataset.clock_offset_s,
        clock_drift_s_per_s=dataset.clock_drift_s_per_s,
    )
    est = np.stack([rr.est_x[sr.frame_indices], rr.est_y[sr.frame_indices]], axis=1)
    gt = np.stack([sr.east, sr.north], axis=1)
    alignment = fit_sim2(est, gt)
    aligned = alignment.apply(est)
    ate = absolute_trajectory_error(aligned, gt)["rmse"]
    path_len = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    return aligned, gt, ate / path_len


def main():
    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8.0,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.35))
    handles = None

    for ax, (dataset_id, title, arms) in zip(axes, WINDOWS):
        dataset = load_dataset(os.path.join(ROOT, "datasets", dataset_id))
        results = {name: arm(dataset, run_id) for name, run_id in arms}

        colour, lw, alpha, label = STYLE["gt"]
        gt = results["rigid"][1]
        h_gt, = ax.plot(gt[:, 0], gt[:, 1], color=colour, lw=lw, zorder=5, label=label)
        handles = [h_gt]
        for name, z in (("full", 4), ("rigid", 3)):
            colour, lw, alpha, label = STYLE[name]
            h, = ax.plot(results[name][0][:, 0], results[name][0][:, 1],
                         color=colour, lw=lw, alpha=alpha, zorder=z, label=label)
            handles.append(h)
            print("%-14s %-7s nATE=%6.2f %%" % (dataset_id, name, 100 * results[name][2]))

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title, pad=4)
        ax.set_xlabel("east (m)", labelpad=1)
        ax.tick_params(length=2.5, pad=1.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.text(0.985, 0.02,
                "\n".join("%s %.2f%%" % (n, 100 * results[n][2])
                          for n in ("full", "rigid")),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2,
                linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.5))

    axes[0].set_ylabel("north (m)", labelpad=1)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.4,
               fontsize=7.8, borderaxespad=0.0)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.subplots_adjust(wspace=0.22)
    fig.savefig(OUT, dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
