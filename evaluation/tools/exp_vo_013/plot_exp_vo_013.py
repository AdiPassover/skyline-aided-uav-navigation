"""`EXP-VO-013` figures. Every one is drawn from a committed artifact; none re-runs the estimator.

    PY=.venv/Scripts/python.exe
    $PY evaluation/tools/exp_vo_013/plot_exp_vo_013.py --eval evaluations/exp-vo-013 \
        [--survey <rtk_profile.json>] [--hat <height_above_takeoff series json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

ARMS = ("fixed", "external", "oracle")
COLOURS = {"fixed": "#c44e52", "external": "#dd8452", "oracle": "#4c72b0", "gt": "#333333"}
LABELS = {"fixed": "A  fixed $h_0$", "external": "B  external relative altitude (RTK, class C)",
          "oracle": "C  oracle AGL (LiDAR, class D)"}


def figdir(out: Path) -> Path:
    d = out / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fig1_heights(res: dict, s: np.lib.npyio.NpzFile, fd: Path) -> None:
    """The three height arrays against each other — the whole experiment in one panel."""
    t = s["frame_t"] - s["frame_t"][0]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    for arm in ARMS:
        ax[0].plot(t, s[f"h_{arm}"], color=COLOURS[arm], lw=1.4, label=LABELS[arm])
    ax[0].set_ylabel("height used, m")
    ax[0].legend(loc="lower left", fontsize=8)
    ax[0].set_title("The three height arms. A and B are indistinguishable because the aircraft's\n"
                    "takeoff-relative altitude is constant; all the variation is terrain.",
                    fontsize=10)
    d = res["terrain_decomposition"]
    h0 = res["height_provenance"]["h0_m"]
    dh_air = s["h_external"] - h0
    dh_terr = h0 + dh_air - s["h_oracle"]
    ax[1].plot(t, dh_air, color="#55a868", lw=1.4,
               label=fr"$\Delta h_{{aircraft}}$  (ptp {d['dh_aircraft_m']['ptp']:.2f} m)")
    ax[1].plot(t, dh_terr, color="#8172b3", lw=1.4,
               label=fr"$\Delta$terrain  (ptp {d['dh_terrain_m']['ptp']:.1f} m)")
    ax[1].axhline(0, color="0.7", lw=0.7)
    ax[1].set_xlabel("time into window, s")
    ax[1].set_ylabel("m")
    ax[1].legend(loc="upper left", fontsize=8)
    ax[1].set_title(r"$h_{AGL} = h_0 + \Delta h_{aircraft} - \Delta$terrain   "
                    f"({100 * d['share_of_agl_variation_from_terrain']:.1f} % of the variation "
                    "is terrain)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fd / "fig1_height_arms_and_decomposition.png", dpi=150)
    plt.close(fig)


def fig2_trajectories(res: dict, s: np.lib.npyio.NpzFile, fd: Path) -> None:
    """Reference-initialised trajectories — ZERO fitted parameters, so the scale error is visible."""
    keep = s["keep"]
    gt = np.stack([s["gt_east"], s["gt_north"]], axis=1)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_vo_012"))
    import metric_metrics as mm

    # The camera's derived yaw mounting angle, not the airframe heading -- see analyse.py.
    heading0 = res["camera_frame_rotation"]["camera_frame_rotation_deg"]
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.plot(gt[:, 0], gt[:, 1], color=COLOURS["gt"], lw=2.0, label="RTK ground truth")
    for arm in ARMS:
        est = np.stack([s[f"{arm}_east_m"], s[f"{arm}_north_m"]], axis=1)[keep]
        a = mm.reference_initialised(est, gt, heading0).apply(est)
        e = res["arms"][arm]["ref_init"]["endpoint_error_pct_of_path"]
        ax.plot(a[:, 0], a[:, 1], color=COLOURS[arm], lw=1.2, alpha=0.9,
                label=f"{LABELS[arm]} — {e:.2f} % of path")
    ax.set_aspect("equal")
    ax.set_xlabel("east, m")
    ax.set_ylabel("north, m")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Reference-initialised metric trajectories — zero fitted parameters.\n"
                 "No scale is fitted, so a wrong height shows up as a wrong-sized trajectory.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "fig2_trajectories_reference_initialised.png", dpi=150)
    plt.close(fig)


def fig3_error_vs_distance(res: dict, s: np.lib.npyio.NpzFile, fd: Path) -> None:
    keep = s["keep"]
    gt = np.stack([s["gt_east"], s["gt_north"]], axis=1)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_vo_012"))
    import metric_metrics as mm

    fig, ax = plt.subplots(figsize=(9, 4.6))
    for arm in ARMS:
        est = np.stack([s[f"{arm}_east_m"], s[f"{arm}_north_m"]], axis=1)[keep]
        dist, err = mm.error_vs_distance(
            est, gt, res["camera_frame_rotation"]["camera_frame_rotation_deg"])
        ax.plot(dist, err, color=COLOURS[arm], lw=1.3, label=LABELS[arm])
    ax.set_xlabel("travelled distance, m")
    ax.set_ylabel("horizontal metric error, m")
    ax.legend(fontsize=8)
    ax.set_title("Metric error against distance travelled, reference-initialised "
                 "(no alignment, no fitted scale)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "fig3_error_vs_distance.png", dpi=150)
    plt.close(fig)


def fig4_scale_metrics(res: dict, fd: Path) -> None:
    """Path-length ratio measured, against the closed form predicted from LiDAR + GT path alone."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    x = np.arange(len(ARMS))
    meas = [res["arms"][a]["path_length_ratio"] for a in ARMS]
    pred = [res["arms"][a]["closed_form"]["path_weighted_mean_h_used_over_h_true"] for a in ARMS]
    ax[0].bar(x - 0.2, meas, 0.4, label=f"measured (step {res['path_step']})", color="#4c72b0")
    ax[0].bar(x + 0.2, pred, 0.4, label="closed form, from LiDAR + GT path only", color="#c44e52")
    ax[0].axhline(1.0, color="0.4", lw=0.9, ls="--")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["A fixed", "B external", "C oracle"], fontsize=8)
    ax[0].set_ylabel("estimated / ground-truth path length")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Path-length ratio: measured vs predicted", fontsize=10)

    b = res["sim2_blindness"]
    ax[1].bar([0, 1], [b["refinit_endpoint_pct_fixed"], b["refinit_endpoint_pct_oracle"]],
              0.35, color="#4c72b0", label="reference-initialised (0 fitted)")
    ax[1].bar([2.2, 3.2], [b["sim2_nate_pct_fixed"], b["sim2_nate_pct_oracle"]],
              0.35, color="#c44e52", label="Sim(2)-aligned nATE (4 fitted, scale included)")
    ax[1].set_xticks([0, 1, 2.2, 3.2])
    ax[1].set_xticklabels(["A fixed", "C oracle", "A fixed", "C oracle"], fontsize=8)
    ax[1].set_ylabel("% of path")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"H5: the metric family separates the arms by "
                    f"{b['refinit_relative_difference_pct']:.0f} % relative;\n"
                    f"Sim(2) alignment by {b['sim2_relative_difference_pct']:.0f} %", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "fig4_scale_metrics_and_sim2_blindness.png", dpi=150)
    plt.close(fig)


def fig5_b1_height_above_takeoff(hat_path: Path, fd: Path) -> None:
    """B1: the DJI FC relative-altitude channel, against RTK, on the same axes.

    This is the figure that makes the negative result unarguable: the channel that a barometer-aided
    readout would consume reads zero while the aircraft is demonstrably 80 m up.
    """
    obj = json.loads(Path(hat_path).read_text())
    h = np.array(obj["height_above_takeoff"], float)
    r = np.array(obj["rtk_alt"], float)
    t0 = h[0, 0]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(r[:, 0] - t0, r[:, 1] - 1074.116, "o-", color="#4c72b0", ms=3, lw=1.2,
            label="/dji_osdk_ros/rtk_position, altitude − ground datum  (class C)")
    ax.plot(h[:, 0] - t0, h[:, 1], ".", color="#c44e52", ms=4,
            label="/dji_osdk_ros/height_above_takeoff  (the FC channel — class E, unusable)")
    ax.set_xlabel("time into recording, s")
    ax.set_ylabel("metres")
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("B1: MARS-LVIG has no usable flight-controller relative-altitude channel.\n"
                 "It is present and populated, and reads ±1 µm while RTK reads 80 m.", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "fig5_b1_height_above_takeoff_is_empty.png", dpi=150)
    plt.close(fig)


def fig6_b2_survey(survey_path: Path, fd: Path) -> None:
    """B2: whole-flight altitude for every candidate, with the path travelled while it changes."""
    obj = json.loads(Path(survey_path).read_text())
    scenes = obj["scenes"]
    fig, ax = plt.subplots(1, len(scenes), figsize=(4.0 * len(scenes), 3.8), sharey=False)
    if len(scenes) == 1:
        ax = [ax]
    for a, (name, e) in zip(ax, scenes.items()):
        s = e["_series"]["rtk"]
        t = np.asarray(s["t"]); up = np.asarray(s["up"]); cum = np.asarray(s["cum"])
        cruise = float(np.median(up[up > 0.8 * up.max()]))
        off = np.abs(up - cruise) > 0.05 * cruise
        path_off = float(np.sum(np.diff(cum)[off[:-1]]))
        a.plot(t - t[0], up, color="#4c72b0", lw=1.3)
        a.fill_between(t - t[0], 0, up, where=off, color="#c44e52", alpha=0.25)
        a.set_title(f"{name}\npath while altitude changes: {path_off:.0f} m of "
                    f"{cum[-1]:.0f} m ({100 * path_off / cum[-1]:.1f} %)", fontsize=9)
        a.set_xlabel("time, s")
    ax[0].set_ylabel("altitude above ground datum, m")
    fig.suptitle("B2: every MARS-LVIG flight climbs to survey altitude essentially in place — "
                 "so no window has aircraft-altitude variation over a usable path", fontsize=10)
    fig.tight_layout()
    fig.savefig(fd / "fig6_b2_altitude_survey.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--survey", default=None)
    ap.add_argument("--hat", default=None)
    a = ap.parse_args()
    out = Path(a.eval)
    fd = figdir(out)
    made = []
    if (out / "results.json").exists():
        res = json.loads((out / "results.json").read_text())
        s = np.load(out / "series.npz")
        fig1_heights(res, s, fd); made.append("fig1")
        fig2_trajectories(res, s, fd); made.append("fig2")
        fig3_error_vs_distance(res, s, fd); made.append("fig3")
        fig4_scale_metrics(res, fd); made.append("fig4")
    if a.hat:
        fig5_b1_height_above_takeoff(Path(a.hat), fd); made.append("fig5")
    if a.survey:
        fig6_b2_survey(Path(a.survey), fd); made.append("fig6")
    print("wrote", ", ".join(made), "->", fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
