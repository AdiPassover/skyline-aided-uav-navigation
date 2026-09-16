"""`EXP-VO-014`'s figures, from `evaluations/exp-vo-014/`.

Two figures, and the first one is the argument:

    F1  exp-vo-014-paired.png   THE PAIRED RESULT. One panel: metric path-length scale error for
                               each arm, constant-height run beside varying-height run, with the
                               closed-form prediction drawn on the same axes. If the experiment
                               worked, FIXED jumps and BARO does not, and you can see it without
                               reading a number.
    F2  exp-vo-014-detail.png   Six panels: the two ground tracks, the height series each arm
                               actually used against true AGL, and metric error vs travelled
                               distance for both runs.

The predictions are drawn **on the same axes** as the measurements rather than quoted in a caption,
for the reason `EXP-VO-012`'s plot module gives: a closed form that is only ever printed next to a
curve is not really being tested.

    PY=.venv/Scripts/python.exe
    $PY evaluation/tools/exp_vo_014/plot_exp_vo_014.py --results evaluations/exp-vo-014
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# Categorical slots from the validated palette (validate_palette.js: all checks pass, worst
# adjacent CVD dE 23.1 protan / 9.6 tritan, normal-vision 24.0). Roles match EXP-VO-012's figures
# so the two experiments' plots read as one series of results.
ARM_COLOUR = {"fixed": "#eb6834", "baro": "#2a78d6", "oracle": "#1baf7a",
              "oracle_smooth": "#8a8a86"}
ARM_LABEL = {"fixed": "A — FIXED  $h_0$",
             "baro": "B — BARO  $h_0 + h_{baro}(t)$",
             "oracle": "C — ORACLE  true AGL",
             "oracle_smooth": "C′ — ORACLE, median-5"}
GT_COLOUR = "#52514e"
GRID = dict(color="#dedddA", linewidth=0.6, alpha=0.9)
MAIN_ARMS = ("fixed", "baro", "oracle")


def style(ax):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b8b7b3")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.xaxis.label.set_color("#0b0b0b")
    ax.yaxis.label.set_color("#0b0b0b")


def fig_paired(res: dict, out: Path) -> Path:
    """F1 — the money panel. Grouped bars: scale error by arm, const vs varying."""
    p = res["paired"]
    cases = {c["label"]: c for c in res["cases"]}
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    style(ax)

    # Six bars at six positions, grouped in pairs. A two-level x axis (run under bar, arm under
    # group) keeps every label on its own row, so nothing has to be squeezed between the marks.
    pos, meas, pred, colours, runlab = [], [], [], [], []
    for i, a in enumerate(MAIN_ARMS):
        for k, (lab, tag) in enumerate((("A", "constant"), ("B", "varying"))):
            pos.append(i * 2.5 + k * 0.92)
            meas.append(p[a]["scale_error_pct_const" if lab == "A" else "scale_error_pct_vary"])
            pred.append(100 * (cases[lab]["arms"][a]["closed_form_path_weighted_h_ratio"] - 1))
            colours.append(ARM_COLOUR[a])
            runlab.append(tag)
    pos = np.array(pos)

    # The constant-height bar of each pair is drawn lighter, so the run is legible from the mark as
    # well as from the tick label. matplotlib's `alpha=` takes a scalar only, so it goes per-bar.
    bars = ax.bar(pos, meas, 0.82, color=colours, edgecolor="#fcfcfb", linewidth=2)
    for k, b in enumerate(bars):
        b.set_alpha(0.45 if k % 2 == 0 else 1.0)
    for xp, pv in zip(pos, pred):
        ax.plot([xp - 0.44, xp + 0.44], [pv, pv], color="#0b0b0b", linewidth=1.8,
                solid_capstyle="butt", zorder=5)
    ax.axhline(0, color="#0b0b0b", linewidth=1.0)

    # Value labels go on the far side of the prediction rule, so the two never overlap.
    for xp, v, pv in zip(pos, meas, pred):
        top = max(v, pv, 0)
        bot = min(v, pv, 0)
        if v >= 0:
            ax.annotate(f"{v:+.1f}%", (xp, top), ha="center", va="bottom", fontsize=9,
                        color="#0b0b0b", xytext=(0, 4), textcoords="offset points")
        else:
            ax.annotate(f"{v:+.1f}%", (xp, bot), ha="center", va="top", fontsize=9,
                        color="#0b0b0b", xytext=(0, -4), textcoords="offset points")

    lo, hi = min(min(meas), min(pred), 0), max(max(meas), max(pred), 0)
    span = hi - lo
    ax.set_ylim(lo - 0.30 * span, hi + 0.22 * span)

    add_f = p["fixed"]["added_by_height_change_pp"]
    add_b = p["baro"]["added_by_height_change_pp"]
    rel = p["oracle_relative"]
    ax.annotate("", xy=(pos[1] + 0.5, meas[1]), xytext=(pos[1] + 0.5, meas[0]),
                arrowprops=dict(arrowstyle="<->", color=ARM_COLOUR["fixed"], linewidth=1.4))
    ax.annotate(f"the height change alone\nadds {add_f:+.1f} pp",
                xy=(pos[1] + 0.56, (meas[0] + meas[1]) / 2), fontsize=9,
                color=ARM_COLOUR["fixed"], ha="left", va="center")
    ax.annotate(f"adds {add_b:+.2f} pp\n({rel['baro_added_by_height_change_pp']:+.2f} pp "
                f"oracle-relative)",
                xy=(pos[3], max(meas[2], meas[3])), xytext=(0, 22), textcoords="offset points",
                fontsize=9, color=ARM_COLOUR["baro"], ha="center", va="bottom")

    ax.set_xticks(pos)
    ax.set_xticklabels(runlab, fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    for i, a in enumerate(MAIN_ARMS):
        ax.annotate(ARM_LABEL[a], xy=(i * 2.5 + 0.46, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -34), textcoords="offset points", ha="center", fontsize=9.5,
                    color="#0b0b0b")
    ax.plot([], [], color="#0b0b0b", linewidth=1.8,
            label="pre-registered closed-form prediction")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    ax.set_ylabel("metric path-length scale error (% of ground truth)")
    ax.set_title("EXP-VO-014 — the same figure-8 flown at constant and at varying height\n"
                 "zero fitted scale parameters anywhere in this chart",
                 fontsize=10.5, color="#0b0b0b", loc="left")
    fig.tight_layout()
    path = out / "exp-vo-014-paired.png"
    fig.savefig(path, dpi=200, facecolor="#fcfcfb")
    plt.close(fig)
    return path


#: On the constant-height run FIXED and BARO are BIT-IDENTICAL (H1), so one curve would sit exactly
#: on top of the other and look like a missing series. Dashing BARO makes the coincidence visible as
#: a coincidence, which is the point rather than an artefact to hide.
ARM_STYLE = {"fixed": "-", "baro": (0, (5, 4)), "oracle": "-", "oracle_smooth": (0, (1, 2))}


def fig_detail(res: dict, curves: dict, out: Path) -> Path:
    """F2 — tracks, heights, error-vs-distance, both runs."""
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12.5))
    titles = {"A": "Run A — constant height", "B": "Run B — varying height"}
    for j, lab in enumerate(("A", "B")):
        c = next(x for x in res["cases"] if x["label"] == lab)
        cur = curves[lab]
        gt = np.array(cur["_gt_xy"])
        t = np.array(cur["_t_s"])

        ax = axes[0][j]; style(ax)
        ax.plot(gt[:, 0], gt[:, 1], color=GT_COLOUR, linewidth=2.0, label="ground truth (camera)")
        for a in MAIN_ARMS:
            xy = np.array(cur[a]["xy"])
            # reference-initialised placement: nothing fitted, heading0 ~ 0 here
            ax.plot(xy[:, 0] + gt[0, 0], xy[:, 1] + gt[0, 1], color=ARM_COLOUR[a],
                    linewidth=1.5, linestyle=ARM_STYLE[a], label=ARM_LABEL[a])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
        ax.set_title(f"{titles[lab]} — ground track"
                     "\nreference-initialised, nothing fitted", fontsize=9.5, loc="left")
        if j == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="best")

        ax = axes[1][j]; style(ax)
        ax.plot(t, cur["_agl_m"], color=GT_COLOUR, linewidth=1.8, label="true AGL (what geometry wants)")
        for a in MAIN_ARMS:
            ax.plot(t, cur[a]["h_used_m"], color=ARM_COLOUR[a], linewidth=1.5,
                    linestyle=ARM_STYLE[a], label=ARM_LABEL[a], alpha=0.95)
        ax.set_xlabel("time (s)"); ax.set_ylabel("height used (m)")
        sub = ("FIXED and BARO coincide EXACTLY here — baro is identically zero (H1)"
               if lab == "A" else "BARO tracks the climb; its offset from ORACLE is the terrain")
        ax.set_title(f"{titles[lab]} — the height each arm multiplied in\n{sub}",
                     fontsize=9.5, loc="left")
        if j == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="best")

        ax = axes[2][j]; style(ax)
        for a in MAIN_ARMS:
            ax.plot(cur[a]["distance_m"], cur[a]["error_m"], color=ARM_COLOUR[a],
                    linewidth=1.5, linestyle=ARM_STYLE[a], label=ARM_LABEL[a])
        ax.set_xlabel("ground-truth distance travelled (m)")
        ax.set_ylabel("metric horizontal error (m)")
        ax.set_title(f"{titles[lab]} — metric error vs travelled distance"
                     "\nscale fixed at 1, nothing fitted", fontsize=9.5, loc="left")
        if j == 0:
            ax.legend(fontsize=7.5, frameon=False, loc="best")

    lo = min(ax.get_ylim()[0] for ax in axes[2])
    hi = max(ax.get_ylim()[1] for ax in axes[2])
    for ax in axes[2]:
        ax.set_ylim(lo, hi)                          # one scale, so the two runs are comparable
    fig.tight_layout()
    path = out / "exp-vo-014-detail.png"
    fig.savefig(path, dpi=170, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(REPO / "evaluations" / "exp-vo-014"))
    a = p.parse_args(argv)
    root = Path(a.results)
    res = json.loads((root / "results.json").read_text())
    curves = {lab: json.loads((root / f"curves_{lab}.json").read_text()) for lab in ("A", "B")}
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_paired(res, out), fig_detail(res, curves, out)):
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
