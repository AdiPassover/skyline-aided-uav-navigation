"""EXP-INT-004 figures: per-recording error-versus-time overlays of the four arms with accepted
re-anchors marked, and the accuracy-versus-false-correction trade-off scatter.

    $PY evaluation/tools/claim_closure/int_figures.py --out evaluations/claim-closure-2026-09/int

Axes are shared per panel and start at zero; nothing is rescaled to flatter an arm.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ARMS = ["vo_only", "NAIVE_1V", "NAIVE_2V", "FROZEN"]
STYLE = {"vo_only": ("k", "-", 1.6), "NAIVE_1V": ("tab:red", "-", 1.0), "NAIVE_2V": ("tab:orange", "--", 1.0), "FROZEN": ("tab:blue", "-", 1.4)}
PANELS = ["ho1-mtn-fig8", "ho1-mtn-fig8-scaled", "ho1-mtn-trinity", "ho1-vil-fig8",
          "interesting-r1-vary", "interesting-r3-const", "easier-sq-const", "fig8-flat-const", "mtn-r2-vary-synthetic-loss"]


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "evaluations/claim-closure-2026-09/int")
    args = ap.parse_args()
    out = args.out.resolve()
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(3, 3, figsize=(17, 11))
    for ax, label in zip(axes.ravel(), PANELS):
        ev = out / "eval" / label
        rows = read_csv(ev / "error_curve.csv")
        t = np.asarray([float(r["time_s"]) for r in rows])
        for arm in ARMS:
            e = np.asarray([float(r[f"err_{arm}_m"]) if r[f"err_{arm}_m"] != "" else np.nan for r in rows])
            c, ls, lw = STYLE[arm]
            ax.plot(t, e, color=c, ls=ls, lw=lw, label=arm, alpha=0.9)
            if arm != "vo_only":
                re = read_csv(ev / f"reanchors_{arm}.csv")
                for r in re:
                    if "frame" not in r or not r.get("frame"):
                        continue
                    fi = int(r["frame"])
                    idx = np.searchsorted([int(x["frame"]) for x in rows], fi)
                    if idx < len(rows):
                        wp = r.get("wrong_place", "False") == "True"
                        ax.plot(t[idx], np.nan_to_num(e[idx], nan=0.0), marker="x" if wp else "o", color=c, ms=9 if wp else 6,
                                mfc="none", mew=1.5)
        tr = res["tracks"][label]
        ax.set_title(f"{label} [{tr['role']}]", fontsize=10)
        ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("EXP-INT-004: persistent-position error vs time — VO_ONLY, naive single-view top-match, naive two-view top-match, frozen gate "
                 "(○ accepted re-anchor, × wrong-place accept; gaps = position UNKNOWN)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig_int_arms_error_vs_time.png", dpi=130); plt.close(fig)

    # trade-off scatter
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arm in ARMS[1:]:
        c, _, _ = STYLE[arm]
        xs, ys, labels = [], [], []
        for label, tr in res["tracks"].items():
            a = tr["arms"][arm]
            xs.append(a["ate_change_pct"]); ys.append(a["reanchors"]["wrong_place"] + a["reanchors"]["severely_harmful"]); labels.append(label)
        ax.scatter(xs, ys, color=c, label=arm, s=50, alpha=0.8, edgecolor="k")
        for x, y, l in zip(xs, ys, labels):
            if y > 0 or x < -10:
                ax.annotate(l, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=50)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ATE change vs VO_ONLY (%, negative = better; symlog)")
    ax.set_ylabel("wrong-place + severely harmful accepts (count)")
    ax.set_title("EXP-INT-004: accuracy versus false-correction trade-off per recording and arm")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_int_tradeoff.png", dpi=150); plt.close(fig)
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
