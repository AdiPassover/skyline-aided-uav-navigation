"""EXP-VO-011 figures. Every one regenerates from committed artifacts alone -- no estimator run and
no imagery -- so it works in a fresh worktree.

    python evaluation/tools/exp_vo_011/plot_exp_vo_011.py --out evaluations/exp-vo-011/figures
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import analyse as an  # noqa: E402
import scale_series as ss  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "evaluations" / "exp-vo-011"

MODEL_COLOUR = {"affine": "#1f77b4", "homography": "#d62728", "similarity": "#2ca02c",
                "AFFINE": "#1f77b4", "HOMOGRAPHY": "#d62728", "SIMILARITY": "#2ca02c"}


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=140)
    plt.close(fig)
    print(f"  {name}")


def fig1_real_bias(budget: dict, out: Path) -> None:
    """Required figure 1: real per-frame log-scale bias by dataset and model."""
    arms = budget["arms"] + budget["controls"]
    labels = [f"{a['window']}\n{a['model']}" for a in arms]
    means = [a["inc_log_scale"]["mean"] for a in arms]
    errs = [a["inc_log_scale"]["sd"] / math.sqrt(a["inc_log_scale"]["n_eff"]) for a in arms]
    cols = [MODEL_COLOUR[a["model"]] for a in arms]

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(range(len(arms)), means, yerr=errs, color=cols, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean per-frame log-scale increment")
    ax.set_title("EXP-VO-011 fig 1 - the quantity to explain: a positive per-frame log-scale bias\n"
                 "on every window and every model class (error bars: AR(1)-corrected standard error)")
    for i, a in enumerate(arms):
        ax.text(i, means[i], f"  {a['phantom_climb_ms']:.2f} m/s", ha="center",
                va="bottom" if means[i] > 0 else "top", fontsize=7, rotation=90)
    ax.margins(y=0.25)
    _save(fig, out, "fig1_real_per_frame_bias.png")


def fig2_accumulated(budget: dict, out: Path) -> None:
    """Required figure 2: accumulated visual scale against the physical envelope."""
    arms = budget["arms"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(arms))
    vis = [a["accum_scale_observed"] for a in arms]
    lo = [a["physical"]["lidar"]["span_scale_min"] for a in arms]
    hi = [a["physical"]["lidar"]["span_scale_max"] for a in arms]
    end = [a["physical"]["lidar"]["accum_scale"] for a in arms]

    ax.bar(x, [h - l for h, l in zip(hi, lo)], bottom=lo, width=0.6, color="#cccccc",
           label="physically plausible band (LiDAR height above the imaged surface)")
    ax.plot(x, end, "k_", ms=22, label="physical end-of-run scale")
    ax.plot(x, vis, "o", color="#d62728", ms=9, label="accumulated visual scale")
    for i, a in enumerate(arms):
        ax.annotate(f"x{a['observed_over_physical_scale']:.1f}", (i, vis[i]),
                    textcoords="offset points", xytext=(9, 0), fontsize=8, color="#d62728")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a['window']}\n{a['model']}" for a in arms], fontsize=8)
    ax.set_ylabel("scale relative to the first frame (log axis)")
    ax.set_title("EXP-VO-011 fig 2 - accumulated visual scale against the physical envelope.\n"
                 "Exponentiation is what makes a 3e-4 per-frame bias look enormous; fig 1 is the "
                 "honest target.")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, out, "fig2_accumulated_vs_physical.png")


def fig3_distortion_sweep(mech: list, targets: dict, out: Path) -> None:
    """Required figure 3: distortion strength against scale bias, random and deterministic."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, cond, title in (
            (axes[0], "dist", "random feature placement\n(Monte Carlo error swamps the effect)"),
            (axes[1], "dist-lattice-quadrant",
             "deterministic one-quadrant lattice\n(exact: the largest asymmetry worth considering)")):
        for model in ("AFFINE", "HOMOGRAPHY", "SIMILARITY"):
            rows = an.pick(mech, condition=cond, model=model, travel_angle_deg=0.0)
            if not rows:
                continue
            rows.sort(key=lambda r: r["corner_distortion_px"] * (1 if r["k1"] <= 0 else -1))
            xs = [r["corner_distortion_px"] * (1 if r["k1"] <= 0 else -1) for r in rows]
            ys = [r["bias_log_scale"] for r in rows]
            es = [3 * r["se_log_scale"] for r in rows]
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.2, capsize=2,
                        color=MODEL_COLOUR[model], label=model.lower())
        tgt = targets["overall_median"]
        ax.axhspan(-abs(tgt), abs(tgt), color="#ffe9a8", zorder=0,
                   label="magnitude of the real bias" if ax is axes[0] else None)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvspan(9.4, 47.0, color="#e8f0ff", zorder=-1,
                   label="1-5 % corner distortion (realistic band)" if ax is axes[0] else None)
        ax.set_xlabel("corner radial displacement (px); negative k1 to the right of zero")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("per-frame log-scale bias")
    fig.suptitle("EXP-VO-011 fig 3 - lens distortion against the real bias. Error bars are 3 SE.",
                 fontsize=11)
    _save(fig, out, "fig3_distortion_strength_vs_bias.png")


def fig4_correction(report: dict, mech: list, out: Path) -> None:
    """Required figure 4: distortion corrected against uncorrected."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    models = ("AFFINE", "HOMOGRAPHY", "SIMILARITY")
    width = 0.25
    for i, model in enumerate(models):
        rows = [r for r in an.pick(mech, condition="dist-lattice-quadrant", model=model,
                                   travel_angle_deg=0.0) if r["k1"] != 0]
        rows.sort(key=lambda r: r["k1"])
        corr = {r["k1"]: r for r in an.pick(mech, condition="dist-corrected", model=model)}
        xs = np.arange(len(rows)) + (i - 1) * width
        ax.bar(xs, [abs(r["bias_log_scale"]) for r in rows], width, color=MODEL_COLOUR[model],
               label=f"{model.lower()} uncorrected")
        ax.bar(xs, [abs(corr[r["k1"]]["bias_log_scale"]) if r["k1"] in corr else 0 for r in rows],
               width, color="k", alpha=0.85,
               label="corrected (all models)" if i == 0 else None)
        if i == 1:
            ax.set_xticks(np.arange(len(rows)))
            ax.set_xticklabels([f"{r['k1']:g}\n{r['corner_distortion_px']:.0f} px" for r in rows],
                               fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("|per-frame log-scale bias|")
    ax.set_xlabel("k1 / corner radial displacement")
    ax.set_title("EXP-VO-011 fig 4 - correct rectification removes the distortion effect exactly.\n"
                 "The corrected bars are at machine precision; the question is whether what it "
                 "removes matters.")
    ax.legend(fontsize=7)
    _save(fig, out, "fig4_distortion_corrected.png")


def fig5_relief(mech: list, targets: dict, out: Path) -> None:
    """Required figure 5: terrain depth spread against scale bias."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for model in ("AFFINE", "HOMOGRAPHY", "SIMILARITY"):
        for kind, ls in (("SLOPE", "-"), ("RANDOM_FIELD", "--"), ("STRUCTURES", ":")):
            rows = sorted(an.pick(mech, condition="relief", model=model, relief=kind,
                                  travel_angle_deg=0.0), key=lambda r: r["relief_spread_m"])
            if not rows:
                continue
            xs = [r["relief_spread_m"] for r in rows]
            axes[0].plot(xs, [r["bias_log_scale"] for r in rows], ls, marker="o", ms=3,
                         color=MODEL_COLOUR[model],
                         label=f"{model.lower()} {kind.lower()}" if model == "AFFINE" or kind == "SLOPE" else None)
            axes[1].plot(xs, [r["sd_log_scale"] for r in rows], ls, marker="o", ms=3,
                         color=MODEL_COLOUR[model])
    tgt = abs(targets["overall_median"])
    for ax in axes:
        for v in (11.3, 22.0):
            ax.axvline(v, color="#888", lw=0.8, ls="-.")
        ax.set_xlabel("in-footprint depth spread, p95-p5 (m)")
    axes[0].axhspan(-tgt, tgt, color="#ffe9a8", zorder=0)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("per-frame log-scale BIAS")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("per-frame log-scale SD (the variance half)")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].text(11.3, axes[0].get_ylim()[1], " AMtown01", fontsize=7, va="top")
    axes[1].text(22.0, axes[1].get_ylim()[1], " HKairport01", fontsize=7, va="top")
    fig.suptitle("EXP-VO-011 fig 5 - relief. A coherent SLOPE produces a large deterministic bias; "
                 "statistically homogeneous relief produces variance and no bias.", fontsize=10)
    _save(fig, out, "fig5_relief_depth_vs_bias.png")


def fig6_factorial(report: dict, out: Path) -> None:
    """Required figure 6: none / distortion / relief / both at realistic levels."""
    inter = report["interaction"]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    keys = [k for k in inter if k.endswith("|22.0")]
    labels, groups = [], []
    for k in keys:
        v = inter[k]
        labels.append(k.split("|")[0].lower())
        groups.append([0.0, v["distortion"], v["relief"], v["both"], v["additive_prediction"]])
    x = np.arange(5)
    for i, (lab, g) in enumerate(zip(labels, groups)):
        ax.bar(x + (i - 1) * 0.25, g, 0.25, color=MODEL_COLOUR[lab.upper()], label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["neither", "distortion\nonly", "relief\nonly", "both", "additive\nprediction"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("per-frame log-scale bias")
    ax.set_title("EXP-VO-011 fig 6 - the pre-declared 2x2 at realistic levels (k1 = -0.02, "
                 "22 m depth spread).\nNeither single effect is resolvable above its own Monte "
                 "Carlo error, so the interaction test is inconclusive by construction.")
    ax.legend(fontsize=8)
    _save(fig, out, "fig6_factorial.png")


def fig7_lidar_covariate(cov: dict, out: Path) -> None:
    """Required figure 7: real scale increment against the LiDAR depth-spread covariate."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (w, m) in zip(axes, [("hkairport01-b", "affine"), ("amtown01-c", "affine"),
                                 ("amtown01-c", "homography")]):
        arm = ss.load_arm(w, m)
        c = ss.covariates_for(arm)
        y, x = arm["inc_log_scale"], c["lidar_spread_m"]
        n = (len(y) // 100) * 100
        yb = y[:n].reshape(-1, 100).mean(axis=1)
        xb = x[:n].reshape(-1, 100).mean(axis=1)
        ax.scatter(x, y, s=1, alpha=0.08, color="#999")
        ax.scatter(xb, yb, s=26, color=MODEL_COLOUR[m], edgecolor="k", lw=0.4,
                   label="100-frame window means")
        rec = next(a for a in cov["arms"] if a["window"] == w and a["model"] == m)
        p = rec["per_frame"]["lidar_spread_m"]["partial"]
        ax.set_title(f"{w} {m}\npartial r = {p['r']:+.3f} (p = {p['p']:.3f}), M5 needs |r| >= 0.20",
                     fontsize=9)
        ax.set_xlabel("LiDAR in-footprint depth spread (m)")
        ax.axhline(0, color="k", lw=0.7)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("per-frame log-scale increment")
    axes[0].set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.5))
    fig.suptitle("EXP-VO-011 fig 7 - the real increment does not covary with LiDAR depth spread, "
                 "after controlling for elapsed time and travelled distance.", fontsize=10)
    _save(fig, out, "fig7_lidar_depth_covariate.png")


def fig8_reversal(report: dict, out: Path) -> None:
    """Replaces the feature-radius covariate figure, which the run record cannot supply.

    `frames.csv` and `logical_transform.csv` carry no per-frame feature-radius distribution, so the
    distortion-sensitivity covariate the brief asked for is not measurable without a new probe. The
    time-reversal arm answers the same question -- is the bias odd in the direction of travel? --
    directly, on the same imagery, and without any modelling.
    """
    rev = report["reversal"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(rev))
    ax.bar(x - 0.2, [r["forward"]["mean"] for r in rev], 0.4, color="#1f77b4", label="forward")
    ax.bar(x + 0.2, [r["reversed"]["mean"] for r in rev], 0.4, color="#ff7f0e",
           label="time-reversed replay")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['window']}\n{r['model']}" for r in rev], fontsize=8)
    ax.set_ylabel("mean per-frame log-scale increment")
    ax.set_title("the bias reverses with the arrow of time", fontsize=9)
    ax.legend(fontsize=8)
    for i, r in enumerate(rev):
        ax.text(i, 0, f"  {100 * r['odd_fraction']:.0f} % odd", ha="center", va="bottom", fontsize=7)

    for i, r in enumerate(rev):
        ax2.bar(i - 0.2, r["odd_component"], 0.4, color="#2ca02c",
                label="odd in travel direction" if i == 0 else None)
        ax2.bar(i + 0.2, r["even_component"], 0.4, color="#9467bd",
                label="even (arrow of time)" if i == 0 else None)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r['window']}\n{r['model']}" for r in rev], fontsize=8)
    ax2.set_title("decomposed: the odd half carries it", fontsize=9)
    ax2.legend(fontsize=8)
    fig.suptitle("EXP-VO-011 fig 8 - replaying the SAME real frames backwards. A geometric "
                 "mechanism cannot prefer a direction of time; a tracking asymmetry cannot reverse.",
                 fontsize=10)
    _save(fig, out, "fig8_time_reversal.png")


def fig9_model_response(budget: dict, mech: list, out: Path) -> None:
    """Required figure 9: affine against homography (and similarity) scale response."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for w in ("hkairport01-a", "hkairport01-b", "amtown01-c"):
        arms = [a for a in budget["arms"] + budget["controls"] if a["window"] == w]
        order = {"similarity": 0, "affine": 1, "homography": 2}
        arms.sort(key=lambda a: order[a["model"]])
        ax.plot([order[a["model"]] for a in arms],
                [a["inc_log_scale"]["mean"] for a in arms], "o-", label=w)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["similarity\n4 DoF", "affine\n6 DoF", "homography\n8 DoF"])
    ax.set_ylabel("mean per-frame log-scale increment")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("real flights: the bias is nearly model-class-independent", fontsize=9)
    ax.legend(fontsize=8)

    for model in ("SIMILARITY", "AFFINE", "HOMOGRAPHY"):
        rows = sorted(an.pick(mech, condition="relief", model=model, relief="SLOPE",
                              travel_angle_deg=0.0), key=lambda r: r["relief_spread_m"])
        ax2.plot([r["relief_spread_m"] for r in rows], [r["bias_log_scale"] for r in rows],
                 "o-", color=MODEL_COLOUR[model], label=model.lower())
    ax2.set_xlabel("in-footprint depth spread of a coherent slope (m)")
    ax2.set_ylabel("per-frame log-scale bias")
    ax2.set_title("synthetic slope: the three models respond alike", fontsize=9)
    ax2.legend(fontsize=8)
    fig.suptitle("EXP-VO-011 fig 9 - model dependence. The similarity model cannot represent "
                 "anisotropy at all, and drifts the same.", fontsize=10)
    _save(fig, out, "fig9_model_response.png")


def fig10_budget(report: dict, budget: dict, out: Path) -> None:
    """Required figure 10: the explanatory-magnitude summary."""
    tgt = abs(report["targets"]["overall_median"])
    rows = []
    mech = an.load_mechanisms(EVAL / "mechanisms.csv")

    def add(label, value):
        rows.append((label, abs(value) / tgt))

    d = an.one(mech, condition="dist-lattice-quadrant", model="AFFINE", k1=-0.02,
               travel_angle_deg=0.0)
    add("lens distortion, realistic k1,\none-quadrant support (upper bound)", d["bias_log_scale"])
    add("lens distortion, symmetric support",
        an.one(mech, condition="dist-lattice", model="AFFINE", k1=-0.02,
               travel_angle_deg=0.0)["bias_log_scale"])
    add("coherent slope at AMtown01's 11.3 m",
        an.one(mech, condition="relief", model="AFFINE", relief="SLOPE", relief_spread_m=11.3,
               travel_angle_deg=0.0)["bias_log_scale"])
    add("coherent slope at HKairport01's 22.0 m",
        an.one(mech, condition="relief", model="AFFINE", relief="SLOPE", relief_spread_m=22.0,
               travel_angle_deg=0.0)["bias_log_scale"])
    add("homogeneous relief, 22.0 m",
        an.one(mech, condition="relief", model="AFFINE", relief="RANDOM_FIELD",
               relief_spread_m=22.0, travel_angle_deg=0.0)["bias_log_scale"])
    add("elevated structures, 22.0 m",
        an.one(mech, condition="relief", model="AFFINE", relief="STRUCTURES",
               relief_spread_m=22.0, travel_angle_deg=0.0)["bias_log_scale"])
    add("errors-in-variables at 0.5 px",
        an.one(mech, condition="eiv", model="AFFINE", sigma_px=0.5)["bias_log_scale"])
    add("tilt held at 6.5 deg",
        an.one(mech, condition="tilt", model="AFFINE", tilt_deg=6.5)["bias_log_scale"])
    add("principal-point offset at 6.5 deg tilt",
        an.one(mech, condition="ppoffset-tilt", model="AFFINE", tilt_deg=6.5)["bias_log_scale"])

    rev = report["reversal"]
    add("measured: the odd-in-travel component",
        float(np.mean([r["odd_component"] for r in rev])))
    add("measured: the even component",
        float(np.mean([r["even_component"] for r in rev])))

    fig, ax = plt.subplots(figsize=(10.5, 6))
    y = np.arange(len(rows))
    cols = ["#2ca02c" if v >= 0.5 else "#ff7f0e" if v >= 0.2 else "#c62828" for _, v in rows]
    ax.barh(y, [v for _, v in rows], color=cols)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=8)
    ax.invert_yaxis()
    for thresh, lab in ((0.5, "M1 dominant"), (0.2, "M2 material"), (0.05, "M3 negligible")):
        ax.axvline(thresh, color="k", ls="--", lw=0.8)
        ax.text(thresh, -0.7, lab, fontsize=7, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("|bias| as a fraction of the real per-frame bias "
                  f"({tgt:.2e} per frame, median of six arms)")
    ax.set_title("EXP-VO-011 fig 10 - the scale-bias budget. Materiality bands are the ones fixed\n"
                 "in the pre-registration, before any arm ran.")
    _save(fig, out, "fig10_scale_bias_budget.png")


def fig11_rendered(report: dict, out: Path) -> None:
    """Phase 3b: the rendered arms through the real tracker."""
    ren = report.get("rendered", [])
    if not ren:
        return
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = {"none": 0, "distorted": 1, "rectified": 2}
    for model in ("affine", "homography"):
        rows = sorted([r for r in ren if r["model"] == model], key=lambda r: order[r["arm"]])
        ax.errorbar([order[r["arm"]] for r in rows], [r["mean"] for r in rows],
                    yerr=[r["sd"] / math.sqrt(r["n_eff"]) for r in rows],
                    marker="o", capsize=3, color=MODEL_COLOUR[model], label=model)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["ideal pinhole", "distorted,\nuncorrected", "distorted,\nrectified"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean per-frame log-scale increment")
    ax.set_title("EXP-VO-011 fig 11 - Phase 3b: rendered distortion through the REAL tracker,\n"
                 "600 frames, k1 = -0.02 (18.8 px of corner barrel).")
    ax.legend(fontsize=8)
    _save(fig, out, "fig11_rendered_distortion.png")


def fig12_value(report: dict, out: Path) -> None:
    """Phase 9: what scale is worth now, and what it is worth once heading is correct."""
    sub = report["substitution"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    labels = list(sub)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [-100 * sub[k]["scale_gain_vs_baseline"] for k in labels], 0.4,
           color="#9ecae1", label="perfect scale, against today's baseline")
    ax.bar(x + 0.2, [-100 * sub[k]["scale_gain_given_correct_heading"] for k in labels], 0.4,
           color="#08519c", label="perfect scale, GIVEN correct heading")
    for i, k in enumerate(labels):
        ax.text(i - 0.2, -100 * sub[k]["scale_gain_vs_baseline"],
                f"{-100 * sub[k]['scale_gain_vs_baseline']:.1f}%", ha="center", va="bottom",
                fontsize=8)
        ax.text(i + 0.2, -100 * sub[k]["scale_gain_given_correct_heading"],
                f"{-100 * sub[k]['scale_gain_given_correct_heading']:.1f}%", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("normalised-ATE reduction (%)")
    ax.set_title("EXP-VO-011 fig 12 - Phase 9: solving scale is worth four to six times more once\n"
                 "the heading error EXP-VO-010 closed on is removed. Known-answer substitution, "
                 "not implementable.")
    ax.legend(fontsize=8)
    _save(fig, out, "fig12_value_of_scale.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)

    budget = json.loads((EVAL / "scale_budget.json").read_text())
    cov = json.loads((EVAL / "covariates.json").read_text())
    report = json.loads((EVAL / "report.json").read_text())
    mech = an.load_mechanisms(EVAL / "mechanisms.csv")

    print("writing figures:")
    fig1_real_bias(budget, out)
    fig2_accumulated(budget, out)
    fig3_distortion_sweep(mech, report["targets"], out)
    fig4_correction(report, mech, out)
    fig5_relief(mech, report["targets"], out)
    fig6_factorial(report, out)
    fig7_lidar_covariate(cov, out)
    fig8_reversal(report, out)
    fig9_model_response(budget, mech, out)
    fig10_budget(report, budget, out)
    fig11_rendered(report, out)
    fig12_value(report, out)


if __name__ == "__main__":
    main()
