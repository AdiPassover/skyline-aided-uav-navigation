"""EXP-VO-011: assemble every phase into one report and score the pre-registered hypotheses against
the materiality criteria fixed before any arm ran.

Reads:
  evaluations/exp-vo-011/scale_budget.json    (Phase 2, from committed run records)
  evaluations/exp-vo-011/covariates.json      (Phase 6, from committed run records + LiDAR)
  evaluations/exp-vo-011/mechanisms.csv       (Phases 3-5, ScaleBiasMonteCarloApp)
  runs/*-reversed-*/                          (Phase 6b, the time-reversal arms)
  runs/synth-dist-*/                          (Phase 3b, the rendered distortion arms)
  evaluations/exp-vo-008/subst_*.json         (Phase 9, known-answer substitution)

Writes `evaluations/exp-vo-011/report.json` and prints the tables the record quotes.

The materiality criteria are **not** re-derived here; they are transcribed from the frozen
pre-registration and applied. M1 dominant: same sign and >= 50 % of the matched real bias. M2
material: >= 20 %. M3 negligible: < 5 %. M4 correction meaningful: removes >= 50 %. M5 covariation:
|partial r| >= 0.20 at p < 0.01 with the same sign on both sequences. M6 interaction: departure from
additivity >= 25 % of the larger single effect. M7 a correction worth implementing: >= 10 % ATE on
both sequences, or >= 50 % of the bias on both.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import anisotropy_axis as ax
import flow_direction as fd
import scale_series as ss

REPO = Path(__file__).resolve().parents[3]

M1_DOMINANT = 0.50
M2_MATERIAL = 0.20
M3_NEGLIGIBLE = 0.05
M4_CORRECTION = 0.50
M5_R = 0.20
M5_P = 0.01
M6_INTERACTION = 0.25

# Which real arm each synthetic condition is compared against. The realistic composites are matched
# to their own flight; everything else is scored against the median real arm of the same model,
# because the mechanism sweeps are not flight-specific.
REVERSAL_ARMS = [("hkairport01-a", "affine"), ("hkairport01-a", "homography"),
                 ("amtown01-c", "affine")]
RENDER_ARMS = [(arm, model) for arm in ("none", "distorted", "rectified")
               for model in ("affine", "homography")]


def load_mechanisms(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                if k in ("condition", "model", "relief", "spread"):
                    out[k] = v
                elif k == "correct_distortion":
                    out[k] = v.strip().lower() == "true"
                else:
                    out[k] = float(v)
            rows.append(out)
    return rows


def pick(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def one(rows, **kw):
    got = pick(rows, **kw)
    if len(got) != 1:
        raise KeyError(f"expected exactly one row for {kw}, got {len(got)}")
    return got[0]


def resolvable(row: dict) -> bool:
    """Is this arm's mean distinguishable from zero at all?

    Every Monte Carlo cell has a standard error, and several mechanisms in this grid sit below it.
    Quoting such a cell as "the bias distortion produces" would be reading noise, so every table
    carries the SE and every verdict is gated on |bias| > 3 SE.
    """
    se = row.get("se_log_scale", float("nan"))
    return se == se and se > 0 and abs(row["bias_log_scale"]) > 3.0 * se


def classify(bias: float, target: float) -> str:
    if target == 0:
        return "undefined"
    frac = bias / target
    if frac >= M1_DOMINANT:
        return "M1 dominant"
    if frac >= M2_MATERIAL:
        return "M2 material"
    if frac < M3_NEGLIGIBLE:
        return "M3 negligible" if frac >= 0 else "M3 negligible (opposite sign)"
    return "between M2 and M3"


def real_targets(budget: dict) -> dict:
    """The per-frame bias each synthetic arm has to be compared against."""
    per_arm = {f"{a['window']}-{a['model']}": a["inc_log_scale"]["mean"]
               for a in budget["arms"] + budget["controls"]}
    by_model = {}
    for model in ("affine", "homography", "similarity"):
        vals = [a["inc_log_scale"]["mean"] for a in budget["arms"] + budget["controls"]
                if a["model"] == model]
        if vals:
            by_model[model] = float(np.median(vals))
    all_vals = [a["inc_log_scale"]["mean"] for a in budget["arms"]]
    return {"per_arm": per_arm, "by_model": by_model,
            "overall_median": float(np.median(all_vals)),
            "overall_min": float(np.min(all_vals)), "overall_max": float(np.max(all_vals))}


def rendered_arms() -> list[dict]:
    """Phase 3b: the rendered distortion arms, read the same way as the real ones."""
    out = []
    for arm, model in RENDER_ARMS:
        d = REPO / "runs" / f"synth-dist-{arm}-{model}-v1" / "logical_transform.csv"
        if not d.exists():
            continue
        vals, events = [], []
        with d.open(newline="") as fh:
            for row in csv.DictReader(fh):
                # A run still being written leaves a partial final row; skip it rather than
                # crashing, so the analysis can be re-run while an arm is in flight.
                if row.get("inc_log_scale") in (None, ""):
                    continue
                events.append(row["event"])
                vals.append(float(row["inc_log_scale"]))
        keep = [v for v, e in zip(vals, events) if e not in ss.DROPPED_EVENTS]
        st = ss.describe(np.array(keep))
        st.update({"arm": arm, "model": model,
                   "restarts": events.count("restart"), "recenters": events.count("recenter"),
                   "accum_scale": math.exp(st["sum"])})
        out.append(st)
    return out


def epoch_structure() -> dict:
    """Is the drift concentrated at keyframe transitions, or spread over ordinary frames?

    `LIT-VO-003` section 7 predicts that a per-transition bias makes `lambda` grow linearly in the
    KEYFRAME count rather than the frame count, which would put the mechanism at the keyframe change
    rather than in the ordinary per-frame fit. Keyframe changes are not reported by BoofCV, so this
    uses `EXP-VO-001`'s verified proxy -- a track-count rise of at least 20 %, plus every recenter,
    both of which are *sufficient* conditions for a keyframe change and therefore a lower bound on
    their number.
    """
    out = {}
    for w, m in ss.REAL_ARMS + ss.CONTROL_ARMS:
        arm = ss.load_arm(w, m)
        y, tc = arm["inc_log_scale"], arm["track_count"]
        kf = np.diff(tc, prepend=tc[0]) >= 0.20 * np.maximum(tc, 1)
        kf |= (arm["event"] == "recenter")
        total = float(y.sum())
        out[f"{w}-{m}"] = {
            "n_keyframe_proxy": int(kf.sum()), "keyframe_frame_fraction": float(kf.mean()),
            "mean_on_keyframe_frames": float(y[kf].mean()) if kf.any() else float("nan"),
            "mean_on_ordinary_frames": float(y[~kf].mean()),
            "drift_fraction_from_keyframe_frames": float(y[kf].sum()) / total if total else float("nan"),
            "median_epoch_frames": float(np.median(np.diff(np.where(kf)[0]))) if kf.sum() > 2
            else float("nan"),
        }
    return out


def substitution_value() -> dict:
    """Phase 9: what perfect physical scale is worth, from `EXP-VO-008`'s committed artifacts.

    Reported two ways, because they answer different questions. Against the raw baseline it measures
    what scale buys **today**, with the heading error still present and dominating. Conditional on
    correct heading it measures what scale buys once the dominant error is removed -- which is the
    number that matters for a system that intends to fix heading with an external sensor.
    """
    out = {}
    for f in sorted((REPO / "evaluations" / "exp-vo-008").glob("subst_*.json")):
        o = json.loads(f.read_text())
        v = o["variants"]
        base = v["baseline"]["ate_rmse_normalised"]
        gth = v["gt_heading"]["ate_rmse_normalised"]
        out[o["label"]] = {
            "baseline": base,
            "lidar_scale_applied": v["lidar_scale_applied"]["ate_rmse_normalised"],
            "gt_heading": gth,
            "gt_heading_lidar_scale": v["gt_heading_lidar_scale"]["ate_rmse_normalised"],
            "visual_scale_applied": v["visual_scale_applied"]["ate_rmse_normalised"],
            "scale_gain_vs_baseline": (v["lidar_scale_applied"]["ate_rmse_normalised"] - base) / base,
            "scale_gain_given_correct_heading":
                (v["gt_heading_lidar_scale"]["ate_rmse_normalised"] - gth) / gth,
            "heading_gain_vs_baseline": (gth - base) / base,
            "visual_scale_penalty": (v["visual_scale_applied"]["ate_rmse_normalised"] - base) / base,
            "scale_agreement": o["scale_agreement"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)

    budget = json.loads((out / "scale_budget.json").read_text())
    cov = json.loads((out / "covariates.json").read_text())
    mech = load_mechanisms(out / "mechanisms.csv")
    targets = real_targets(budget)

    report = {"record": "EXP-VO-011", "targets": targets}

    # ---------------------------------------------------------------- Phase 6b: time reversal
    rev = [fd.reversal_pair(w, m) for w, m in REVERSAL_ARMS]
    report["reversal"] = rev
    print("\n=== Phase 6b: time-reversed replay of the same real frames ===")
    print(f"{'arm':<28} {'forward':>12} {'reversed':>12} {'flip':>5} {'odd %':>8} {'even':>12} "
          f"{'travel R':>9}")
    for r in rev:
        print(f"{r['window'] + '-' + r['model']:<28} {r['forward']['mean']:>12.4e} "
              f"{r['reversed']['mean']:>12.4e} {str(r['sign_flipped']):>5} "
              f"{100 * r['odd_fraction']:>8.1f} {r['even_component']:>12.4e} "
              f"{r['forward_travel']['concentration_flow_weighted']['resultant_R']:>9.3f}")

    # ---------------------------------------------------------------- Phase 3: distortion
    print("\n=== Phase 3: lens distortion, at the realistic keyframe baseline ===")
    print(f"{'model':<12} {'k1':>8} {'corner px':>10} {'0 deg':>12} {'90':>12} {'180':>12} "
          f"{'270':>12} {'mean|4 dirs|':>13} {'% of target':>12} {'max SE':>10} {'resolved':>9}")
    dist = {}
    for model in ("HOMOGRAPHY", "AFFINE", "SIMILARITY"):
        tgt = targets["by_model"][model.lower()]
        for k1 in sorted({r["k1"] for r in pick(mech, condition="dist", model=model)}):
            rows = {r["travel_angle_deg"]: r for r in pick(mech, condition="dist", model=model, k1=k1)}
            b = {a: rows[a]["bias_log_scale"] for a in sorted(rows)}
            mean_abs = float(np.mean([abs(v) for v in b.values()]))
            corner = rows[0.0]["corner_distortion_px"]
            max_se = max(r["se_log_scale"] for r in rows.values())
            res = sum(1 for r in rows.values() if resolvable(r))
            dist[(model, k1)] = {"by_angle": b, "mean_abs": mean_abs, "corner_px": corner,
                                 "frac_of_target": mean_abs / abs(tgt),
                                 "max_se": max_se, "resolved_of_4": res}
            print(f"{model:<12} {k1:>8.3f} {corner:>10.1f} " +
                  " ".join(f"{b[a]:>12.3e}" for a in sorted(b)) +
                  f" {mean_abs:>13.3e} {100 * mean_abs / abs(tgt):>11.1f}% {max_se:>10.2e} "
                  f"{str(res) + '/4':>9}")
    report["distortion"] = {f"{m}|{k}": v for (m, k), v in dist.items()}

    # ---------------------------------------------------------------- Phase 3: correction (H2)
    print("\n=== Phase 3 / H2: correct distortion handling ===")
    corr = {}
    for model in ("HOMOGRAPHY", "AFFINE", "SIMILARITY"):
        for k1 in sorted({r["k1"] for r in pick(mech, condition="dist-corrected", model=model)}):
            if k1 == 0.0:
                continue
            un = one(mech, condition="dist", model=model, k1=k1, travel_angle_deg=0.0)
            co = one(mech, condition="dist-corrected", model=model, k1=k1)
            removed = 1.0 - abs(co["bias_log_scale"]) / abs(un["bias_log_scale"]) \
                if un["bias_log_scale"] else float("nan")
            corr[f"{model}|{k1}"] = {"uncorrected": un["bias_log_scale"],
                                     "corrected": co["bias_log_scale"], "removed_fraction": removed}
            print(f"{model:<12} k1={k1:>7.3f}  uncorrected {un['bias_log_scale']:>11.3e}  "
                  f"corrected {co['bias_log_scale']:>11.3e}  removed {100 * removed:>6.2f}%")
    report["distortion_correction"] = corr

    # ---------------------------------------------------------------- Phase 4: relief
    print("\n=== Phase 4: terrain relief (bias at 0 deg travel; sd is the variance half) ===")
    # For a plane tilted by s = tan(alpha) relative to nadir, pure horizontal translation induces
    # an image transform whose linear block is diag(1 - a s, 1) with a = t/(d sqrt(1+s^2)) -- the
    # stretch is along the slope direction ONLY. So sqrt(|det|) reads HALF the true relative-height
    # rate, and with the opposite sign. Derived in the record; predicted here so the arm is scored
    # against a closed form rather than eyeballed.
    footprint_m = 1224.0 * 80.0 / 735.53
    def slope_prediction(spread_m: float) -> float:
        slope = spread_m / (0.9 * footprint_m)
        return 0.5 * slope * 0.30 / 80.0

    print(f"{'model':<12} {'relief':<14} {'spread m':>9} {'actual':>8} {'bias':>12} {'sd':>10} "
          f"{'% of target':>12} {'SE':>10} {'predicted':>11} {'meas/pred':>10}")
    relief = {}
    for model in ("HOMOGRAPHY", "AFFINE", "SIMILARITY"):
        tgt = targets["by_model"][model.lower()]
        for kind in ("SLOPE", "RANDOM_FIELD", "STRUCTURES"):
            for r in sorted(pick(mech, condition="relief", model=model, relief=kind,
                                 travel_angle_deg=0.0), key=lambda x: x["relief_spread_m"]):
                pred = slope_prediction(r["relief_spread_m"]) if kind == "SLOPE" else float("nan")
                relief[f"{model}|{kind}|{r['relief_spread_m']}"] = {
                    "bias": r["bias_log_scale"], "sd": r["sd_log_scale"],
                    "se": r["se_log_scale"], "resolvable": resolvable(r),
                    "actual_spread_m": r["actual_spread_m"],
                    "mean_inliers": r["mean_inliers"],
                    "slope_prediction": pred,
                    "measured_over_predicted": (r["bias_log_scale"] / pred) if pred else float("nan"),
                    "frac_of_target": r["bias_log_scale"] / tgt}
                ratio = (r["bias_log_scale"] / pred) if pred == pred and pred else float("nan")
                print(f"{model:<12} {kind:<14} {r['relief_spread_m']:>9.1f} "
                      f"{r['actual_spread_m']:>8.1f} {r['bias_log_scale']:>12.3e} "
                      f"{r['sd_log_scale']:>10.3e} {100 * r['bias_log_scale'] / tgt:>11.1f}% "
                      f"{r['se_log_scale']:>10.2e} {pred:>11.3e} {ratio:>10.3f}")
    report["relief"] = relief

    # ---------------------------------------------------------------- Phase 5: interaction
    print("\n=== Phase 5: distortion x relief, at realistic levels only ===")
    inter = {}
    for model in ("HOMOGRAPHY", "AFFINE", "SIMILARITY"):
        for spread in (11.3, 22.0):
            both = one(mech, condition="both", model=model, relief_spread_m=spread)
            d = one(mech, condition="dist", model=model, k1=-0.02, travel_angle_deg=0.0)
            rl = one(mech, condition="relief", model=model, relief="RANDOM_FIELD",
                     relief_spread_m=spread, travel_angle_deg=0.0)
            additive = d["bias_log_scale"] + rl["bias_log_scale"]
            dep = both["bias_log_scale"] - additive
            larger = max(abs(d["bias_log_scale"]), abs(rl["bias_log_scale"]))
            inter[f"{model}|{spread}"] = {
                "singles_resolvable": bool(resolvable(d) and resolvable(rl)),
                "both_resolvable": bool(resolvable(both)),
                "distortion": d["bias_log_scale"], "relief": rl["bias_log_scale"],
                "both": both["bias_log_scale"], "additive_prediction": additive,
                "departure": dep,
                "departure_over_larger": abs(dep) / larger if larger else float("inf"),
                "m6_material": bool(larger and abs(dep) / larger >= M6_INTERACTION)}
            print(f"{model:<12} spread {spread:>5.1f}  dist {d['bias_log_scale']:>11.3e}  "
                  f"relief {rl['bias_log_scale']:>11.3e}  both {both['bias_log_scale']:>11.3e}  "
                  f"additive {additive:>11.3e}  departure {100 * abs(dep) / larger:>7.1f}% of larger"
                  f"  singles resolved: {resolvable(d)}/{resolvable(rl)}")
    report["interaction"] = inter

    # ---------------------------------------------------------------- H6b / H6c / tilt / baseline
    print("\n=== H6b errors-in-variables, H6c principal-point offset, tilt, keyframe baseline ===")
    extras = {}
    for model in ("HOMOGRAPHY", "AFFINE", "SIMILARITY"):
        tgt = targets["by_model"][model.lower()]
        for cond_name in ("eiv", "eiv-clustered"):
            for r in sorted(pick(mech, condition=cond_name, model=model),
                            key=lambda x: x["sigma_px"]):
                extras[f"{cond_name}|{model}|{r['sigma_px']}"] = {
                    "bias": r["bias_log_scale"], "sd": r["sd_log_scale"],
                    "frac_of_target": r["bias_log_scale"] / tgt}
        for cond_name in ("tilt", "tilt-changing", "ppoffset-tilt"):
            for r in sorted(pick(mech, condition=cond_name, model=model),
                            key=lambda x: x["tilt_deg"]):
                extras[f"{cond_name}|{model}|{r['tilt_deg']}"] = {
                    "bias": r["bias_log_scale"], "frac_of_target": r["bias_log_scale"] / tgt}
        for r in pick(mech, condition="ppoffset-dist", model=model):
            extras[f"ppoffset-dist|{model}"] = {"bias": r["bias_log_scale"],
                                                "frac_of_target": r["bias_log_scale"] / tgt}
        for cond_name in ("baseline-dist", "baseline-relief", "baseline-eiv", "baseline-ideal"):
            for r in sorted(pick(mech, condition=cond_name, model=model),
                            key=lambda x: x["baseline_frames"]):
                extras[f"{cond_name}|{model}|{int(r['baseline_frames'])}"] = {
                    "bias": r["bias_log_scale"], "frac_of_target": r["bias_log_scale"] / tgt}
        for cond_name in ("realistic-hkairport", "realistic-amtown"):
            r = one(mech, condition=cond_name, model=model)
            key = "hkairport01-b" if "hk" in cond_name else "amtown01-c"
            t = targets["per_arm"][f"{key}-{model.lower()}"]
            extras[f"{cond_name}|{model}"] = {"bias": r["bias_log_scale"], "target": t,
                                              "frac_of_target": r["bias_log_scale"] / t,
                                              "verdict": classify(r["bias_log_scale"], t)}
            print(f"{cond_name:<22} {model:<12} bias {r['bias_log_scale']:>11.3e}  "
                  f"target {t:>10.3e}  {100 * r['bias_log_scale'] / t:>7.1f}%  "
                  f"{classify(r['bias_log_scale'], t)}")
    report["extras"] = extras

    print("\n  keyframe-baseline dependence (bias, distortion k1=-0.02, distributed features):")
    for model in ("HOMOGRAPHY", "AFFINE", "SIMILARITY"):
        row = [extras[f"baseline-dist|{model}|{b}"]["bias"] for b in (0, 1, 5, 10, 20, 40, 80)]
        print(f"    {model:<12} " + " ".join(f"{v:>11.3e}" for v in row))
    print("  (baselines 0, 1, 5, 10, 20, 40, 80 frames since the keyframe)")

    print("\n  errors-in-variables sweep (bias, distributed features):")
    for model in ("HOMOGRAPHY", "AFFINE", "SIMILARITY"):
        sig = sorted({r["sigma_px"] for r in pick(mech, condition="eiv", model=model)})
        row = [extras[f"eiv|{model}|{s}"]["bias"] for s in sig]
        print(f"    {model:<12} " + " ".join(f"{v:>11.3e}" for v in row))
        print(f"    {'sigma px':<12} " + " ".join(f"{s:>11.1f}" for s in sig))
        break

    # ---------------------------------------------------------------- Phase 3b: rendered
    ren = rendered_arms()
    report["rendered"] = ren
    if ren:
        print("\n=== Phase 3b: rendered distortion arms, through the real tracker ===")
        print(f"{'arm':<14} {'model':<12} {'n':>6} {'bias':>12} {'sd':>10} {'accum scale':>12} "
              f"{'restarts':>9}")
        for r in ren:
            print(f"{r['arm']:<14} {r['model']:<12} {r['n']:>6} {r['mean']:>12.3e} "
                  f"{r['sd']:>10.3e} {r['accum_scale']:>12.4f} {r['restarts']:>9}")

    # ------------------------------------------------- the slope mechanism's signature, on real data
    sig = {f"{w}-{m}": ax.signature(ss.load_arm(w, m)) for w, m in ss.REAL_ARMS + ss.CONTROL_ARMS}
    report["anisotropy_signature"] = sig
    print("\n=== Does the real data carry the slope mechanism's signature? ===")
    print(f"{'arm':<28} {'observed':>11} {'from aniso':>12} {'ratio':>8} {'corr':>7} "
          f"{'axis R (2x)':>12} {'perp frac':>10}")
    for k, v in sig.items():
        print(f"{k:<28} {v['mean_observed']:>11.3e} "
              f"{v['mean_predicted_from_anisotropy']:>12.3e} "
              f"{v['ratio_observed_over_predicted']:>8.3f} {v['correlation']:>7.3f} "
              f"{v['axis_vs_travel_doubled']['resultant_R']:>12.3f} "
              f"{v['fraction_axis_within_30deg_of_perpendicular']:>10.3f}")
    print("  (a uniform axis distribution gives R = 0 and perp frac = 1/3; the slope mechanism "
          "requires the axis to align with travel)")

    # ---------------------------------------------------------------- keyframe structure
    ep = epoch_structure()
    report["epoch_structure"] = ep
    print("\n=== Is the drift concentrated at keyframe transitions? (EXP-VO-001's proxy) ===")
    print(f"{'arm':<28} {'n_kf':>6} {'kf frac':>8} {'mean on kf':>12} {'mean other':>12} "
          f"{'drift from kf':>14} {'median epoch':>13}")
    for k, v in ep.items():
        print(f"{k:<28} {v['n_keyframe_proxy']:>6} {v['keyframe_frame_fraction']:>8.3f} "
              f"{v['mean_on_keyframe_frames']:>12.3e} {v['mean_on_ordinary_frames']:>12.3e} "
              f"{100 * v['drift_fraction_from_keyframe_frames']:>13.1f}% "
              f"{v['median_epoch_frames']:>13.1f}")

    # ---------------------------------------------------------------- Phase 9
    sub = substitution_value()
    report["substitution"] = sub
    print("\n=== Phase 9: what a perfect physical scale is worth (EXP-VO-008 artifacts) ===")
    for k, v in sub.items():
        print(f"{k:<24} baseline {100 * v['baseline']:.3f}%  "
              f"scale alone {100 * v['scale_gain_vs_baseline']:+.1f}%  "
              f"heading alone {100 * v['heading_gain_vs_baseline']:+.1f}%  "
              f"scale GIVEN heading {100 * v['scale_gain_given_correct_heading']:+.1f}%  "
              f"visual scale {100 * v['visual_scale_penalty']:+.1f}%")

    # ---------------------------------------------------------------- H5 from covariates
    report["covariates"] = {a["window"] + "-" + a["model"]: {
        k: a["per_frame"][k]["partial"] for k in a["per_frame"] if a["per_frame"][k]["partial"]}
        for a in cov["arms"]}
    print("\n=== H5a/H5b: does the real increment covary with the physical quantity? (M5: |r|>=0.20, p<0.01) ===")
    for name in ("lidar_spread_m", "lidar_height_m", "tilt_deg"):
        rs = [(a["window"] + "-" + a["model"], a["per_frame"][name]["partial"]) for a in cov["arms"]]
        worst = max(abs(p["r"]) for _, p in rs)
        passes = [n for n, p in rs if abs(p["r"]) >= M5_R and p["p"] < M5_P]
        print(f"  {name:<18} max |partial r| = {worst:.3f} over {len(rs)} arms; arms meeting M5: "
              f"{passes if passes else 'none'}")

    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
