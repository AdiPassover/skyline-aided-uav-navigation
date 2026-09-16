"""The two canonical figures the closure audit found missing — `F17` and `F8`.

`VO_CANONICAL_RESULTS.md` §9 indexes sixteen canonical figures that already exist and names exactly
two that do not:

**F17 — the residual-accumulation mechanism.** The whole `RIGID_MOTION` architecture rests on
*"small per-frame residuals, individually inside the physical envelope, integrated thousands of
times"*. `F5` shows the per-frame distribution and Table 4 shows the accumulated result; nothing
showed the **link**. This module draws both halves on one figure so the composition is visible.

**F8 — raw yaw versus offset-aligned heading.** Every yaw RMSE this lane published before
`EXP-VO-009` R2 is 99.23–99.92 % a *constant frame offset* between the estimator's first-frame yaw
datum and the Sim(2) rotation fitted to positions. The tracking component is 3.1–9.4° rms. Table 2
carries the numbers; this draws them.

**Committed artifacts only.** Run records under `runs/` (`logical_transform.csv`, `frames.csv`),
dataset descriptors under `datasets/`, and `evaluations/exp-vo-010/comparison.json`. **No estimator
is run and no imagery is read**, so this regenerates in a fresh worktree in well under a minute.
Nothing here is fitted and no historical metric is rewritten.

    PY=.venv/Scripts/python.exe
    $PY evaluation/tools/vo_closure/closure_figures.py \
        --out figures/vo-closure

It also writes `closure_figures.json` beside the images: every quantity either figure's caption
quotes, so each has an artifact behind it.

**Definitions, fixed here so the figures cannot drift from the records.**

*Per-frame* anisotropy and scale are the sidecar's own `inc_anisotropy` / `inc_log_scale` columns —
`σ₁/σ₂` and `log √|det J|` of the Jacobian of `D_k⁻¹` at the image centre, which is exactly what
`RigidMotionDecomposition` computes in Java (`COMP-001` §3.7). Rows whose `event` is `init` or
`restart` are dropped, which is `EXP-VO-011`'s **pre-declared** row filter
(`scale_series.DROPPED_EVENTS`): neither row carries a valid inter-frame increment. Applying it is
what makes this module's per-frame means agree with `evaluations/exp-vo-011/scale_budget.json` to
every digit, which `_cross_check_per_frame` then asserts rather than assumes.

*Accumulated* scale is the sidecar's `rigid_accum_scale` — `exp(Σ inc_log_scale)`, the pipeline's own
cumulative product, matching `VO_CANONICAL_RESULTS.md` Table 4.

*Accumulated* anisotropy is `σ₁/σ₂` of the Jacobian of **`G_k⁻¹`** at the image centre, `G_k : L → C_k`
being the composed logical transform recorded in `g00..g22`. The inverse direction and the centre
evaluation point both matter: this definition reproduces `EXP-VO-004` R4's published medians
(1.39 / 1.35 / 3.39 / 2.19) and its "above 1.10 on 89–98 % of frames" to the printed digit, and the
forward direction does not. It is *not* the product of the per-frame anisotropies — rotations
interleave between the stretches, so the composed value has to be read off the composed transform.

The physical envelope drawn on both accumulated panels is the same one the 2026-08-26 report uses:
0.87×–1.30× for scale, from the LiDAR-measured height range on both flights; and each flight's own
`1/cos θ_max` tilt ceiling for anisotropy, read from its `attitude.csv`.
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
sys.path.insert(0, str(EVALUATION_DIR / "tools" / "exp_vo_008"))
sys.path.insert(0, str(EVALUATION_DIR / "tools" / "exp_vo_010"))

import recompose as rc                                                   # noqa: E402
from heading_metrics import (circular_mean_deg, circular_rms_about_deg,  # noqa: E402
                             wrap180)
from naveval.alignment import fit_sim2                                   # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.runrecord import load_run_record                           # noqa: E402
from naveval.sync import synchronize                                     # noqa: E402

# Working geometry: MARS-LVIG 2448x2048 at downsampleFactor 2 (`VO_CLOSURE.md` §3.2).
WIDTH, HEIGHT = 1224, 1024
CX, CY = WIDTH / 2.0, HEIGHT / 2.0

# The physically explicable accumulated-scale band, both flights (2026-08-26 report, LiDAR height
# range). Drawn, not fitted.
SCALE_BAND = (0.87, 1.30)

# The six reference-configuration arms. `hkairport01-a` is a strict 180 s PREFIX of `-b` — one
# flight, not two — and is labelled as such everywhere it appears.
ARMS = [
    ("hkairport01-a", "homography", "runs/hkairport01-a-homography-rigid-v1", "datasets/hkairport01-a"),
    ("hkairport01-a", "affine",     "runs/hkairport01-a-affine-rigid-v1",     "datasets/hkairport01-a"),
    ("hkairport01-b", "homography", "runs/hkairport01-b-homography-rigid-v1", "datasets/hkairport01-b"),
    ("hkairport01-b", "affine",     "runs/hkairport01-b-affine-rigid-v1",     "datasets/hkairport01-b"),
    ("amtown01-c",    "homography", "runs/amtown01-c-homography-rigid-v1",    "datasets/amtown01-c"),
    ("amtown01-c",    "affine",     "runs/amtown01-c-affine-rigid-v1",        "datasets/amtown01-c"),
]

# F17's per-frame panels use the four arms of `VO_CANONICAL_RESULTS.md` F5 — the primary development
# window and the independent validation window. Adding the `-a` prefix there would draw the same
# flight twice.
PER_FRAME_ARMS = [a for a in ARMS if a[0] != "hkairport01-a"]

COLOUR = {("hkairport01-a", "homography"): "#6b8fb5",
          ("hkairport01-a", "affine"):     "#8fbfc9",
          ("hkairport01-b", "homography"): "tab:blue",
          ("hkairport01-b", "affine"):     "tab:cyan",
          ("amtown01-c", "homography"):    "tab:red",
          ("amtown01-c", "affine"):        "tab:orange"}
STYLE = {"homography": "-", "affine": "--"}


def label(window: str, model: str) -> str:
    suffix = "  (prefix of -b)" if window == "hkairport01-a" else ""
    return f"{window} · {model}{suffix}"


# --------------------------------------------------------------------------- artifact readers

#: `EXP-VO-011`'s pre-declared row filter. `init` is the identity seed row; `restart` is the frame
#: at which the estimator discarded its accumulation, so its "increment" is not an inter-frame
#: transform at all. Kept identical to `scale_series.DROPPED_EVENTS`.
DROPPED_EVENTS = ("init", "restart")


def sidecar_columns(run_rel: str) -> dict:
    """Per-frame and accumulated series straight out of the committed sidecar.

    The per-frame series have `DROPPED_EVENTS` removed; the accumulated series does not, because
    it is a running state that every row carries a valid value of.
    """
    path = REPO / run_rel / "logical_transform.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    events = np.array([r["event"] for r in rows])
    keep = ~np.isin(events, DROPPED_EVENTS)
    col = lambda k: np.array([float(r[k]) for r in rows])                # noqa: E731
    return {"inc_anisotropy": col("inc_anisotropy")[keep],
            "inc_log_scale": col("inc_log_scale")[keep],
            "inc_perspective": col("inc_perspective")[keep],
            "accum_scale": col("rigid_accum_scale"),
            "n_dropped": int((~keep).sum()),
            "n_restart": int((events == "restart").sum()),
            "events": events}


def accumulated_anisotropy(run_rel: str) -> np.ndarray:
    """`σ₁/σ₂` of `J(G_k⁻¹)` at the image centre — see the module docstring for why this direction."""
    rec = rc.load_recording(REPO / run_rel)
    g_inv = np.linalg.inv(rec.G)
    jac = np.array([rc._jacobian(g_inv[k], CX, CY) for k in range(len(rec))])
    sv = np.linalg.svd(jac, compute_uv=False)
    return sv[:, 0] / sv[:, 1]


def tilt_ceiling(dataset_rel: str) -> float | None:
    """`1/cos θ_max` from the flight's own attitude record — the anisotropy a rigid camera tilt can
    explain. Returns None when the dataset carries no attitude column."""
    path = REPO / dataset_rel / "attitude.csv"
    if not path.exists():
        return None
    with path.open(newline="") as f:
        tilt = np.array([float(r["tilt_deg"]) for r in csv.DictReader(f)])
    return float(1.0 / np.cos(np.radians(tilt.max())))


def mad_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _cross_check_per_frame(per_frame: dict) -> list[dict]:
    """Assert this module's per-frame statistics against `EXP-VO-011`'s committed budget.

    A figure that quietly disagreed with the record it illustrates would be worse than no figure,
    so the agreement is checked rather than trusted. Returns the check log; raises on any failure.
    """
    path = REPO / "evaluations/exp-vo-011/scale_budget.json"
    if not path.exists():
        return [{"note": f"{path} absent — per-frame cross-check skipped", "ok": True}]
    committed = {(a["window"], a["model"]): a for a in json.loads(path.read_text())["arms"]}
    checks = []
    for key, pf in per_frame.items():
        ref = committed.get(key)
        if ref is None:
            continue
        v = pf["inc_log_scale"]
        for name, mine, theirs, tol in (
                ("n", float(v.size), float(ref["inc_log_scale"]["n"]), 0.0),
                ("mean", float(v.mean()), ref["inc_log_scale"]["mean"], 1e-9),
                ("mad_sigma", mad_sigma(v), ref["inc_log_scale"]["mad_sigma"], 1e-9)):
            ok = abs(mine - theirs) <= tol + 1e-12 * max(1.0, abs(theirs))
            checks.append({"arm": f"{key[0]}/{key[1]}", "key": f"inc_log_scale.{name}",
                           "mine": mine, "committed": theirs, "ok": bool(ok)})
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise SystemExit("F17 cross-check against evaluations/exp-vo-011/scale_budget.json "
                         "failed:\n" + json.dumps(failed, indent=2))
    return checks


# --------------------------------------------------------------------------- F17

def figure_f17(out: Path, derived: dict) -> None:
    """Per-frame residual → accumulated deformation, in one figure.

    Top row: what a single frame looks like. Bottom row: what the same quantities do after being
    composed a few thousand times. The point of putting them together is that the bottom row is
    *arithmetic on* the top row, not a separate phenomenon.
    """
    ceilings = {ds: tilt_ceiling(ds) for _, _, _, ds in ARMS}
    per_frame, accum = {}, {}
    for window, model, run_rel, ds_rel in ARMS:
        per_frame[(window, model)] = sidecar_columns(run_rel)
        accum[(window, model)] = accumulated_anisotropy(run_rel)
    checks = _cross_check_per_frame(per_frame)

    stats = {}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.8))

    # ---- (a) per-frame anisotropy, against each flight's own tilt-explained ceiling ------------
    ax = axes[0][0]
    for window, model, _, ds_rel in PER_FRAME_ARMS:
        a = np.sort(per_frame[(window, model)]["inc_anisotropy"])
        ax.plot(a, np.linspace(0, 1, a.size), STYLE[model], lw=1.3,
                color=COLOUR[(window, model)], label=label(window, model))
    for ds_rel, ce in sorted(ceilings.items()):
        if ce is None or "hkairport01-a" in ds_rel:
            continue
        ax.axvline(ce, color="k" if "amtown" in ds_rel else "grey", ls=":", lw=1.3,
                   label=f"{Path(ds_rel).name} tilt ceiling 1/cos θmax = {ce:.4f}")
    ax.set_xlim(1.0, 1.045)
    ax.set_xlabel("per-frame anisotropy  σ₁/σ₂  of  D$_k^{-1}$  at the image centre")
    ax.set_ylabel("cumulative fraction of frames")
    ax.set_title("(a) ONE FRAME — essentially a similarity\n"
                 "almost every frame sits inside what the airframe's own tilt explains", fontsize=10)
    ax.legend(fontsize=7.2, loc="lower right")
    ax.grid(alpha=0.25)

    # ---- (b) per-frame log-scale increment: the bias is far smaller than the spread -----------
    ax = axes[0][1]
    for window, model, _, _ in PER_FRAME_ARMS:
        v = per_frame[(window, model)]["inc_log_scale"]
        s = mad_sigma(v)
        ax.hist(v, bins=200, range=(-6e-3, 6e-3), histtype="step", density=True, lw=1.2,
                color=COLOUR[(window, model)], linestyle=STYLE[model],
                label=f"{label(window, model)} — mean {v.mean():+.2e}, "
                      f"robust σ {s:.2e}  ({s / abs(v.mean()):.1f}×)")
        ax.axvline(v.mean(), color=COLOUR[(window, model)], lw=1.6, alpha=0.9)
        # Robust ±1σ ticks, so "the bias is smaller than the spread" is a measurement on the axis
        # rather than an assertion in the title.
        for sign in (-1, 1):
            ax.plot([v.mean() + sign * s], [0], marker="|", ms=14, mew=1.6,
                    color=COLOUR[(window, model)])
    ax.axvline(0.0, color="k", lw=1.0)
    ax.set_xlim(-6e-3, 6e-3)
    ax.set_xlabel("per-frame log-scale increment  log √|det J|\n"
                  "thick vertical line = arm's mean;  ticks on the axis = ±1 robust σ")
    ax.set_ylabel("density")
    ax.set_title("(b) ONE FRAME — the bias is invisible inside the noise\n"
                 "the mean is 3–6× smaller than the robust per-frame spread", fontsize=10)
    ax.legend(fontsize=6.6, loc="upper left")
    ax.grid(alpha=0.25)

    # ---- (c) accumulated scale ----------------------------------------------------------------
    ax = axes[1][0]
    for window, model, _, _ in ARMS:
        s = per_frame[(window, model)]["accum_scale"]
        v = per_frame[(window, model)]["inc_log_scale"]
        ax.plot(s, STYLE[model], lw=1.2, color=COLOUR[(window, model)],
                label=f"{label(window, model)} — ends {s[-1]:.2f}×")
        # The deterministic prediction from the arm's OWN mean increment, drawn to make the
        # arithmetic visible: exp(n * mean). It is not a fit; it uses only panel (b)'s number.
        n = np.arange(s.size)
        ax.plot(np.exp(n * v.mean()), ":", lw=1.0, color=COLOUR[(window, model)], alpha=0.6)
    ax.axhspan(*SCALE_BAND, color="tab:green", alpha=0.16,
               label=f"physically explicable band {SCALE_BAND[0]}×–{SCALE_BAND[1]}×\n"
                     "(LiDAR-measured height range, both flights)")
    ax.axhline(1.0, color="k", lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("frame index")
    ax.set_ylabel("accumulated visual scale  exp(Σ log √|det J|)")
    ax.set_title("(c) A FEW THOUSAND FRAMES — accumulated scale\n"
                 "dotted = exp(n × that arm's own mean increment from panel (b))", fontsize=10)
    ax.legend(fontsize=7.0, loc="upper left", ncol=1)
    ax.grid(alpha=0.25, which="both")

    # ---- (d) accumulated anisotropy -----------------------------------------------------------
    ax = axes[1][1]
    y_top = 40.0
    off_scale = []
    for window, model, _, ds_rel in ARMS:
        a = accum[(window, model)]
        ax.plot(a, STYLE[model], lw=1.2, color=COLOUR[(window, model)],
                label=f"{label(window, model)} — median {np.median(a):.2f}")
        if a.max() > y_top:
            off_scale.append((label(window, model), float(a.max())))
    ceil_vals = [c for d, c in ceilings.items() if c]
    if ceil_vals:
        ax.axhspan(1.0, max(ceil_vals), color="tab:green", alpha=0.16,
                   label=f"tilt-explained ceiling ≤ {max(ceil_vals):.4f}")
    ax.axhline(1.0, color="k", lw=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.95, y_top)
    if off_scale:
        # Clipping is stated, not silent: the excursions are real and their size is printed.
        ax.text(0.985, 0.03,
                "clipped at the top — " + "; ".join(f"{n} peaks at {v:.3g}" for n, v in off_scale),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6, color="tab:red")
    ax.set_xlabel("frame index")
    ax.set_ylabel("accumulated anisotropy  σ₁/σ₂  of  G$_k^{-1}$  at the image centre")
    ax.set_title("(d) A FEW THOUSAND FRAMES — accumulated anisotropy\n"
                 "the composed transform is not an isotropic scale at all", fontsize=10)
    ax.legend(fontsize=7.0, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("F17 — the failure is ACCUMULATION of small residual non-rigid terms, not frequent "
                 "catastrophic per-frame transforms\n"
                 "Same quantity, same runs, top row per frame and bottom row composed. "
                 "RIGID_MOTION exists to stop the bottom row happening.", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out / "F17_residual_accumulation.png", dpi=150)
    plt.close(fig)

    # ---- the numbers the caption quotes -------------------------------------------------------
    for window, model, run_rel, ds_rel in ARMS:
        pf, a = per_frame[(window, model)], accum[(window, model)]
        ceiling = ceilings[ds_rel]
        v = pf["inc_log_scale"]
        stats[f"{window}/{model}"] = {
            "n_frames": int(v.size),
            "per_frame_anisotropy_p99": float(np.percentile(pf["inc_anisotropy"], 99)),
            "per_frame_anisotropy_max": float(pf["inc_anisotropy"].max()),
            "per_frame_anisotropy_frac_above_tilt_ceiling":
                (float((pf["inc_anisotropy"] > ceiling).mean()) if ceiling else None),
            "per_frame_anisotropy_frac_above_1p10": float((pf["inc_anisotropy"] > 1.10).mean()),
            "per_frame_log_scale_mean": float(v.mean()),
            "per_frame_log_scale_mad_sigma": mad_sigma(v),
            "per_frame_spread_over_bias": float(mad_sigma(v) / abs(v.mean())),
            "accum_scale_end": float(pf["accum_scale"][-1]),
            "accum_anisotropy_median": float(np.median(a)),
            "accum_anisotropy_end": float(a[-1]),
            "accum_anisotropy_max": float(a.max()),
            "accum_anisotropy_frac_above_1p10": float((a > 1.10).mean()),
            "tilt_ceiling": ceiling,
            "n_rows_dropped": pf["n_dropped"],
            "n_restarts": pf["n_restart"],
        }
    hk = [k for k in stats if k.startswith("hkairport01")]
    derived["F17"] = {
        "figure": "F17_residual_accumulation.png",
        "definitions": {
            "per_frame": "sidecar inc_anisotropy / inc_log_scale — J(D_k^-1) at the image centre, "
                         "rows with event in ('init', 'restart') dropped (EXP-VO-011's filter)",
            "accum_scale": "sidecar rigid_accum_scale = exp(sum inc_log_scale)",
            "accum_anisotropy": "sigma1/sigma2 of J(G_k^-1) at the image centre",
            "tilt_ceiling": "1/cos(max tilt_deg) from the dataset's own attitude.csv — the same "
                            "convention as the 2026-08-26 report's F5, which is why it matches "
                            "VO_CANONICAL_RESULTS.md section 1 and not scale_budget.json's "
                            "per-window figure",
        },
        "cross_check_against": "evaluations/exp-vo-011/scale_budget.json (inc_log_scale n/mean/"
                               "mad_sigma, exact)",
        "cross_checks_passed": len(checks),
        "scale_band_drawn": list(SCALE_BAND),
        # The ranges the closure package quotes, recomputed here so the figure and the prose cannot
        # drift apart. Accumulated anisotropy 1.35-3.39 is an HKairport01 range (EXP-VO-004 R4 ran
        # only that flight); the AMtown01 values are computed here for the first time.
        "accum_scale_range_all_arms": [min(s["accum_scale_end"] for s in stats.values()),
                                       max(s["accum_scale_end"] for s in stats.values())],
        "accum_anisotropy_median_range_hkairport01":
            [min(stats[k]["accum_anisotropy_median"] for k in hk),
             max(stats[k]["accum_anisotropy_median"] for k in hk)],
        "accum_anisotropy_median_range_all_arms":
            [min(s["accum_anisotropy_median"] for s in stats.values()),
             max(s["accum_anisotropy_median"] for s in stats.values())],
        "per_frame_anisotropy_frac_above_1p10_max":
            max(s["per_frame_anisotropy_frac_above_1p10"] for s in stats.values()),
        "arms": stats,
    }


# --------------------------------------------------------------------------- F8

def heading_series(run_rel: str, dataset_rel: str) -> dict:
    """The aligned yaw error series and its constant/tracking split, for one arm.

    Circular-safe throughout: every difference is wrapped into (−180, 180] and the offset is a
    circular mean (`atan2` of summed unit vectors), reusing `EXP-VO-010`'s own primitives. A naive
    arithmetic mean is wrong by up to 180° for series that straddle the wrap, and these offsets are
    62–77°, i.e. anywhere on the circle.
    """
    ds = load_dataset(REPO / dataset_rel)
    rr = load_run_record(REPO / run_rel)
    sync = synchronize(rr.timestamps_s, ds.gt_timestamps, ds.gt_east, ds.gt_north,
                       ds.gt_heading if ds.has_heading else None, ds.gt_valid,
                       clock_offset_s=ds.clock_offset_s,
                       clock_drift_s_per_s=ds.clock_drift_s_per_s)
    idx = sync.frame_indices
    est = np.stack([rr.est_x[idx], rr.est_y[idx]], axis=1)
    gt = np.stack([sync.east, sync.north], axis=1)
    al = fit_sim2(est, gt)

    err = wrap180(al.transform_yaw(rr.est_yaw_deg[idx]) - sync.heading)
    offset = circular_mean_deg(err)
    residual = wrap180(err - offset)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    return {
        "distance_m": dist,
        "aligned_error_deg": err,
        "residual_deg": residual,
        "constant_offset_deg": float(offset),
        "raw_yaw_rmse_deg": float(np.sqrt((err ** 2).mean())),
        "aligned_heading_rms_deg": circular_rms_about_deg(err, offset),
        "offset_fraction_of_raw": float(abs(offset) / np.sqrt((err ** 2).mean())),
        "n": int(err.size),
    }


def committed_heading(window: str, model: str) -> dict | None:
    """The same three quantities as `EXP-VO-010` committed them, for cross-checking the figure."""
    path = REPO / "evaluations/exp-vo-010/comparison.json"
    if not path.exists():
        return None
    arms = json.loads(path.read_text())["windows"].get(window, {})
    return arms.get(f"{model}-refineOff", {}).get("heading")


def figure_f8(out: Path, derived: dict) -> None:
    """Raw aligned yaw error, the fitted constant offset, and what is left after removing it."""
    series = {(w, m): heading_series(r, d) for w, m, r, d in ARMS}

    # Cross-check every drawn number against the committed EXP-VO-010 artifact before plotting it.
    checks = []
    for (w, m), s in series.items():
        ref = committed_heading(w, m)
        if not ref:
            continue
        for key, tol in (("raw_yaw_rmse_deg", 5e-3), ("aligned_heading_rms_deg", 5e-3),
                         ("offset_fraction_of_raw", 5e-4)):
            ok = abs(s[key] - ref[key]) <= tol * max(1.0, abs(ref[key]))
            checks.append({"arm": f"{w}/{m}", "key": key, "mine": s[key], "committed": ref[key],
                           "ok": bool(ok)})
        d_off = abs(float(wrap180(s["constant_offset_deg"] - ref["constant_offset_deg"])))
        checks.append({"arm": f"{w}/{m}", "key": "constant_offset_deg",
                       "mine": s["constant_offset_deg"], "committed": ref["constant_offset_deg"],
                       "ok": bool(d_off <= 0.01)})
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise SystemExit(f"F8 cross-check against evaluations/exp-vo-010/comparison.json failed:\n"
                         + json.dumps(failed, indent=2))

    fig = plt.figure(figsize=(16.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.35), wspace=0.24,
                          left=0.045, right=0.99, top=0.79, bottom=0.12)

    for col, window in enumerate(("hkairport01-b", "amtown01-c")):
        ax = fig.add_subplot(gs[0, col])
        for model in ("homography", "affine"):
            s = series[(window, model)]
            ax.plot(s["distance_m"], s["aligned_error_deg"], STYLE[model], lw=0.8,
                    color=COLOUR[(window, model)], alpha=0.85,
                    label=f"{model} — aligned yaw error, RMSE {s['raw_yaw_rmse_deg']:.1f}°")
            ax.axhline(s["constant_offset_deg"], color=COLOUR[(window, model)], ls=":", lw=1.6)
            ax.plot(s["distance_m"], s["residual_deg"], STYLE[model], lw=0.8,
                    color=COLOUR[(window, model)],
                    label=f"{model} — after removing the offset, "
                          f"{s['aligned_heading_rms_deg']:.2f}° rms")
        s = series[(window, "homography")]
        ax.annotate(f"fitted constant offset\n{s['constant_offset_deg']:+.1f}°",
                    xy=(s["distance_m"][-1] * 0.55, s["constant_offset_deg"]),
                    xytext=(s["distance_m"][-1] * 0.30,
                            s["constant_offset_deg"] + (18 if s["constant_offset_deg"] < 0 else -22)),
                    fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.9))
        ax.axhline(0.0, color="k", lw=1.0)
        ax.set_ylim(-95, 95)
        ax.set_xlabel("distance travelled (m)")
        ax.set_ylabel("heading error (deg)")
        ax.set_title(f"({'ab'[col]}) {window} — the SAME series, twice\n"
                     "upper/lower cloud: before and after removing one constant", fontsize=10)
        # Park the legend in whichever half the two clouds have left empty.
        ax.legend(fontsize=6.9, framealpha=0.9,
                  loc="upper right" if s["constant_offset_deg"] < 0 else "lower right")
        ax.grid(alpha=0.25)

    # ---- (c) the decomposition across all six reference arms ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    keys = [(w, m) for w, m, _, _ in ARMS]
    x = np.arange(len(keys))
    raw = [series[k]["raw_yaw_rmse_deg"] for k in keys]
    off = [abs(series[k]["constant_offset_deg"]) for k in keys]
    res = [series[k]["aligned_heading_rms_deg"] for k in keys]
    b1 = ax.bar(x - 0.27, raw, 0.26, color="tab:grey", label="raw yaw RMSE — what was published")
    b2 = ax.bar(x, off, 0.26, color="tab:purple", label="|fitted constant frame offset|")
    b3 = ax.bar(x + 0.27, res, 0.26, color="tab:green",
                label="residual heading tracking, rms about that offset")
    for b, fmt in ((b1, "%.1f"), (b2, "%.1f"), (b3, "%.2f")):
        ax.bar_label(b, fmt=fmt, fontsize=7.0, padding=2, rotation=90)
    for i, k in enumerate(keys):
        ax.text(i, -7.0, f"{100 * series[k]['offset_fraction_of_raw']:.2f} %", ha="center",
                fontsize=7.2, color="tab:purple")
    ax.text(len(keys) - 0.45, -14.0, "offset ÷ raw", ha="right", fontsize=7.4, color="tab:purple")
    short = {"hkairport01-a": "hk-a\n(prefix\nof -b)", "hkairport01-b": "hk-b", "amtown01-c": "AM-c"}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{short[w]}\n{m[:4]}" for w, m in keys], fontsize=7.4)
    ax.set_ylim(-17, 108)
    ax.set_ylabel("degrees")
    ax.set_title("(c) every reference arm, decomposed\n"
                 "the published metric is 99.2–99.9 % one constant", fontsize=10)
    ax.legend(fontsize=7.4, loc="upper center")
    ax.grid(axis="y", alpha=0.25)

    fig.text(0.5, 0.965,
             "F8 — the historical 50–150° yaw figures are overwhelmingly a constant DATUM OFFSET, "
             "not heading drift", ha="center", va="top", fontsize=12.5)
    fig.text(0.5, 0.905,
             "The estimator's yaw datum is its own first frame; naveval corrects by the Sim(2) "
             "rotation fitted to POSITIONS. Those are different quantities.\n"
             "Removing one constant per arm leaves 3.1–9.4° rms of actual heading tracking. "
             "Circular-safe throughout: circular mean, differences wrapped into (−180, 180].",
             ha="center", va="top", fontsize=9.5)
    fig.savefig(out / "F8_raw_vs_offset_aligned_yaw.png", dpi=150)
    plt.close(fig)

    derived["F8"] = {
        "figure": "F8_raw_vs_offset_aligned_yaw.png",
        "cross_check_against": "evaluations/exp-vo-010/comparison.json (refineOff arms)",
        "cross_checks_passed": len(checks),
        "arms": {f"{w}/{m}": {k: v for k, v in series[(w, m)].items()
                              if not isinstance(v, np.ndarray)} for w, m in keys},
        "aligned_heading_rms_range_deg": [min(res), max(res)],
        "offset_fraction_range": [min(series[k]["offset_fraction_of_raw"] for k in keys),
                                  max(series[k]["offset_fraction_of_raw"] for k in keys)],
    }


# --------------------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/vo-closure",
                    help="directory for the two PNGs and closure_figures.json")
    ap.add_argument("--only", choices=("f17", "f8"), default=None)
    a = ap.parse_args()
    out = (REPO / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    derived: dict = {"generated_from": "committed artifacts only — runs/, datasets/, "
                                       "evaluations/exp-vo-010/comparison.json",
                     "working_resolution_px": [WIDTH, HEIGHT]}
    if a.only in (None, "f17"):
        figure_f17(out, derived)
        print(f"wrote {out / 'F17_residual_accumulation.png'}")
    if a.only in (None, "f8"):
        figure_f8(out, derived)
        print(f"wrote {out / 'F8_raw_vs_offset_aligned_yaw.png'}")
    (out / "closure_figures.json").write_text(json.dumps(derived, indent=2))
    print(f"wrote {out / 'closure_figures.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
