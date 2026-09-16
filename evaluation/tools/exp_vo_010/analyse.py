"""`EXP-VO-010` Phases 6, 7, 9 and 10: every arm scored in one table.

Twelve arms — three windows × {affine, homography} × {refine false, true} — with the `refine = false`
side supplied by the committed `EXP-VO-006`/`EXP-VO-007` baselines and every arm re-scored through
the *same* code, so no number in the comparison comes from a different computation than any other.

Primary metrics and per-frame geometric diagnostics are imported unchanged from `EXP-VO-007`'s
`analyse` module, exactly as `EXP-VO-009` did, so this experiment's figures are directly comparable
with those records and with the 2026-08-26 synthesis report: one global Sim(2) alignment
(`DEC-003`), `naveval` untouched, the same RPE lengths.

Heading is reported through `heading_metrics`, which splits the aligned yaw error into the constant
frame offset `EXP-VO-009` R2 identified and the tracking component underneath it. **The raw
`yaw_rmse_deg` is carried alongside for continuity with prior records and is not used as evidence.**

Runtime carries `EXP-VO-002`'s confound and is reported with track and inlier counts beside it; the
authoritative timing figures come from `timing_compare.py`, which runs the arms back to back.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Both EXP-VO-007 and this experiment name their driver `analyse`, so load by path.
_a7 = _load("exp_vo_007_analyse", EVALUATION_DIR / "tools" / "exp_vo_007" / "analyse.py")
_hm = _load("exp_vo_010_heading_metrics", Path(__file__).resolve().parent / "heading_metrics.py")

from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.runrecord import load_run_record                            # noqa: E402

MODELS = ("affine", "homography")
WINDOWS = {
    "hkairport01-a": "datasets/hkairport01-a",
    "hkairport01-b": "datasets/hkairport01-b",
    "amtown01-c": "datasets/amtown01-c",
}


def run_dir(window: str, model: str, refine: bool) -> Path:
    """`refine = false` reuses the committed EXP-VO-006/007 rigid baselines verbatim."""
    suffix = "rigid-refine-v1" if refine else "rigid-v1"
    return REPO / "runs" / f"{window}-{model}-{suffix}"


def support_and_cost(rd: Path) -> dict:
    rr = load_run_record(rd)
    tracks = np.array([f.track_count for f in rr.frame_estimates
                       if f.track_count is not None], dtype=float)
    inliers = np.array([f.inlier_count for f in rr.frame_estimates
                        if f.inlier_count is not None], dtype=float)
    times = np.array([f.process_time_ns for f in rr.frame_estimates
                      if f.process_time_ns is not None], dtype=float)
    out = {}
    if tracks.size:
        out["track_median"] = float(np.median(tracks))
        out["track_p05"] = float(np.percentile(tracks, 5))
    if inliers.size:
        out["inlier_median"] = float(np.median(inliers))
        out["inlier_p05"] = float(np.percentile(inliers, 5))
    if tracks.size and inliers.size:
        n = min(tracks.size, inliers.size)
        ratio = np.divide(inliers[:n], tracks[:n], out=np.full(n, np.nan), where=tracks[:n] > 0)
        out["inlier_ratio_median"] = float(np.nanmedian(ratio))
    if times.size:
        out["ms_per_frame_median"] = float(np.median(times) / 1e6)
        out["ms_per_frame_p95"] = float(np.percentile(times, 95) / 1e6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="hkairport01-a,hkairport01-b,amtown01-c")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    result: dict = {"windows": {}}
    for window in [w.strip() for w in a.windows.split(",") if w.strip()]:
        ds = load_dataset(REPO / WINDOWS[window])
        rows = {}
        for model in MODELS:
            for refine in (False, True):
                rd = run_dir(window, model, refine)
                key = f"{model}-refine{'On' if refine else 'Off'}"
                if not (rd / "frames.csv").exists():
                    print(f"  {window:14s} {key:20s} MISSING {rd}")
                    continue
                ev = _a7.evaluate(rd, ds)
                row = {k: v for k, v in ev.items() if not k.startswith("_")}
                row["run_dir"] = str(rd.relative_to(REPO)).replace("\\", "/")
                row["refine"] = refine
                row["model"] = model
                row.update(support_and_cost(rd))
                diag = _a7.diagnostics(rd)
                row["diagnostics"] = {k: v for k, v in diag.items() if not k.startswith("_")}
                row["heading"] = _hm.analyse_run(
                    WINDOWS[window], str(rd.relative_to(REPO)).replace("\\", "/"))
                rows[key] = row
                h = row["heading"]
                print(f"  {window:14s} {key:20s} normATE {100 * row['ate_rmse_normalised']:7.3f} %  "
                      f"perFrameSD {h.get('per_frame_sd_deg', float('nan')):.4f}  "
                      f"alignedRMS {h['aligned_heading_rms_deg']:6.2f} deg  "
                      f"drift {h.get('cumulative_drift_deg', float('nan')):+7.2f}  "
                      f"rawYaw {h['raw_yaw_rmse_deg']:6.2f}  "
                      f"inl {row.get('inlier_median', float('nan')):6.0f}  "
                      f"{row.get('ms_per_frame_median', float('nan')):6.2f} ms", flush=True)
        result["windows"][window] = rows

    # Relative changes, refine=true against the committed refine=false baseline.
    deltas = {}
    for window, rows in result["windows"].items():
        for model in MODELS:
            off, on = rows.get(f"{model}-refineOff"), rows.get(f"{model}-refineOn")
            if not off or not on:
                continue
            def rel(a_, b_):
                return float((b_ - a_) / a_) if a_ else float("nan")
            deltas[f"{window}/{model}"] = {
                "ate_rel": rel(off["ate_rmse_normalised"], on["ate_rmse_normalised"]),
                "per_frame_sd_rel": rel(off["heading"].get("per_frame_sd_deg", float("nan")),
                                        on["heading"].get("per_frame_sd_deg", float("nan"))),
                "aligned_heading_rms_rel": rel(off["heading"]["aligned_heading_rms_deg"],
                                               on["heading"]["aligned_heading_rms_deg"]),
                "cumulative_drift_abs_rel": rel(abs(off["heading"].get("cumulative_drift_deg", np.nan)),
                                                abs(on["heading"].get("cumulative_drift_deg", np.nan))),
                "ms_per_frame_rel": rel(off.get("ms_per_frame_median", float("nan")),
                                        on.get("ms_per_frame_median", float("nan"))),
                "inlier_ratio_rel": rel(off.get("inlier_ratio_median", float("nan")),
                                        on.get("inlier_ratio_median", float("nan"))),
                "restarts_off": off["n_restarts"], "restarts_on": on["n_restarts"],
                "success_off": off["success_rate"], "success_on": on["success_rate"],
            }
    result["refine_deltas"] = deltas

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
