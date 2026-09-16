"""Build the VO closure package's canonical results tables from committed artifacts.

Reads only tracked JSON under `evaluations/`, `datasets/` and `runs/` -- no estimator run,
no imagery -- so it works in a fresh checkout. Emits `canonical_results.json`, the machine-readable
form of the VO result tables.

Nothing here computes a new result. Every number is read from the artifact the experiment
that produced it committed; the only arithmetic is percentage differences between two such
numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Window identity. `hkairport01-a` is a strict 180 s prefix of `hkairport01-b`; they are
# NOT independent flights and must never be averaged or counted as two. `amtown01-c` is the
# one independent-scene window, and is itself a 14-of-27 sub-window prefix of its cruise.
WINDOWS = {
    "hkairport01-a": {"role": "development", "prefix_of": "hkairport01-b"},
    "hkairport01-b": {"role": "development (primary)", "prefix_of": None},
    "amtown01-c": {"role": "independent validation", "prefix_of": None},
}

# EXP-VO-006 keys its arms by the short window letter; EXP-VO-007 has one window so it does not.
SIX = {"hkairport01-a": "a", "hkairport01-b": "b"}


def read_json(rel: str):
    with (ROOT / rel).open(encoding="utf-8") as fh:
        return json.load(fh)


def flatten_naveval(doc: dict) -> dict:
    """A raw `naveval` metrics.json carries a LIST of metric objects under "metrics", plus
    separate "runtime" and "event_stats" blocks. The per-arm blocks inside an experiment's
    own summary.json are already a flat dict. Both shapes appear in this package; this maps
    the first onto the second so `pick` sees one shape. Names differ in two places, so they
    are mapped explicitly rather than by guessing."""
    flat = {m["name"]: m["value"] for m in doc.get("metrics", [])}
    flat.setdefault("ate_rmse_m", flat.get("ate_rmse"))
    flat.setdefault("endpoint_error_m", flat.get("endpoint_error"))
    flat.setdefault("yaw_rmse_deg", flat.get("yaw_rmse"))
    # `path_length` is not emitted as a metric row; recover it from ATE and its normalisation.
    if flat.get("ate_rmse") and flat.get("ate_rmse_normalised"):
        flat.setdefault("path_length_m", flat["ate_rmse"] / flat["ate_rmse_normalised"])
    flat["runtime"] = doc.get("runtime", {})
    ev = doc.get("event_stats", {})
    flat["n_restarts"] = ev.get("n_restarts")
    flat["n_recenters"] = ev.get("n_recenters")
    flat["success_rate"] = ev.get("success_rate")
    return flat


def pick(metrics: dict) -> dict:
    """The subset of a naveval metrics block the canonical tables quote."""
    out = {
        k: metrics.get(k)
        for k in (
            "ate_rmse_m", "ate_rmse_normalised", "endpoint_error_m", "path_length_m",
            "yaw_rmse_deg", "rpe_10m", "rpe_50m", "rpe_200m", "rpe_400m",
            "n_recenters", "n_restarts", "success_rate", "track_count_median",
            "discontinuities_5x_median_step", "alignment_scale", "n_points",
        )
    }
    rt = metrics.get("runtime") or {}
    out["ms_per_frame_median"] = rt.get("median_ms")
    return out


def navigation_table() -> dict:
    """Window x model x readout -- the main comparison. Sources, in priority order:
    EXP-VO-006 for HKairport01 legacy/full-logical/rigid, EXP-VO-007 for AMtown01,
    EXP-VO-009 for the similarity arm on all three windows."""
    table: dict[str, dict[str, dict]] = {w: {} for w in WINDOWS}

    s6 = read_json("evaluations/exp-vo-006/summary.json")["metrics"]
    for win, letter in SIX.items():
        for model in ("homography", "affine"):
            for src in ("legacy", "full_logical", "rigid"):
                key = f"{letter}-{model}-{src}"
                if key in s6:
                    table[win][f"{model}/{src}"] = pick(s6[key]) | {
                        "source": f"evaluations/exp-vo-006/summary.json::metrics.{key}"}

    s7 = read_json("evaluations/exp-vo-007/summary.json")["metrics"]
    alias = {"legacy": "legacy", "logical": "full_logical", "rigid": "rigid"}
    for key, block in s7.items():
        model, src = key.split("-")
        table["amtown01-c"][f"{model}/{alias[src]}"] = pick(block) | {
            "source": f"evaluations/exp-vo-007/summary.json::metrics.{key}"}

    s9 = read_json("evaluations/exp-vo-009/comparison.json")["windows"]
    for win, arms in s9.items():
        blk = arms.get("similarity-rigid")
        if blk:
            table[win]["similarity/rigid"] = pick(blk) | {
                "source": f"evaluations/exp-vo-009/comparison.json::windows.{win}.similarity-rigid"}
    return table


def refinement_table() -> dict:
    """EXP-VO-010's ablation, kept in its OWN table on purpose: the refine-on arms are a
    diagnostic configuration, never part of the reference configuration, and one of them
    (hkairport01-a homography) spans an avoided restart so it is not a clean ablation."""
    c10 = read_json("evaluations/exp-vo-010/comparison.json")
    out: dict = {"arms": {}, "deltas": c10["refine_deltas"]}
    for win, arms in c10["windows"].items():
        for key, blk in arms.items():
            hd = blk.get("heading") or {}
            extra = {k: hd.get(k) for k in (
                "per_frame_sd_deg", "per_frame_mad_sigma_deg", "aligned_heading_rms_deg",
                "cumulative_drift_deg", "per_frame_mean_deg", "raw_yaw_rmse_deg",
                "constant_offset_deg", "offset_fraction_of_raw")}
            extra["inlier_ratio_median"] = blk.get("inlier_ratio_median")
            out["arms"][f"{win}/{key}"] = pick(blk) | extra
    return out


def heading_table() -> dict:
    """The three heading quantities EXP-VO-009 R2 showed the raw yaw metric conflates.
    Raw `yaw_rmse_deg` is ~95 % a constant frame offset; the quantity that measures heading
    TRACKING is the rms about that mean. Both are carried so they cannot be confused again."""
    out = {}
    for win in WINDOWS:
        rel = f"evaluations/exp-vo-009/rotation_{win}.json"
        if (ROOT / rel).exists():
            out[win] = read_json(rel)
    return out


def scale_table() -> dict:
    """EXP-VO-011's per-frame log-scale bias -- the primitive quantity. The accumulated
    figures other records quote are exp(bias * n) of these."""
    return read_json("evaluations/exp-vo-011/scale_budget.json")


def substitution_table() -> dict:
    """Known-answer substitutions (EXP-VO-008's method). Diagnostics, not implementable."""
    return {tag: read_json(f"evaluations/exp-vo-008/subst_{tag}.json")
            for tag in ("amtown01-c_affine", "hkairport01-b_affine")}


def legacy_baseline_table() -> dict:
    """EXP-002 / EXP-VO-002 under the LEGACY readout. Historical: superseded as an accuracy
    figure by the rigid readout, preserved because it is what every pre-2026-08-25 record
    reports and because retracting it would be wrong -- the numbers are correct for that
    readout."""
    out = {}
    for name in ("exp-002-a", "exp-002-b"):
        out[name] = pick(flatten_naveval(read_json(f"evaluations/{name}/metrics.json")))
    for arm in ("a-homography", "a-affine", "b-homography", "b-affine"):
        out[f"exp-vo-002/{arm}"] = pick(
            flatten_naveval(read_json(f"evaluations/exp-vo-002/{arm}/metrics.json")))
    out["exp-vo-002/comparison"] = read_json("evaluations/exp-vo-002/comparison.json")
    return out


def dataset_table() -> dict:
    out: dict = {}
    for win in WINDOWS:
        d = read_json(f"datasets/{win}/dataset.json")
        out[win] = {k: d.get(k) for k in (
            "dataset_id", "source_type", "evidence_tier", "environment", "capture_date",
            "n_frames", "camera", "ground_truth")}
    out["_lidar_probe"] = read_json("evaluations/exp-vo-007/candidate_probe.json")
    out["_prefix_closure"] = read_json("evaluations/exp-vo-007/prefix_closure.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="evaluations/vo-closure")
    args = ap.parse_args()

    payload = {
        "_note": "Generated by evaluation/tools/vo_closure/canonical_results.py from "
                 "committed artifacts only. Do not hand-edit.",
        "windows": WINDOWS,
        "navigation": navigation_table(),
        "refinement": refinement_table(),
        "heading": heading_table(),
        "scale": scale_table(),
        "substitution": substitution_table(),
        "legacy_baseline": legacy_baseline_table(),
        "datasets": dataset_table(),
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "canonical_results.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
