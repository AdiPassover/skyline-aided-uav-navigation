"""The shared extraction-evaluation harness — one code path scores every method.

Spec FR-006/007/010; thresholds and the catastrophic rule come from plan.md §Pre-registration
(commit ``121d80c``) and are **frozen**: nothing here may be re-tuned after a final-split number
exists. The harness enforces the process rules mechanically where it can:

- a **final split** run (``eval_only`` or ``eval``) requires the method's frozen-config digest file
  to exist and match, requires the pre-registration reference, and refuses to overwrite an
  existing result — re-running is a new pre-registration, not an overwrite;
- every record carries its evidence tier and caveat (from the GT store manifest) and the
  pre-registration reference, so no exported number can shed its provenance;
- an image-level **catastrophic** flag is computed separately from accuracy and neither hides the
  other (the Web-dataset lesson: μ 7.85 with catastrophes pooled in vs 0.79 with them out).

Per-image metrics are computed over the **intersection** of GT-valid and prediction-valid columns;
an explicit extraction failure carries MAE = +inf so it lands at the worst end of every
distribution instead of quietly vanishing from the denominator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import cv2
import numpy as np

from hsreloc.extraction.curves import CanonicalCurve, valid_intersection
from hsreloc.extraction.gt.store import load_store
from hsreloc.extraction.methods.base import (
    STATUS_FAILURE,
    STATUS_OK,
    ExtractionOutput,
    get_method,
)
from hsreloc.retrieval.record import code_revision

# Frozen by plan.md §Pre-registration (121d80c). Do not edit without a new pre-registration.
TOLERANCES_PX = (1, 2, 5, 10, 20)
CATASTROPHIC_MAE_NORM = 0.05
CATASTROPHIC_MIN_COVERAGE = 0.50
FINAL_SPLITS = ("eval_only", "eval")


class EvaluationError(Exception):
    """The evaluation cannot proceed as configured. Always names the rule that refused."""


def wilson_interval(k: int, n: int, z: float = 1.959963984540054):
    """Wilson 95 % interval for a binomial rate — the lane's standard for every reported rate."""
    if n <= 0:
        raise EvaluationError("Wilson interval over an empty denominator")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = 0.0 if k == 0 else max(0.0, centre - half)   # exact at the degenerate counts;
    hi = 1.0 if k == n else min(1.0, centre + half)   # the formula leaves fp residue there
    return lo, hi


def per_image_metrics(gt: CanonicalCurve, output: ExtractionOutput) -> dict:
    """The FR-006 metric set for one image. Failure ⇒ MAE = +inf and catastrophic = True."""
    height = gt.height_px
    base = {
        "status": output.status,
        "failure_reason": output.reason or "",
        "runtime_s": output.runtime_s,
        "gt_invalid_rate": 1.0 - gt.n_valid / gt.width_px,
    }
    if output.status == STATUS_FAILURE:
        base.update({"coverage": 0.0, "pred_invalid_rate": 1.0,
                     "mae_px": math.inf, "median_px": math.inf, "p90_px": math.inf,
                     "p95_px": math.inf, "max_px": math.inf,
                     "mae_norm": math.inf,
                     **{f"frac_within_{t}px": 0.0 for t in TOLERANCES_PX},
                     "catastrophic": True, "catastrophic_reason": "extraction_failure"})
        return base

    pred = output.curve
    both = valid_intersection(gt, pred)
    coverage = float(both.sum()) / gt.n_valid
    base["coverage"] = coverage
    base["pred_invalid_rate"] = 1.0 - pred.n_valid / pred.width_px
    if both.sum() == 0:
        base.update({"mae_px": math.inf, "median_px": math.inf, "p90_px": math.inf,
                     "p95_px": math.inf, "max_px": math.inf, "mae_norm": math.inf,
                     **{f"frac_within_{t}px": 0.0 for t in TOLERANCES_PX},
                     "catastrophic": True, "catastrophic_reason": "no_overlapping_valid_columns"})
        return base

    err = np.abs(np.asarray(gt.rows)[both] - np.asarray(pred.rows)[both])
    mae = float(err.mean())
    base.update({
        "mae_px": mae,
        "median_px": float(np.median(err)),
        "p90_px": float(np.percentile(err, 90)),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(err.max()),
        "mae_norm": mae / height,
    })
    for t in TOLERANCES_PX:
        base[f"frac_within_{t}px"] = float((err <= t).mean())

    if mae / height > CATASTROPHIC_MAE_NORM:
        base.update({"catastrophic": True, "catastrophic_reason": "mae_norm_exceeded"})
    elif coverage < CATASTROPHIC_MIN_COVERAGE:
        base.update({"catastrophic": True, "catastrophic_reason": "coverage_below_floor"})
    else:
        base.update({"catastrophic": False, "catastrophic_reason": ""})
    return base


def aggregate(per_image: list) -> dict:
    """Distributions over images — never a single pooled mean (FR-007)."""
    if not per_image:
        raise EvaluationError("nothing to aggregate")
    n = len(per_image)
    n_fail = sum(1 for m in per_image if m["status"] == STATUS_FAILURE)
    n_cat = sum(1 for m in per_image if m["catastrophic"])
    lo, hi = wilson_interval(n_cat, n)
    maes_all = [m["mae_px"] for m in per_image]
    noncat = [m for m in per_image if not m["catastrophic"]]

    def dist(values):
        if not values:
            return {"median": None, "p90": None, "mean": None}
        a = np.asarray(values, dtype=np.float64)
        return {"median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
                "mean": float(a.mean())}

    agg = {
        "n_images": n,
        "n_extraction_failures": n_fail,
        "catastrophic_count": n_cat,
        "catastrophic_rate": n_cat / n,
        "catastrophic_rate_wilson95": [lo, hi],
        # B1's number: the median over ALL images (failures included as +inf).
        "median_mae_all_images_px": float(np.median(np.asarray(maes_all))),
        # Comparability with the published convention that excludes catastrophic images.
        "mae_over_noncatastrophic_px": dist([m["mae_px"] for m in noncat]),
        "median_px_over_noncatastrophic": dist([m["median_px"] for m in noncat]),
        "p90_px_over_noncatastrophic": dist([m["p90_px"] for m in noncat]),
        "coverage": dist([m["coverage"] for m in per_image]),
        "gt_invalid_rate": dist([m["gt_invalid_rate"] for m in per_image]),
        "pred_invalid_rate": dist([m["pred_invalid_rate"] for m in per_image]),
        "runtime_s": dist([m["runtime_s"] for m in per_image]),
    }
    for t in TOLERANCES_PX:
        agg[f"frac_within_{t}px_over_noncatastrophic"] = dist(
            [m[f"frac_within_{t}px"] for m in noncat])
    return agg


def method_config_digest(method_block: dict) -> str:
    return hashlib.sha256(
        json.dumps(method_block, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve(config_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (config_dir / p).resolve()


def run_evaluation(config: dict, config_path: Path) -> dict:
    """Run one method over one GT-store split and persist the evaluation record."""
    config_dir = Path(config_path).resolve().parent
    method_block = config["method"]
    digest = method_config_digest(method_block)
    split = config["split"]
    store_dir = _resolve(config_dir, config["gt_store"])
    out_dir = _resolve(config_dir, config["out_dir"])
    benchmark_root = Path(config["benchmark_root"])

    manifest, samples = load_store(store_dir)
    selected = [s for s in samples if s["split"] == split]
    if not selected:
        raise EvaluationError(f"no samples with split {split!r} in {store_dir}")

    final = split in FINAL_SPLITS
    prereg = config.get("preregistration_ref", "")
    if final:
        if not prereg:
            raise EvaluationError(
                "final-split run without preregistration_ref — the frozen criteria commit must be "
                "named before this run may exist")
        digest_file = config.get("frozen_digest_file", "")
        if not digest_file:
            raise EvaluationError(
                "final-split run without frozen_digest_file — freeze the method config first")
        frozen = _resolve(config_dir, digest_file)
        if not frozen.exists():
            raise EvaluationError(f"frozen digest file {frozen} does not exist")
        expect = frozen.read_text(encoding="utf-8").strip()
        if expect != digest:
            raise EvaluationError(
                f"method config digest {digest[:12]}… does not match frozen {expect[:12]}… — "
                f"the config changed after the freeze")
        if (out_dir / "metrics.json").exists():
            raise EvaluationError(
                f"{out_dir} already holds a final-split result — re-running is a new "
                f"pre-registration and a new output directory, never an overwrite")

    method = get_method(method_block["id"])
    method.configure(method_block.get("params", {}))

    per_image = []
    for s in selected:
        img_path = benchmark_root / s["image_relpath"]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise EvaluationError(f"{s['sample_id']}: cannot read image {img_path}")
        if image.shape[:2] != (s["height_px"], s["width_px"]):
            raise EvaluationError(
                f"{s['sample_id']}: image is {image.shape[1]}x{image.shape[0]} but the store "
                f"declares {s['width_px']}x{s['height_px']}")
        output = method.extract(image)
        output.validate()
        row = {"sample_id": s["sample_id"], "camera_id": s["camera_id"], "split": s["split"]}
        row.update(per_image_metrics(s["curve"], output))
        per_image.append(row)

    agg = aggregate(per_image)
    record = {
        "method": method_block,
        "method_config_digest": digest,
        "gt_store": store_dir.name,
        "dataset": config.get("dataset", store_dir.name),
        "split": split,
        "final_split": final,
        "preregistration_ref": prereg,
        "tier": manifest["tier"],
        "evidence_caveat": manifest["evidence_caveat"],
        "conversion_version": manifest.get("conversion_version", ""),
        "code_revision": code_revision(),
        "hardware": platform.platform(),
        "is_target_hardware": False,
        "aggregates": agg,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(per_image[0].keys())
    with (out_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in per_image:
            w.writerow(row)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    return record
