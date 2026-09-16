"""Pre-registration for the first simulator retrieval experiment (contracts/preregistration.md).

``write_prereg`` freezes everything the run depends on — the task and grid digests, the six
reference ids, the matcher block, and every source's frozen identifiers — together with the
declared metric list, the 1/6 chance baseline, the four-case interpretation matrix and the
dataset-limitations text, all of which must exist **before** the first final matching run.
``check_prereg`` compares the built artifacts against the frozen file verbatim and refuses on the
first difference. Editing the file afterwards is a new pre-registration, and the runner also
refuses to overwrite any existing record, so a re-run is always a visible, deliberate event.

The declared interpretation matrix and limitations live here as module constants so that the
prereg file, the CLI and the experiment record all quote one source.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PREREG_VERSION = "1.0.0"

#: Declared BEFORE results exist (research R9 / contracts/preregistration.md).
INTERPRETATION_MATRIX = {
    "case_1_extraction_problem": (
        "GT retrieval succeeds (Recall@1 >= 5/6 on stage-0/near queries and >= 2/3 overall) while "
        "SegFormer (or DP) is materially lower (>= 2 queries flip) -> the bottleneck is extraction."),
    "case_2_representation_problem": (
        "GT itself fails (Recall@1 within 2 queries of the extractor sources and < 2/3 overall) -> "
        "the bottleneck is representation/matching/discriminability, not extraction."),
    "case_3_viable": (
        "GT and at least one extractor both >= 2/3 overall with positive margins on the majority of "
        "correct answers -> the current pipeline is viable in this regime."),
    "case_4_geometry_dominated": (
        "GT Recall@1 declines monotonically with displacement bin and falls below 1/2 in the largest "
        "bin -> characterise the geometric deformation before any matcher work."),
    "note": "evaluated separately per query axis (displacement; condition); the axes may land in "
            "different cases",
}

METRICS = [
    "Recall@1", "Recall@k", "correct-reference rank", "correct-reference score",
    "best-incorrect score", "score margin", "acceptance/rejection", "confident-false rate",
    "sliced by condition_translation_m bin, direction class, condition_hour (time of day), "
    "condition_clouds, and curve source",
]

LIMITATIONS = (
    "PILOT/DEV evidence, naveval tier T2 (ue5_simulator): one level "
    "(asian_village_hills_background), six anchors (chance baseline 1/6), uneven condition cells "
    "(anchor a0 has no DUSK and carries the only CLOUDY captures), manual capture spacing "
    "(~11-25 m), one base altitude, pitch/roll signs declared but unverified (no R0-S render), "
    "quaternion convention not discriminable on a level batch. No DEV/final split is possible with "
    "six anchors, so no parameter is tuned here; descriptive statistics only, no significance "
    "claims, no real-flight or onboard claim."
)


class PreregError(Exception):
    """The configuration is not the pre-registered experiment."""


def _eq(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _source_pins(descriptions: dict) -> dict:
    out = {}
    for key, d in sorted(descriptions.items()):
        if key == "sim_exact":
            out[key] = {"provenance": d.get("provenance"), "gt": True, "kind": d.get("kind")}
        elif key == "dp":
            out[key] = {k: d.get(k) for k in ("provenance", "method_id", "method_config_digest")}
        elif key == "segformer":
            out[key] = {k: d.get(k) for k in
                        ("provenance", "method_id", "model_revision", "label_map_digest",
                         "scale_tag", "convert_version", "closing_px", "min_valid_frac",
                         "full_width_policy", "policy_version")}
        else:
            raise PreregError(f"unknown source key {key!r} in the pre-registration")
    return out


def write_prereg(path: Path, config: dict, task_manifest: dict, deform_manifest: dict,
                 source_descriptions: dict) -> dict:
    grids = task_manifest["grid_counts"]
    pre = {
        "prereg_version": PREREG_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature": "20260902-163608-sky-simulator-viewpoint",
        "experiment": "EXP-SKY-007",
        "task": {
            "name": config["task_name"],
            "spacings_m": [float(s) for s in config["tasks"]["spacings_m"]],
            "tolerances": task_manifest["tolerances"],
            "task_digest": task_manifest["content_digest"],
            "deform_task_digest": deform_manifest["content_digest"],
            "grid_counts": grids,
            "reference_observation_ids": list(config["tasks"]["reference_observation_ids"]),
            "stage_rule": task_manifest["stage_rule"],
        },
        "matcher": {"block": config["matcher"], "baselines": list(config["baselines"]),
                    "primary_baseline": "ncc"},
        "geodetic_origin": config["geodetic_origin"],
        "sources": _source_pins(source_descriptions),
        "metrics": METRICS,
        "chance_baseline": {"recall_at_1": 1.0 / 6.0, "n_references": 6},
        "interpretation_matrix": INTERPRETATION_MATRIX,
        "limitations": LIMITATIONS,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pre, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pre


# ==================================================================================================
# The FINAL-validation freeze (feature 20260903-225416; contracts/freeze.md). Additive: the pilot
# prereg above is a closed experiment's artifact and is not touched.
# ==================================================================================================

FINAL_PREREG_VERSION = "1.0.0"

#: Declared metric list for the held-out phase (research R9).
FINAL_METRICS = [
    "extraction: median/p90/p95 abs px, normalized, coverage, refusal rate, catastrophic frac "
    "(|err| > 5% height), signed bias — per level x time_of_day x clouds x TIMExCLOUD x anchor",
    "consistency: 36 pairs per anchor (TIME-held / CLOUD-held / mixed), raw + offset-removed, "
    "GT pairs asserted 0 px, refusal consistency, catastrophic switches",
    "retrieval: Recall@1/@3/@5, correct rank, correct/best-incorrect score, top1-top2 margin, "
    "cross-anchor aliasing, acceptance, confident-false — per level, per condition, vs chance "
    "1/n_references and per-source effective database size",
    "swipes: rank-1 vs total displacement / nearest-reference distance / per-direction distance / "
    "outbound-inbound; R90/R80 radii only where >= 30 attainable queries support the bin",
    "acceptance: precision / coverage / confident-false for the frozen rule AND accept-v2-margin, "
    "side by side, unchanged from their frozen definitions",
]

#: The pre-registered closure interpretation matrix (research R10), quoted verbatim by the CLI,
#: the prereg file and EXP-SKY-008.
FINAL_INTERPRETATION = {
    "criterion_1_extraction": (
        "SegFormer median abs error <= 5 px AND catastrophic-frame rate <= 10% per "
        "level x condition slice -> sufficient for that slice; else insufficient there."),
    "criterion_2_discriminability": (
        "exact-pose Recall@1 (GT and SegFormer, C0) >= 5x chance AND >= 60% at 27 and 30 "
        "references -> demonstrated; 30-60% -> partial; below -> not demonstrated."),
    "criterion_3_viewpoint_tolerance": (
        "R80 = the largest displacement radius with swipe Recall@1 >= 80% supported by >= 30 "
        "attainable queries per level (R90 likewise at 90%); no radius clears -> none claimed."),
    "criterion_4_safety": (
        "an acceptance rule (frozen or variant) with held-out accepted precision >= 90% and "
        "confident-false <= 5% at coverage >= 20% -> demonstrated; else not."),
    "criterion_5_domain_dependence": (
        "criteria 1-4 stated per level; a level failing while another passes -> CONDITIONALLY "
        "SUPPORTED / DOMAIN-LIMITED with the failing mechanism named."),
    "verdicts": (
        "SUPPORTED OPERATING REGIME iff 1-4 pass on both held-out levels; CONDITIONALLY SUPPORTED "
        "/ DOMAIN-LIMITED iff they pass on at least one level or in a nameable sub-regime; NOT "
        "SUPPORTED iff discriminability or extraction fails everywhere; NOT INTERPRETABLE iff a "
        "validity problem (GT inconsistency, leak, structural defect) blocks the reading. "
        "Domain-limited is closure-worthy; closure is never forced."),
}

FINAL_LIMITATIONS = (
    "naveval tier T2 (ue5_simulator), synthetic geodetic origin; pitch/roll signs declared but "
    "unverified (no R0-S render) and the quaternion convention non-discriminable on a level batch "
    "— both standing caveats ride every session; DEV = asian_village_hills_background (also the "
    "pilot's level), held-out = mountains + large_city_flat; database sizes 10/30/27; known "
    "world-locked-North heading; no yaw/attitude variation, no night/rain, no real-flight or "
    "onboard claim."
)


def build_final_prereg(config: dict, raw_digest: str, index_manifest: dict, sidecar_digests: dict,
                       task_manifests: dict, source_descriptions: dict) -> dict:
    """Assemble the freeze content from BUILT artifacts (nothing is restated by hand)."""
    if config.get("acceptance_variant", {}).get("frozen") is None:
        raise PreregError("acceptance_variant.frozen is null — run the DEV calibration (ext-accept) "
                          "and record the chosen thresholds before freezing")
    tasks_block = {}
    for name, m in sorted(task_manifests.items()):
        tasks_block[name] = {
            "task_digest": m["content_digest"],
            "grid_counts": m["grid_counts"],
            "tolerances": m["tolerances"],
            "n_query_rows": m["n_query_rows"],
            "reference_observation_ids_digest": _ids_digest(m.get("reference_observation_ids") or []),
        }
    return {
        "prereg_version": FINAL_PREREG_VERSION,
        "feature": "20260903-225416-sky-final-sim-validation",
        "experiment": "EXP-SKY-008",
        "raw": {"root": Path(config["raw_root"]).name, "digest": raw_digest},
        "index": {"content_digest": index_manifest["content_digest"],
                  "n_rows": index_manifest["n_rows"],
                  "split_rule": index_manifest["split_rule"],
                  "sidecar_digests": dict(sorted(sidecar_digests.items()))},
        "tasks": tasks_block,
        "matcher": {"block": config["matcher"], "baselines": list(config["baselines"]),
                    "primary_baseline": "ncc",
                    "successor": config.get("matcher_successor")},
        "acceptance": {"frozen_rule": config["matcher"]["acceptance"],
                       "variant": config["acceptance_variant"]},
        "run_matrix": config["run_matrix"],
        "geodetic_origin": config["geodetic_origin"],
        "sources": _source_pins(source_descriptions),
        "metrics": FINAL_METRICS,
        "interpretation": FINAL_INTERPRETATION,
        "limitations": FINAL_LIMITATIONS,
    }


def _ids_digest(ids: list) -> str:
    import hashlib
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def write_final_prereg(path: Path, built: dict) -> dict:
    pre = dict(built, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(pre, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pre


def check_final_prereg(path: Path, built: dict) -> dict:
    """Verbatim comparison, block by block, naming the first differing field."""
    p = Path(path)
    if not p.exists():
        raise PreregError(f"no freeze pre-registration at {p} — run `ext-prereg` and COMMIT it "
                          f"before any `--population final` command (contracts/freeze.md)")
    pre = json.loads(p.read_text(encoding="utf-8"))
    for field in ("prereg_version", "feature", "raw", "index", "tasks", "matcher", "acceptance",
                  "run_matrix", "geodetic_origin", "sources", "metrics", "interpretation",
                  "limitations"):
        if not _eq(pre.get(field), built.get(field)):
            raise PreregError(f"freeze field {field!r} differs from the committed pre-registration "
                              f"— a changed experiment is a NEW pre-registration (expected "
                              f"{json.dumps(pre.get(field), sort_keys=True)[:200]}…, got "
                              f"{json.dumps(built.get(field), sort_keys=True)[:200]}…)")
    return pre


def check_prereg(path: Path, config: dict, task_manifest: dict,
                 source_descriptions: dict) -> dict:
    p = Path(path)
    if not p.exists():
        raise PreregError(f"no pre-registration at {p} — run `prereg` and COMMIT it before `run` "
                          f"(the first final matching is a pre-registered event)")
    pre = json.loads(p.read_text(encoding="utf-8"))
    checks = [
        ("task digest", pre["task"]["task_digest"], task_manifest["content_digest"]),
        ("task name", pre["task"]["name"], config["task_name"]),
        ("spacings", pre["task"]["spacings_m"], [float(s) for s in config["tasks"]["spacings_m"]]),
        ("tolerances", pre["task"]["tolerances"], task_manifest["tolerances"]),
        ("reference ids", pre["task"]["reference_observation_ids"],
         list(config["tasks"]["reference_observation_ids"])),
        ("matcher block", pre["matcher"]["block"], config["matcher"]),
        ("baselines", pre["matcher"]["baselines"], list(config["baselines"])),
        ("geodetic origin", pre["geodetic_origin"], config["geodetic_origin"]),
        ("sources", pre["sources"], _source_pins(source_descriptions)),
    ]
    for name, want, got in checks:
        if not _eq(want, got):
            raise PreregError(f"{name} differs from the pre-registration: expected {want!r}, "
                              f"got {got!r} — a changed experiment is a NEW pre-registration")
    return pre
