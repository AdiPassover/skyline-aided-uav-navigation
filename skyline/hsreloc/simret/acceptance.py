"""The one authorized small/medium-database acceptance variant: ``accept-v2-margin``.

``EXP-SKY-007`` showed the frozen Nordland-derived acceptance rule collapses on small simulator
databases (it rejects NCC ≈ 1.0 identity matches). This variant is a **post-hoc, record-level
decision layer**: accept a query iff ``top1_score >= theta_s`` and
``top1_score - top2_score >= theta_m``, computed from the result record's own candidate scores.
The frozen matcher, its in-record rule and every record file are untouched — which is also what
makes the variant retroactively computable on the pilot's records.

Discipline (research R8): the (theta_s, theta_m) grid and the lexicographic selection rule are
declared in the config **before** calibration; calibration reads DEV evidence only (pilot village
records + this feature's DEV records); the chosen thresholds go verbatim into the config and the
freeze prereg; the held-out application runs both rules side by side, unchanged.

Confident-false is counted STRICTLY: an accepted query is a confident-false unless it is
attainable AND its top-1 is the pose-correct reference — accepting an unattainable query claims a
fix where none is valid, which is exactly the safety failure the criterion exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

ACCEPTANCE_VERSION = "1.0.0"
VARIANT_NAME = "accept-v2-margin"


class AcceptanceError(Exception):
    """The variant cannot be calibrated or applied as asked."""


def decide(row: dict, theta_s: float, theta_m: float) -> bool:
    """The whole rule. ``row`` is an extreport rank row (or anything with the two scores)."""
    s, m = row.get("best_score"), row.get("score_margin")
    if s is None:
        return False
    if m is None:
        m = (s - row["second_best_score"]) if row.get("second_best_score") is not None else None
    if m is None:
        return False
    return s >= theta_s and m >= theta_m


def evaluate_cell(rows: list, theta_s: float, theta_m: float) -> dict:
    """Precision / coverage / strict confident-false of one grid cell over rank rows."""
    return {"theta_s": theta_s, "theta_m": theta_m,
            **count_flags(rows, [decide(r, theta_s, theta_m) for r in rows])}


def calibrate(rows: list, grid: dict) -> dict:
    """The declared lexicographic selection over the declared grid, on DEV rows only.

    1. among cells with ZERO strict confident-false: max ``n_accepted_correct``;
       ties -> larger theta_m, then larger theta_s (safer);
    2. if no cell has zero: min confident-false rate, then max coverage, then larger theta_m.
    """
    if not rows:
        raise AcceptanceError("no DEV rows to calibrate on")
    cells = [evaluate_cell(rows, float(s), float(m))
             for s in grid["theta_s"] for m in grid["theta_m"]]
    zero = [c for c in cells if c["n_confident_false_strict"] == 0]
    if zero:
        chosen = max(zero, key=lambda c: (c["n_accepted_correct"], c["theta_m"], c["theta_s"]))
        rule_path = "zero-confident-false branch"
    else:
        chosen = min(cells, key=lambda c: (c["confident_false_rate"],
                                           -(c["coverage_of_attainable"] or 0.0), -c["theta_m"]))
        rule_path = "min-confident-false branch (no zero cell existed)"
    return {"acceptance_version": ACCEPTANCE_VERSION, "variant": VARIANT_NAME,
            "grid": grid, "cells": cells, "chosen": chosen, "rule_path": rule_path}


def apply(rows: list, frozen: dict) -> list:
    """Overlay the frozen variant decision on rank rows; returns new rows, records untouched."""
    theta_s, theta_m = float(frozen["theta_s"]), float(frozen["theta_m"])
    out = []
    for r in rows:
        out.append(dict(r, variant_accepted=decide(r, theta_s, theta_m)))
    return out


def count_flags(rows: list, accepted_flags: list) -> dict:
    """Precision / coverage / strict confident-false for an arbitrary accept decision per row."""
    if len(rows) != len(accepted_flags):
        raise AcceptanceError("one accept flag per row, please")
    att = [r for r in rows if r["attainable"]]
    accepted = [r for r, a in zip(rows, accepted_flags) if a]
    acc_att = [r for r in accepted if r["attainable"]]
    correct = [r for r in acc_att if r["top1_correct"]]
    conf_false = [r for r in accepted if not (r["attainable"] and r["top1_correct"])]
    return {
        "n_rows": len(rows), "n_attainable": len(att), "n_accepted": len(accepted),
        "n_accepted_attainable": len(acc_att), "n_accepted_correct": len(correct),
        "n_confident_false_strict": len(conf_false),
        "confident_false_rate": (len(conf_false) / len(rows)) if rows else None,
        "accepted_precision": (len(correct) / len(acc_att)) if acc_att else None,
        "coverage_of_attainable": (len(acc_att) / len(att)) if att else None,
        "correct_coverage_of_attainable": (len(correct) / len(att)) if att else None,
    }


def side_by_side(rows: list, frozen: dict) -> dict:
    """The held-out comparison: frozen in-record rule vs the variant, same rows, same counting."""
    theta_s, theta_m = float(frozen["theta_s"]), float(frozen["theta_m"])
    return {
        "frozen_rule": {"rule": "frozen in-record acceptance",
                        **count_flags(rows, [r["accepted"] for r in rows])},
        "variant": {"rule": f"{VARIANT_NAME} (theta_s={theta_s}, theta_m={theta_m})",
                    **count_flags(rows, [decide(r, theta_s, theta_m) for r in rows])},
    }


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
