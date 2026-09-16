"""Star-swipe viewpoint-response measurement (Experiment 4 of the final-validation feature).

Every swipe observation's curve is compared to the **swipe's own center curve** — not to any
reference database — separately per leg (lateral / longitudinal / vertical, signed), with the
frozen C0 primitive plus the C1/C3 *diagnostic* alignments as measurement instruments (their
bounds are declared config values; nothing here tunes them). The final leg of every swipe carries
only outbound samples because the star trajectory intentionally omits the last return-to-center
segment (owner correction, 2026-09-03): it appears in the response tables normally and is simply
absent from the outbound/return pairing, which is reported per swipe rather than flagged.

Outbound/return repeatability: within a leg, each outbound observation is paired with the return
observation whose along-axis displacement is nearest (within ``repeat_pair_max_gap_m``); the pair
is compared per source (GT, and optionally the automatic sources) and scored with frozen C0 NCC —
these are the repeatability controls the owner brief asks to exploit, never discarded duplicates.
"""

from __future__ import annotations

import numpy as np

from hsreloc.matchers import FrozenNccMatcher
from hsreloc.retrieval.profile import ProfileConfig, normalize
from hsreloc.retrieval.skyline_curve import CurveError
from hsreloc.simret.geometry import assign_bin
from hsreloc.simret.report import ReportError, curve_disagreement

DEFORMATION_VERSION = "1.0.0"

#: Declared displacement bins (the pilot's, so numbers sit on one axis across experiments).
DISPLACEMENT_BINS_M = [0.0, 1.0, 7.5, 17.5, 37.5, 75.0, 150.0]

#: A column "changed" when the GT boundary moved more than this fraction of image height —
#: the content-change proxy (FOV entry/exit and near-field replacement both move it a lot).
CONTENT_CHANGE_FRAC = 0.05
#: The outer band (per side) whose changes are counted separately: content enters/leaves there.
EDGE_BAND_FRAC = 0.1


def _content_change(center_rows, q_rows, height_px: int) -> dict:
    c = np.asarray(center_rows, dtype=np.float64)
    q = np.asarray(q_rows, dtype=np.float64)
    changed = np.abs(q - c) > CONTENT_CHANGE_FRAC * float(height_px)
    band = max(1, int(round(EDGE_BAND_FRAC * len(c))))
    edges = np.zeros(len(c), dtype=bool)
    edges[:band] = True
    edges[-band:] = True
    return {
        "changed_frac": float(changed.mean()),
        "edge_changed_frac": float(changed[edges].mean()),
        "interior_changed_frac": float(changed[~edges].mean()),
    }


def swipe_deformation(session_id: str, index_rows: list, source, height_px: int,
                      matchers: dict | None = None) -> dict:
    """Per-observation response of one swipe relative to its own center observation."""
    rows = [r for r in index_rows if r["session_id"] == session_id and r["kind"] == "swipe"]
    if not rows:
        raise ReportError(f"{session_id}: no swipe rows in the index")
    center_row = min(rows, key=lambda r: (r["total_displacement_m"], r["observation_id"]))
    matchers = dict(matchers or {})
    matchers.setdefault("c0_frozen_ncc", FrozenNccMatcher())
    profile_config = ProfileConfig()
    try:
        center_curve = source.get(center_row["observation_id"])
    except CurveError as exc:
        return {"session_id": session_id, "center": center_row["observation_id"],
                "center_refused": str(exc)[:200], "rows": [], "refused": [], "pairs": []}
    center_profile = normalize(center_curve, profile_config)

    out, refused = [], []
    for r in sorted(rows, key=lambda r: r["observation_id"]):
        if r["observation_id"] == center_row["observation_id"]:
            continue
        try:
            qc = source.get(r["observation_id"])
        except CurveError as exc:
            refused.append({"observation_id": r["observation_id"], "reason": str(exc)[:200]})
            continue
        d = curve_disagreement(center_curve.row_per_col, qc.row_per_col, height_px)
        axis_value = {"lateral": r["lateral_east_m"], "longitudinal": r["longitudinal_north_m"],
                      "vertical": r["vertical_up_m"]}.get(r["leg_axis"])
        row = {"session_id": session_id, "observation_id": r["observation_id"],
               "leg_id": r["leg_id"], "leg_axis": r["leg_axis"],
               "leg_direction": r["leg_direction"], "phase": r["phase"],
               "lateral_east_m": r["lateral_east_m"],
               "longitudinal_north_m": r["longitudinal_north_m"],
               "vertical_up_m": r["vertical_up_m"],
               "total_displacement_m": r["total_displacement_m"],
               "signed_axis_displacement_m": axis_value,
               "abs_axis_displacement_m": None if axis_value is None else abs(float(axis_value)),
               "displacement_bin": assign_bin(r["total_displacement_m"], DISPLACEMENT_BINS_M),
               "hard_tag": r["hard_tag"], "time_of_day": r["time_of_day"], "clouds": r["clouds"],
               **{f"curve_{k}": v for k, v in d.items()},
               **_content_change(center_curve.row_per_col, qc.row_per_col, height_px)}
        qp = normalize(qc, profile_config)
        for name, m in matchers.items():
            res = m.match(qp, center_profile)
            row[f"{name}_score"] = res.score
            row[f"{name}_shift"] = res.shift
            row[f"{name}_scale"] = res.scale
        out.append(row)

    return {"session_id": session_id, "center": center_row["observation_id"],
            "center_refused": None, "rows": out, "refused": refused,
            "pairs": _repeat_pairs(out),
            "final_leg_note": _final_leg_note(rows)}


def _final_leg_note(rows: list) -> str:
    legs = sorted({r["leg_id"] for r in rows if r["leg_id"] is not None})
    if not legs:
        return ""
    last = legs[-1]
    phases = {r["phase"] for r in rows if r["leg_id"] == last}
    if "inbound" not in phases:
        return (f"leg {last}: outbound-only by trajectory design (the star intentionally omits "
                f"the final return-to-center segment; owner correction 2026-09-03) — no matching "
                f"return samples exist for it")
    return ""


def _repeat_pairs(rows: list, max_gap_m: float = 5.0) -> list:
    """(outbound, return) index pairs of matching along-axis displacement, per leg."""
    pairs = []
    by_leg: dict = {}
    for r in rows:
        if r["leg_id"] is not None and r["abs_axis_displacement_m"] is not None:
            by_leg.setdefault(r["leg_id"], []).append(r)
    for leg, members in sorted(by_leg.items()):
        outb = [m for m in members if m["phase"] == "outbound"]
        inb = [m for m in members if m["phase"] == "inbound"]
        used = set()
        for o in sorted(outb, key=lambda m: m["abs_axis_displacement_m"]):
            best, best_gap = None, None
            for i in inb:
                if i["observation_id"] in used:
                    continue
                gap = abs(i["abs_axis_displacement_m"] - o["abs_axis_displacement_m"])
                if gap <= max_gap_m and (best_gap is None or gap < best_gap):
                    best, best_gap = i, gap
            if best is not None:
                used.add(best["observation_id"])
                pairs.append({"leg_id": leg, "leg_axis": o["leg_axis"],
                              "leg_direction": o["leg_direction"],
                              "outbound_id": o["observation_id"], "return_id": best["observation_id"],
                              "abs_axis_displacement_m": o["abs_axis_displacement_m"],
                              "position_gap_m": best_gap})
    return pairs


def repeatability(pairs: list, sources: dict, height_px: int) -> list:
    """Compare each (outbound, return) pair per source: curve disagreement + frozen C0 NCC."""
    ncc = FrozenNccMatcher()
    profile_config = ProfileConfig()
    rows = []
    for p in pairs:
        for key, source in sources.items():
            row = dict(p, source=key, status="ok")
            try:
                a = source.get(p["outbound_id"])
                b = source.get(p["return_id"])
            except CurveError as exc:
                row.update(status="refused", refusal=str(exc)[:200])
                rows.append(row)
                continue
            d = curve_disagreement(a.row_per_col, b.row_per_col, height_px)
            row.update({f"curve_{k}": v for k, v in d.items()})
            row["c0_ncc"] = ncc.match(normalize(a, profile_config),
                                      normalize(b, profile_config)).score
            rows.append(row)
    return rows
