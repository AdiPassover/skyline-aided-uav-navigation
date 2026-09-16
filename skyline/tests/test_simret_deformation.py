"""Known-answer tests for the swipe viewpoint-response measurement (Experiment 4)."""

import numpy as np
import pytest

from hsreloc.retrieval.skyline_curve import CurveError, make_curve
from hsreloc.simret import deformation

W = H = 512


class _Source:
    provenance = "oracle:sim_exact"

    def __init__(self, curves: dict):
        self.curves = curves

    def get(self, oid):
        if oid not in self.curves:
            raise CurveError(f"{oid}: no curve")
        return make_curve(oid, np.asarray(self.curves[oid], dtype=np.float64), W, H,
                          self.provenance, oid)


def _base_curve():
    x = np.arange(W, dtype=np.float64)
    return 200.0 + 40.0 * np.sin(2 * np.pi * x / 128.0)


def _rows(session="S", specs=()):
    """specs: (oid, leg_id, axis, direction, phase, lat, lon, up)"""
    rows = []
    for oid, leg, axis, direction, phase, lat, lon, up in specs:
        total = float(np.sqrt(lat * lat + lon * lon + up * up))
        rows.append({"session_id": session, "observation_id": oid, "kind": "swipe",
                     "leg_id": leg, "leg_axis": axis, "leg_direction": direction, "phase": phase,
                     "lateral_east_m": lat, "longitudinal_north_m": lon, "vertical_up_m": up,
                     "total_displacement_m": total, "hard_tag": False,
                     "time_of_day": "DAY", "clouds": "CLEAR"})
    return rows


class TestSwipeDeformation:
    def test_pure_shift_recovered_by_c1_and_absorbed_offset_by_vertical(self):
        base = _base_curve()
        curves = {
            "c": base,
            "shifted": np.roll(base, 8),                       # lateral-like horizontal shift
            "raised": base - 12.0,                             # vertical-like constant offset
        }
        rows = _rows(specs=(
            ("c", None, "", "", "center", 0.0, 0.0, 0.0),
            ("shifted", 0, "lateral", "east", "outbound", 30.0, 0.0, 0.0),
            ("raised", 1, "vertical", "up", "outbound", 0.0, 0.0, 30.0),
        ))
        from hsreloc.matchers import build_matcher
        res = deformation.swipe_deformation("S", rows, _Source(curves), H,
                                            matchers={"c1_bounded_lag_ncc":
                                                      build_matcher("c1_bounded_lag_ncc",
                                                                    {"max_lag_samples": 16})})
        by = {r["observation_id"]: r for r in res["rows"]}
        assert by["shifted"]["c1_bounded_lag_ncc_score"] > 0.99
        assert abs(by["shifted"]["c1_bounded_lag_ncc_shift"]) > 0
        # the constant offset is fully absorbed by offset removal
        assert by["raised"]["curve_median_abs_px"] == pytest.approx(12.0)
        assert by["raised"]["curve_median_abs_offset_removed_px"] == pytest.approx(0.0)
        assert by["raised"]["curve_offset_px"] == pytest.approx(-12.0)

    def test_content_change_fractions(self):
        base = _base_curve()
        replaced = base.copy()
        replaced[:52] = 400.0                                  # content entered at the left edge
        curves = {"c": base, "q": replaced}
        rows = _rows(specs=(("c", None, "", "", "center", 0.0, 0.0, 0.0),
                            ("q", 0, "lateral", "west", "outbound", -20.0, 0.0, 0.0)))
        res = deformation.swipe_deformation("S", rows, _Source(curves), H)
        r = res["rows"][0]
        assert r["edge_changed_frac"] > r["interior_changed_frac"]
        assert 0.05 < r["changed_frac"] < 0.2

    def test_refused_observation_recorded_not_dropped(self):
        rows = _rows(specs=(("c", None, "", "", "center", 0.0, 0.0, 0.0),
                            ("missing", 0, "lateral", "east", "outbound", 10.0, 0.0, 0.0)))
        res = deformation.swipe_deformation("S", rows, _Source({"c": _base_curve()}), H)
        assert res["refused"] and res["refused"][0]["observation_id"] == "missing"

    def test_final_leg_outbound_only_is_noted_not_flagged(self):
        base = _base_curve()
        curves = {f"o{i}": base for i in range(5)} | {"c": base}
        rows = _rows(specs=(
            ("c", None, "", "", "center", 0.0, 0.0, 0.0),
            ("o0", 0, "lateral", "east", "outbound", 10.0, 0.0, 0.0),
            ("o1", 0, "lateral", "east", "inbound", 9.0, 0.0, 0.0),
            ("o2", 1, "vertical", "down", "outbound", 0.0, 0.0, -10.0),
            ("o3", 1, "vertical", "down", "outbound", 0.0, 0.0, -20.0),
        ))
        res = deformation.swipe_deformation("S", rows, _Source(curves), H)
        assert "outbound-only by trajectory design" in res["final_leg_note"]


class TestRepeatability:
    def test_pairs_match_positions_within_gap(self):
        rows = [
            {"observation_id": "a", "leg_id": 0, "leg_axis": "lateral", "leg_direction": "east",
             "phase": "outbound", "abs_axis_displacement_m": 20.0},
            {"observation_id": "b", "leg_id": 0, "leg_axis": "lateral", "leg_direction": "east",
             "phase": "inbound", "abs_axis_displacement_m": 21.5},
            {"observation_id": "far", "leg_id": 0, "leg_axis": "lateral", "leg_direction": "east",
             "phase": "inbound", "abs_axis_displacement_m": 40.0},
        ]
        pairs = deformation._repeat_pairs(rows, max_gap_m=5.0)
        assert len(pairs) == 1
        assert (pairs[0]["outbound_id"], pairs[0]["return_id"]) == ("a", "b")
        assert pairs[0]["position_gap_m"] == pytest.approx(1.5)

    def test_repeatability_scores_identical_curves_at_one(self):
        base = _base_curve()
        src = _Source({"a": base, "b": base})
        rows = deformation.repeatability(
            [{"leg_id": 0, "leg_axis": "lateral", "leg_direction": "east",
              "outbound_id": "a", "return_id": "b", "abs_axis_displacement_m": 10.0,
              "position_gap_m": 0.5}], {"sim_exact": src}, H)
        assert rows[0]["c0_ncc"] == pytest.approx(1.0)
        assert rows[0]["curve_median_abs_px"] == 0.0
