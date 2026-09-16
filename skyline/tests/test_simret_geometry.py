"""Known-answer tests for the pose-only viewpoint geometry (PROT-SKY-001 §7, T1 synthetic).

Every pose here is constructed; nothing reads an image, a curve, or a simulator file. The module
under test consumes only the frozen spec-006 observation record, so these tests are valid for any
future adapter that produces one — they do not guess the simulator's export format.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from hsreloc.observation import Observation
from hsreloc.simret import geometry as g

FORBIDDEN_MODULES = ("hsreloc.extraction", "hsreloc.retrieval", "hsreloc.placeret.sources",
                     "hsreloc.placeret.sets", "cv2", "PIL", "torch", "transformers")


def obs(oid: str, session: str, fi: int, e: float, n: float, yaw, pitch=0.0, roll=0.0, up=60.0) -> Observation:
    return Observation(observation_id=oid, session_id=session, frame_index=fi, timestamp_s=float(fi),
                       image_path=f"images/{oid}.png", image_width_px=640, image_height_px=512,
                       gt_source="sim_exact", pos_east_m=e, pos_north_m=n, up_m=up,
                       yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)


# --- angles -----------------------------------------------------------------------------------------

def test_wrap_and_yaw_difference_cross_the_zero_360_boundary():
    assert g.wrap_deg(190.0) == -170.0
    assert g.wrap_deg(-180.0) == -180.0
    assert g.yaw_diff_deg(10.0, 350.0) == 20.0
    assert g.yaw_diff_deg(350.0, 10.0) == -20.0
    assert g.yaw_diff_deg(0.0, 180.0) == -180.0


def test_heading_vectors_follow_dec_004_compass_convention():
    fwd, right = g.heading_vectors(0.0)          # North
    assert np.allclose(fwd, [0, 1]) and np.allclose(right, [1, 0])
    fwd, right = g.heading_vectors(90.0)         # East
    assert np.allclose(fwd, [1, 0]) and np.allclose(right, [0, -1])


# --- viewpoint offset ---------------------------------------------------------------------------------

def test_offset_decomposes_translation_in_the_reference_heading_frame():
    ref = obs("r", "ref", 0, 0.0, 0.0, yaw=0.0)
    q = obs("q", "qry", 0, 3.0, 4.0, yaw=0.0)
    off = g.viewpoint_offset(q, ref)
    assert off.translation_m == pytest.approx(5.0)
    assert off.along_m == pytest.approx(4.0) and off.lateral_m == pytest.approx(3.0)
    # the same displacement seen from an East-facing reference: 3 m ahead, 4 m to the LEFT
    ref_e = obs("r", "ref", 0, 0.0, 0.0, yaw=90.0)
    off_e = g.viewpoint_offset(q, ref_e)
    assert off_e.along_m == pytest.approx(3.0) and off_e.lateral_m == pytest.approx(-4.0)
    assert off_e.yaw_diff_deg == pytest.approx(-90.0)


def test_offset_carries_attitude_and_altitude_differences_or_none():
    ref = obs("r", "ref", 0, 0.0, 0.0, yaw=0.0, pitch=0.0, roll=0.0, up=60.0)
    q = obs("q", "qry", 0, 0.0, 0.0, yaw=0.0, pitch=5.0, roll=-2.0, up=63.0)
    off = g.viewpoint_offset(q, ref)
    assert off.pitch_diff_deg == 5.0 and off.roll_diff_deg == -2.0 and off.up_m == pytest.approx(3.0)
    q2 = obs("q2", "qry", 1, 0.0, 0.0, yaw=0.0, pitch=None, roll=None, up=None)
    off2 = g.viewpoint_offset(q2, ref)
    assert off2.pitch_diff_deg is None and off2.roll_diff_deg is None and off2.up_m is None


def test_offset_refuses_a_pose_without_yaw_or_position():
    ref = obs("r", "ref", 0, 0.0, 0.0, yaw=0.0)
    with pytest.raises(g.GeometryError, match="no yaw"):
        g.viewpoint_offset(obs("q", "qry", 0, 1.0, 1.0, yaw=None), ref)
    no_pos = Observation(observation_id="p", session_id="qry", frame_index=0, timestamp_s=0.0,
                         image_path="images/p.png", image_width_px=640, image_height_px=512,
                         gt_source="sim_exact", lat=1.0, lon=2.0, yaw_deg=0.0)
    with pytest.raises(g.GeometryError, match="no ENU position"):
        g.viewpoint_offset(no_pos, ref)


# --- bins and stages ----------------------------------------------------------------------------------

def test_bins_are_half_open_and_labelled_by_their_edges():
    edges = g.DEFAULT_TRANSLATION_BINS_M
    assert g.assign_bin(0.0, edges) == "0-1"
    assert g.assign_bin(5.0, edges) == "1-7.5"
    assert g.assign_bin(7.5, edges) == "7.5-17.5"
    assert g.assign_bin(149.99, edges) == "75-150"
    assert g.assign_bin(150.0, edges) is None
    with pytest.raises(g.GeometryError):
        g.assign_bin(1.0, [5.0, 1.0])


def test_stage_rule_isolates_one_geometric_difficulty_at_a_time():
    rule = g.StageRule()
    ref = obs("r", "ref", 0, 0.0, 0.0, yaw=0.0)
    same = obs("s", "cond", 0, 0.1, 0.0, yaw=0.0)
    trans = obs("t", "qry", 0, 10.0, 0.0, yaw=0.2)
    yawed = obs("y", "qry", 1, 10.0, 0.0, yaw=5.0)
    pitched = obs("p", "qry", 2, 0.0, 0.0, yaw=0.0, pitch=3.0)
    assert rule.stage_for(g.viewpoint_offset(same, ref)) == "stage0"
    assert rule.stage_for(g.viewpoint_offset(trans, ref)) == "stage1"
    assert rule.stage_for(g.viewpoint_offset(yawed, ref)) == "stage2"
    assert rule.stage_for(g.viewpoint_offset(pitched, ref)) == "stage3"
    unknown = obs("u", "qry", 3, 0.0, 0.0, yaw=0.0, pitch=None, roll=None)
    with pytest.raises(g.GeometryError, match="pitch/roll missing"):
        rule.stage_for(g.viewpoint_offset(unknown, ref))
    assert g.StageRule(require_attitude=False).stage_for(g.viewpoint_offset(unknown, ref)) == "attitude-unknown"


# --- far-field prediction ---------------------------------------------------------------------------

def test_far_field_prediction_is_first_order_in_t_over_range():
    ref = obs("r", "ref", 0, 0.0, 0.0, yaw=0.0)
    lateral = obs("l", "qry", 0, 10.0, 0.0, yaw=0.0)      # 10 m to the right of a North-facing camera
    p = g.far_field_prediction(g.viewpoint_offset(lateral, ref), range_m=1000.0)
    assert p["shift_deg"] == pytest.approx(-np.degrees(0.01))
    assert p["scale"] == pytest.approx(1.0)
    ahead = obs("a", "qry", 1, 0.0, 10.0, yaw=0.0)         # 10 m ahead
    p2 = g.far_field_prediction(g.viewpoint_offset(ahead, ref), range_m=1000.0)
    assert p2["shift_deg"] == pytest.approx(0.0) and p2["scale"] == pytest.approx(1.01)
    assert p2["t_over_range"] == pytest.approx(0.01)
    with pytest.raises(g.GeometryError):
        g.far_field_prediction(g.viewpoint_offset(ahead, ref), range_m=0.0)


# --- task sets ----------------------------------------------------------------------------------------

@pytest.fixture
def straight_line_sessions():
    """A reference flight North along x = 0 at 1 m steps; a parallel query flight 2 m East (yaw 0);
    a yawed query flight (yaw 2°); one far-away query; a same-pose condition session."""
    ref = [obs(f"ref_{i:03d}", "ref", i, 0.0, float(i), yaw=0.0) for i in range(60)]
    par = [obs(f"par_{i:03d}", "par", i, 2.0, float(i) + 0.5, yaw=0.0) for i in range(0, 60, 3)]
    yawed = [obs(f"yaw_{i:03d}", "yawed", i, 0.0, float(i) + 0.5, yaw=2.0) for i in range(0, 60, 5)]
    far = [obs("far_000", "far", 0, 100.0, 30.0, yaw=0.0)]
    cond = [obs(f"cond_{i:03d}", "cond", i, 0.0, float(i), yaw=0.0) for i in (0, 10, 20)]
    return {"ref": ref, "par": par, "yawed": yawed, "far": far, "cond": cond}


def _config(**over):
    cfg = {"reference_session": "ref", "query_sessions": ["par", "yawed", "far", "cond"],
           "spacings_m": [10.0, 25.0]}
    cfg.update(over)
    return cfg


def test_grid_respects_spacing_and_correctness_is_pose_defined(straight_line_sessions):
    tasks = g.build_tasks(straight_line_sessions, _config())
    g10 = tasks["grids"]["10"]
    assert [r["north_m"] for r in g10["references"]] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert g10["tau_pos_m"] == 5.0 and g10["tau_near_m"] == 10.0
    assert tasks["grids"]["25"]["n_references"] == 3          # 0, 25, 50
    q = [r for r in tasks["queries"] if r["spacing_m"] == 10.0 and r["session_id"] == "par"]
    # the grid ends at north 50 (the pool ends at 59), so par_057 (north 57.5) is 7.8 m from its
    # nearest reference — outside tau_pos and therefore honestly out of coverage; the other 19 are in
    assert len(q) == 20 and sum(r["in_coverage"] for r in q) == 19
    assert not next(r for r in q if r["observation_id"] == "par_057")["in_coverage"]
    # par_009 is at north 9.5 -> nearest reference is ref_010 (0.5 m along), 2 m lateral
    r9 = next(r for r in q if r["observation_id"] == "par_009")
    assert r9["nearest_reference_id"] == "ref_010"
    assert r9["condition_along_m"] == pytest.approx(-0.5) and r9["condition_lateral_m"] == pytest.approx(2.0)
    assert r9["condition_translation_bin"] == "1-7.5" and r9["condition_stage"] == "stage1"
    far = next(r for r in tasks["queries"] if r["session_id"] == "far" and r["spacing_m"] == 10.0)
    assert not far["in_coverage"] and far["condition_translation_bin"] == "75-150"


def test_stages_and_bins_are_assigned_from_poses(straight_line_sessions):
    tasks = g.build_tasks(straight_line_sessions, _config())
    rows = [r for r in tasks["queries"] if r["spacing_m"] == 10.0]
    yawed = [r for r in rows if r["session_id"] == "yawed"]
    assert yawed and all(r["condition_stage"] == "stage2" and r["condition_yaw_bin"] == "1.5-3.5" for r in yawed)
    cond = [r for r in rows if r["session_id"] == "cond"]
    assert len(cond) == 3 and all(r["condition_stage"] == "stage0" and r["condition_translation_m"] == 0.0
                                  for r in cond)
    assert all(r["condition_n_references"] == 6 for r in rows)
    assert tasks["manifest"]["stage_counts"]["10|stage0"] == 3


def test_grid_frames_are_never_their_own_queries_and_the_reference_session_may_be_queried(straight_line_sessions):
    tasks = g.build_tasks(straight_line_sessions, _config(query_sessions=["ref"], spacings_m=[10.0]))
    ids = {r["observation_id"] for r in tasks["queries"]}
    assert not ids & {r["reference_id"] for r in tasks["grids"]["10"]["references"]}
    assert len(ids) == 60 - 6


def test_build_refuses_undeclared_sessions_and_bad_spacings(straight_line_sessions):
    with pytest.raises(g.GeometryError, match="reference session"):
        g.build_tasks(straight_line_sessions, _config(reference_session="nope"))
    with pytest.raises(g.GeometryError, match="query sessions not found"):
        g.build_tasks(straight_line_sessions, _config(query_sessions=["par", "ghost"]))
    with pytest.raises(g.GeometryError, match="positive"):
        g.build_tasks(straight_line_sessions, _config(spacings_m=[0.0]))


def test_explicit_reference_observations_span_sessions_and_skip_subsampling(straight_line_sessions):
    """The 2026-09-02 pilot case: declared references from several sessions (research R4)."""
    cfg = _config(reference_session=None,
                  reference_observation_ids=["ref_000", "ref_030", "cond_020"],
                  spacings_m=[10.0])
    cfg.pop("reference_session")
    tasks = g.build_tasks(straight_line_sessions, cfg)
    grid = tasks["grids"]["10"]
    assert [r["reference_id"] for r in grid["references"]] == ["ref_000", "ref_030", "cond_020"]
    assert {r["session_id"] for r in grid["references"]} == {"ref", "cond"}
    # cond_020 sits at north 20 -> a query at north 21 is nearest to it, pose-defined
    q = next(r for r in tasks["queries"] if r["observation_id"] == "par_021")
    assert q["nearest_reference_id"] == "cond_020"
    # the declared references are never their own queries
    ids = {r["observation_id"] for r in tasks["queries"]}
    assert not ids & {"ref_000", "ref_030", "cond_020"}


def test_explicit_references_are_validated_not_trusted(straight_line_sessions):
    base = {"query_sessions": ["par"], "spacings_m": [10.0]}
    with pytest.raises(g.GeometryError, match="exactly one of"):
        g.build_tasks(straight_line_sessions, dict(base))
    with pytest.raises(g.GeometryError, match="exactly one of"):
        g.build_tasks(straight_line_sessions, dict(base, reference_session="ref",
                                                   reference_observation_ids=["ref_000"]))
    with pytest.raises(g.GeometryError, match="not found in any session"):
        g.build_tasks(straight_line_sessions, dict(base, reference_observation_ids=["ghost"]))
    with pytest.raises(g.GeometryError, match="duplicates"):
        g.build_tasks(straight_line_sessions,
                      dict(base, reference_observation_ids=["ref_000", "ref_000"]))
    # the min-pairwise-distance >= spacing assertion still guards the positive class
    with pytest.raises(g.GeometryError, match="positive class"):
        g.build_tasks(straight_line_sessions,
                      dict(base, reference_observation_ids=["ref_000", "ref_003"]))


def test_a_vertical_only_pair_is_a_translation_not_a_same_pose():
    """Research R5: two of the pilot's sweeps are vertical; stage0 would poison every slice."""
    rule = g.StageRule()
    ref = obs("r", "s", 0, 0.0, 0.0, yaw=0.0, up=60.0)
    high = obs("q", "s", 1, 0.0, 0.0, yaw=0.0, up=80.0)
    near = obs("q2", "s", 2, 0.0, 0.0, yaw=0.0, up=60.2)
    assert rule.stage_for(g.viewpoint_offset(high, ref)) == "stage1"
    assert rule.stage_for(g.viewpoint_offset(near, ref)) == "stage0"


def test_write_load_round_trip_is_deterministic(straight_line_sessions, tmp_path):
    tasks = g.build_tasks(straight_line_sessions, _config())
    m1 = g.write_tasks(tasks, tmp_path / "a")
    m2 = g.write_tasks(g.build_tasks(straight_line_sessions, _config()), tmp_path / "b")
    assert m1["content_digest"] == m2["content_digest"]
    back = g.load_tasks(tmp_path / "a")
    assert len(back["queries"]) == len(tasks["queries"])
    orig = {(r["spacing_m"], r["observation_id"]): r for r in tasks["queries"]}
    for r in back["queries"]:
        o = orig[(r["spacing_m"], r["observation_id"])]
        assert r["nearest_reference_id"] == o["nearest_reference_id"]
        assert r["in_coverage"] == o["in_coverage"]
        assert r["condition_stage"] == o["condition_stage"]
        assert r["condition_translation_m"] == pytest.approx(o["condition_translation_m"], abs=1e-4)
    assert back["manifest"]["geometry_version"] == g.GEOMETRY_VERSION


# --- leakage ------------------------------------------------------------------------------------------

def _imports_of(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_leakage_geometry_imports_no_extractor_matcher_or_image_library():
    names = _imports_of(Path(g.__file__))
    for forbidden in FORBIDDEN_MODULES:
        assert not any(n == forbidden or n.startswith(forbidden + ".") for n in names), forbidden
