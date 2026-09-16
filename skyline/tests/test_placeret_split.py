"""Pose-only grids, tasks, attainability and the leakage guarantee (spec FR-001..FR-006)."""

from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from hsreloc.placeret import split as sp
from tests import placeret_fixtures as fx

FORBIDDEN_MODULES = ("hsreloc.extraction.methods", "hsreloc.extraction.silver", "hsreloc.retrieval",
                     "hsreloc.extraction.consistency", "cv2", "PIL", "torch", "transformers")


@pytest.fixture
def index(tmp_path):
    return fx.make_index(tmp_path / "index")


def test_grid_respects_spacing_and_uses_eligible_originals_only(index, tmp_path):
    m = sp.build_split(index, fx.split_config(), tmp_path / "split")
    grids = json.loads((tmp_path / "split" / "grids.json").read_text())
    for s, g in grids.items():
        P = np.array([[r["east_m"], r["north_m"]] for r in g["references"]])
        d = np.linalg.norm(P[:, None] - P[None, :], axis=2)
        np.fill_diagonal(d, np.inf)
        assert d.min() >= float(s) - 1e-9
        assert all(r["sequence"] == "seq1" for r in g["references"])
        assert all(r["reference_id"].endswith("__original") for r in g["references"])
        assert g["tau_pos_m"] == float(s) / 2 and g["tau_near_m"] == float(s)
    assert m["population"] == "all"


def test_families_are_built_from_the_declared_sequences(index, tmp_path):
    sp.build_split(index, fx.split_config(), tmp_path / "split")
    q = sp.load_split(tmp_path / "split")["queries"]
    grids = json.loads((tmp_path / "split" / "grids.json").read_text())
    ref_groups_by_spacing = {float(s): {r["group_id"] for r in g["references"]} for s, g in grids.items()}
    ref_groups = set.union(*ref_groups_by_spacing.values())
    cross = [x for x in q if x["family"].startswith("cross")]
    assert cross and all(x["sequence"] == "seq2" and x["scene"] == "SceneA" for x in cross)
    assert all(x["condition"] == "original" for x in q if x["family"] == "cross-orig")
    assert all(x["condition"] != "original" for x in q if x["family"] == "cross-gen")
    control = [x for x in q if x["family"] == "control"]
    assert control and all(x["group_id"] in ref_groups_by_spacing[x["spacing_m"]] and x["condition"] != "original"
                           for x in control)
    same = [x for x in q if x["family"].startswith("same")]
    assert same and all(x["sequence"] == "seq1" and x["group_id"] not in ref_groups_by_spacing[x["spacing_m"]]
                        for x in same)
    # a group lacking Evening yields no evening query
    assert not any(x["condition"] == "evening" and x["group_id"] == "SceneB__seq1__frame00004" for x in q)
    # tiers by family
    assert {x["tier"] for x in q if x["family"] in ("cross-orig", "same-orig")} == {"T3"}
    assert {x["tier"] for x in q if x["family"] in ("control", "cross-gen", "same-gen")} == {"T2"}


def test_attainability_and_uniqueness(index, tmp_path):
    sp.build_split(index, fx.split_config(), tmp_path / "split")
    s = sp.load_split(tmp_path / "split")
    for x in s["queries"]:
        tau = s["grids"][f"{x['spacing_m']:g}" if f"{x['spacing_m']:g}" in s["grids"] else str(x["spacing_m"])]["tau_pos_m"]
        assert x["attainable"] == (x["nearest_ref_m"] <= tau)
        assert x["n_scene_references"] >= 1
    # the reference frame's own variants are at distance 0 -> attainable, rotation 0
    ctrl = [x for x in s["queries"] if x["family"] == "control"]
    assert all(x["nearest_ref_m"] < 1e-6 and x["rotation_deg"] < 1e-6 and x["attainable"] for x in ctrl)


def test_dev_population_builds_identity_and_control_only(index, tmp_path):
    m = sp.build_split(index, fx.split_config("dev"), tmp_path / "dev")
    q = sp.load_split(tmp_path / "dev")["queries"]
    assert set(x["family"] for x in q) <= {"identity", "control"}
    assert all(x["dev_subset"] for x in q)
    ident = [x for x in q if x["family"] == "identity"]
    assert ident and all(x["condition"] == "original" and x["nearest_ref_m"] < 1e-6 for x in ident)
    assert m["families"] == ["identity", "control"]


def test_digest_is_deterministic_and_100m_is_refused(index, tmp_path):
    a = sp.build_split(index, fx.split_config(), tmp_path / "a")["content_digest"]
    b = sp.build_split(index, fx.split_config(), tmp_path / "b")["content_digest"]
    assert a == b
    cfg = fx.split_config()
    cfg["spacings_m"] = [100.0]
    with pytest.raises(sp.SplitError):
        sp.build_split(index, cfg, tmp_path / "c")


def test_index_revision_guard(index, tmp_path):
    cfg = fx.split_config()
    cfg["expected_index_revision"] = "something-else"
    with pytest.raises(sp.SplitError):
        sp.build_split(index, cfg, tmp_path / "x")


def _imports_of(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_leakage_the_design_modules_import_no_extractor_matcher_or_image_library():
    for mod in ("split.py", "frame.py"):
        names = _imports_of(Path(sp.__file__).with_name(mod))
        for forbidden in FORBIDDEN_MODULES:
            assert not any(n == forbidden or n.startswith(forbidden + ".") for n in names), (mod, forbidden)


def test_leakage_the_builder_opens_only_index_files_and_its_outputs(index, tmp_path, monkeypatch):
    out = tmp_path / "split"
    opened = []
    real_open = builtins.open

    def spy(file, *a, **k):
        opened.append(Path(str(file)).resolve())
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    sp.build_split(index, fx.split_config(), out)
    allowed = {Path(index).resolve(), out.resolve()}
    for p in opened:
        assert any(str(p).startswith(str(a)) for a in allowed), p
