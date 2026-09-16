"""The two frozen sources behind the seam (spec US2; research R4, R5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hsreloc.placeret import sources as src
from hsreloc.retrieval.skyline_curve import CurveError
from tests import placeret_fixtures as fx


@pytest.fixture
def world(tmp_path):
    index = fx.make_index(tmp_path / "index")
    store = fx.make_store(tmp_path / "store", index)
    maps = fx.make_label_maps(tmp_path / "maps", index,
                              statuses={("SceneA__seq1__frame00002", "winter"): "insufficient_sky",
                                        ("SceneA__seq1__frame00003", "evening"): "no_top_connected_sky"})
    return {"index": index, "store": store, "maps": maps}


def test_dp_source_returns_store_bytes_with_its_provenance(world):
    s = src.ConditionedDPSource(world["store"], fx.H, fx.W)
    c = s.get("SceneA__seq1__frame00001__summer")
    assert c.provenance == "automatic:poc_robust_dp"
    assert c.observation_id == "SceneA__seq1__frame00001__summer"
    assert np.allclose(c.row_per_col, fx.place_curve("SceneA__seq1__frame00001"))
    assert s.describe()["provenance"] == "automatic:poc_robust_dp"


def test_dp_source_guards_the_frozen_digest(world, tmp_path):
    man = tmp_path / "extraction_manifest.json"
    man.write_text(json.dumps({"method_id": "poc_robust_dp", "method_config_digest": "abc"}))
    ok = tmp_path / "ok.digest"; ok.write_text("abc\n")
    bad = tmp_path / "bad.digest"; bad.write_text("xyz\n")
    src.ConditionedDPSource(world["store"], fx.H, fx.W, frozen_digest_file=ok, extraction_manifest=man)
    with pytest.raises(src.SourceError):
        src.ConditionedDPSource(world["store"], fx.H, fx.W, frozen_digest_file=bad, extraction_manifest=man)


def test_silver_source_fills_invalid_columns_with_the_top_edge(world):
    s = src.SilverFullWidthSource(world["maps"], fx.H, fx.W,
                                  expected_model_revision="489d5cd81a0b59fab9b7ea758d3548ebe99677da")
    c = s.get("SceneA__seq1__frame00001__original")
    assert c.provenance == "automatic:segformer_b0_ade20k"
    assert c.row_per_col.size == fx.W
    assert np.all(c.row_per_col[fx.W - 8:] == 0.0)                        # the tower columns
    expected = np.floor(fx.place_curve("SceneA__seq1__frame00001")[: fx.W - 8]) - 0.5
    assert np.allclose(c.row_per_col[: fx.W - 8], np.clip(expected, 0, fx.H - 1), atol=1e-9)
    d = s.describe()
    assert d["full_width_policy"]["invalid_columns"] == "top_edge" and d["convert_version"] == "1.0.0"


def test_silver_source_refuses_images_below_the_floor(world):
    s = src.SilverFullWidthSource(world["maps"], fx.H, fx.W)
    with pytest.raises(CurveError, match="insufficient_sky"):
        s.get("SceneA__seq1__frame00002__winter")
    with pytest.raises(CurveError, match="no_top_connected_sky"):
        s.get("SceneA__seq1__frame00003__evening")
    with pytest.raises(CurveError, match="no label map"):
        s.get("SceneB__seq1__frame00004__evening")


def test_sentinel_mode_returns_a_degenerate_top_edge_curve_for_refusals(world):
    from hsreloc.retrieval.profile import ProfileConfig, is_degenerate, normalize
    s = src.SilverFullWidthSource(world["maps"], fx.H, fx.W, refusal_mode=src.REFUSAL_SENTINEL)
    c = s.get("SceneA__seq1__frame00002__winter")                 # insufficient_sky
    assert c.provenance == "automatic:segformer_b0_ade20k" and c.source_ref.startswith("sentinel:insufficient_sky")
    assert np.all(c.row_per_col == 0.0)
    assert is_degenerate(normalize(c, ProfileConfig(n_samples=64)), 1e-6)   # -> EXTRACTION_FAILURE downstream
    d = src.ConditionedDPSource(world["store"], fx.H, fx.W, refusal_mode=src.REFUSAL_SENTINEL)
    c2 = d.get("SceneB__seq1__frame00004__evening")                 # no stored curve (variant absent)
    assert c2.source_ref.startswith("sentinel:") and np.all(c2.row_per_col == 0.0)
    with pytest.raises(CurveError):
        src.ConditionedDPSource(world["store"], fx.H, fx.W).get("SceneB__seq1__frame00004__evening")


def test_silver_source_guards_the_model_revision(world):
    with pytest.raises(src.SourceError):
        src.SilverFullWidthSource(world["maps"], fx.H, fx.W, expected_model_revision="other")


def test_both_sources_are_digested_and_deterministic(world):
    dp = src.ConditionedDPSource(world["store"], fx.H, fx.W)
    seg = src.SilverFullWidthSource(world["maps"], fx.H, fx.W)
    for s in (dp, seg):
        a = s.get("SceneA__seq1__frame00005__original")
        b = s.get("SceneA__seq1__frame00005__original")
        assert a.digest == b.digest and len(a.digest) == 64
