"""Reference-set loading (T016), including the two curve sources and their cross-check.

Research **R4**. ``curves.npz`` is gitignored while the oracle curves it bundles are tracked, so the
observation-store fallback is what lets the whole feature run in a clean lane checkout. Two sources
of truth are only safe if a disagreement is loud, which is what most of this file pins.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from hsreloc.retrieval.profile import ProfileConfig
from hsreloc.retrieval.refset import (FROM_BOTH, FROM_NPZ, FROM_STORE, ReferenceSetError,
                                      load_reference_set)

H = 96


def test_loads_from_curves_npz_alone(sky_corpus, tmp_path):
    copied = tmp_path / "refset"
    shutil.copytree(sky_corpus["reference_set"], copied)
    rs = load_reference_set(copied, image_height_px=H)
    assert rs.curve_source_used == FROM_NPZ
    assert len(rs.references) == 9
    assert all(r.curve.row_per_col.size == 128 for r in rs.references)


def test_loads_from_the_observation_store_when_npz_is_absent(sky_corpus, tmp_path):
    """The lane-checkout case: curves.npz is gitignored, its contents are not."""
    copied = tmp_path / "refset"
    shutil.copytree(sky_corpus["reference_set"], copied)
    (copied / "curves.npz").unlink()
    rs = load_reference_set(copied, image_height_px=H,
                            observation_store_root=sky_corpus["observations"])
    assert rs.curve_source_used == FROM_STORE
    assert len(rs.references) == 9


def test_both_sources_present_are_cross_checked(sky_corpus):
    rs = load_reference_set(sky_corpus["reference_set"], image_height_px=H,
                            observation_store_root=sky_corpus["observations"])
    assert rs.curve_source_used == FROM_BOTH


def test_disagreeing_sources_are_a_hard_error(sky_corpus, tmp_path):
    """Refusing to guess which source is authoritative is the whole point of the cross-check."""
    copied = tmp_path / "refset"
    shutil.copytree(sky_corpus["reference_set"], copied)
    store = tmp_path / "observations"
    shutil.copytree(sky_corpus["observations"], store)

    target = store / "synth_ref" / "skylines_oracle" / "ref_3.csv"
    lines = target.read_text(encoding="utf-8").splitlines()
    lines[1] = "0,42.0"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ReferenceSetError, match="disagree"):
        load_reference_set(copied, image_height_px=H, observation_store_root=store)


def test_missing_curve_names_the_reference(sky_corpus, tmp_path):
    copied = tmp_path / "refset"
    shutil.copytree(sky_corpus["reference_set"], copied)
    (copied / "curves.npz").unlink()
    store = tmp_path / "observations"
    shutil.copytree(sky_corpus["observations"], store)
    (store / "synth_ref" / "skylines_oracle" / "ref_5.csv").unlink()
    with pytest.raises(ReferenceSetError, match="ref_5"):
        load_reference_set(copied, image_height_px=H, observation_store_root=store)


def test_no_curve_source_at_all_is_refused(sky_corpus, tmp_path):
    copied = tmp_path / "refset"
    shutil.copytree(sky_corpus["reference_set"], copied)
    (copied / "curves.npz").unlink()
    with pytest.raises(ReferenceSetError, match="gitignored"):
        load_reference_set(copied, image_height_px=H)


def test_manifest_origin_is_surfaced(sky_corpus):
    """Contract 1.2.0 -- without it the reference coordinates would have no declared frame."""
    rs = load_reference_set(sky_corpus["reference_set"], image_height_px=H)
    assert rs.origin is not None
    assert rs.origin.matches(sky_corpus["origin"])


def test_profiles_are_attached_when_a_profile_config_is_given(sky_corpus):
    rs = load_reference_set(sky_corpus["reference_set"], image_height_px=H,
                            profile_config=ProfileConfig(n_samples=256))
    assert all(r.profile is not None and r.profile.shape == (256,) for r in rs.references)
    assert all(abs(float(np.mean(r.profile))) < 1e-12 for r in rs.references)


def test_reference_source_block_is_carried_verbatim(sky_corpus):
    rs = load_reference_set(sky_corpus["reference_set"], image_height_px=H)
    block = rs.reference_source
    assert block["kind"] == "historical_imagery"
    assert block["extraction_mode"] == "oracle:sim_exact"
    assert block["evidence_tier"] == "T1"
