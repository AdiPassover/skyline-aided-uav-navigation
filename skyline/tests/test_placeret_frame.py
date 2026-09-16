"""Synthetic ENU scene frames (research R1)."""

from __future__ import annotations

import numpy as np
import pytest

from hsreloc.placeret import frame as fr


def _tilted_walk(n=30, tilt=0.6):
    R = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    local = np.stack([np.linspace(0, 60, n), 3 * np.sin(np.linspace(0, 4, n)), np.zeros(n)], axis=1)
    return local @ R.T, local


def test_pca_frame_recovers_the_plane_and_preserves_distances():
    P, local = _tilted_walk()
    f = fr.fit_scene_frame("KingsCollege", P)
    assert f.off_plane_rms_m < 1e-9
    enu = f.to_enu(P)
    d_enu = np.linalg.norm(np.diff(enu[:, :2], axis=0), axis=1)
    d_sfm = np.linalg.norm(np.diff(P, axis=0), axis=1)
    assert np.allclose(d_enu, d_sfm, atol=1e-9)
    assert np.allclose(enu[:, 2], 0.0, atol=1e-9)


def test_offsets_and_synthetic_flag():
    P, _ = _tilted_walk()
    f = fr.fit_scene_frame("StMarysChurch", P)
    assert f.offset_m == (10_000.0, 10_000.0)
    assert f.is_synthetic
    enu = f.to_enu(P)
    assert enu[:, 0].mean() == pytest.approx(10_000.0, abs=1e-6)
    assert fr.SYNTHETIC_ORIGIN["is_synthetic"] is True


def test_unknown_scene_needs_an_explicit_offset():
    P, _ = _tilted_walk()
    with pytest.raises(fr.FrameError):
        fr.fit_scene_frame("Nowhere", P)
    f = fr.fit_scene_frame("Nowhere", P, offset_m=(1.0, 2.0))
    assert f.offset_m == (1.0, 2.0)


def test_frame_is_deterministic_and_round_trips(tmp_path):
    P, _ = _tilted_walk()
    a = fr.fit_scene_frame("OldHospital", P)
    b = fr.fit_scene_frame("OldHospital", P)
    assert np.array_equal(a.basis, b.basis)
    fr.write_frames({"OldHospital": a}, tmp_path / "frames.json")
    back = fr.read_frames(tmp_path / "frames.json")["OldHospital"]
    assert np.allclose(back.to_enu(P), a.to_enu(P))
    assert np.linalg.det(a.basis) > 0


def test_reader_refuses_a_non_synthetic_origin(tmp_path):
    P, _ = _tilted_walk()
    fr.write_frames({"OldHospital": fr.fit_scene_frame("OldHospital", P)}, tmp_path / "frames.json")
    import json
    d = json.loads((tmp_path / "frames.json").read_text())
    d["geodetic_origin"]["is_synthetic"] = False
    (tmp_path / "frames.json").write_text(json.dumps(d))
    with pytest.raises(fr.FrameError):
        fr.read_frames(tmp_path / "frames.json")
