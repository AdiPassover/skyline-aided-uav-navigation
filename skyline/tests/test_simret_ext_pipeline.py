"""Population-guard and freeze-prereg tests (contracts/freeze.md, research R9)."""

import json
from pathlib import Path

import pytest

from hsreloc.simret import ext, prereg

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "skyline" / "configs" / "sim-final.json"


def _cfg():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


class TestDevGuard:
    def test_dev_refuses_held_out_session(self):
        cfg = _cfg()
        mountain_session = cfg["levels"]["mountains"]["sessions"][0]
        with pytest.raises(ext.ExtError, match="behind the freeze"):
            ext.assert_dev_only(cfg, [mountain_session])

    def test_dev_accepts_village_sessions(self):
        cfg = _cfg()
        ext.assert_dev_only(cfg, cfg["levels"]["asian_village_hills_background"]["sessions"])

    def test_unknown_population_refused(self):
        with pytest.raises(ext.ExtError, match="population"):
            ext.levels_for(_cfg(), "test")


class TestFinalGuard:
    def test_final_refuses_without_committed_prereg(self, tmp_path):
        cfg = _cfg()
        cfg["preregistration_ref"] = str(tmp_path / "does-not-exist.prereg")
        with pytest.raises(prereg.PreregError, match="ext-prereg.*COMMIT"):
            ext.guard_population(cfg, CONFIG.parent, "final")


def _fake_built():
    return {
        "prereg_version": prereg.FINAL_PREREG_VERSION,
        "feature": "20260903-225416-sky-final-sim-validation",
        "experiment": "EXP-SKY-008",
        "raw": {"root": "simulator_skyline_data_extended", "digest": "abc  4878 files"},
        "index": {"content_digest": "d1", "n_rows": 2395, "split_rule": "by level",
                  "sidecar_digests": {"anchors_csv:village": "s1"}},
        "tasks": {"final-exactpose-village": {"task_digest": "t1", "grid_counts": {"250": 10},
                                              "tolerances": {"250": {"tau_pos_m": 125.0}},
                                              "n_query_rows": 80,
                                              "reference_observation_ids_digest": "r1"}},
        "matcher": {"block": {"profile": {"n_samples": 256}}, "baselines": ["ncc"],
                    "primary_baseline": "ncc", "successor": None},
        "acceptance": {"frozen_rule": {"reject_threshold": 0.5},
                       "variant": {"name": "accept-v2-margin",
                                   "frozen": {"theta_s": 0.7, "theta_m": 0.1}}},
        "run_matrix": {"exactpose": {"sources": ["sim_exact"], "levels": ["village"]}},
        "geodetic_origin": {"lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0, "is_synthetic": True},
        "sources": {"sim_exact": {"provenance": "oracle:sim_exact", "gt": True, "kind": "k"}},
        "metrics": prereg.FINAL_METRICS,
        "interpretation": prereg.FINAL_INTERPRETATION,
        "limitations": prereg.FINAL_LIMITATIONS,
    }


class TestFinalPrereg:
    def test_roundtrip_and_tamper_detection(self, tmp_path):
        built = _fake_built()
        path = tmp_path / "sim-final.prereg"
        prereg.write_final_prereg(path, built)
        prereg.check_final_prereg(path, built)                       # identical -> passes
        tampered = json.loads(json.dumps(built))
        tampered["matcher"]["block"]["profile"]["n_samples"] = 512
        with pytest.raises(prereg.PreregError, match="'matcher'"):
            prereg.check_final_prereg(path, tampered)

    def test_unfrozen_acceptance_variant_refused(self):
        cfg = _cfg()
        assert cfg["acceptance_variant"]["frozen"] is None or isinstance(
            cfg["acceptance_variant"]["frozen"], dict)
        cfg["acceptance_variant"]["frozen"] = None
        with pytest.raises(prereg.PreregError, match="frozen is null"):
            prereg.build_final_prereg(cfg, "d", {"content_digest": "x", "n_rows": 1,
                                                 "split_rule": "by level"}, {}, {}, {})

    def test_missing_file_names_the_commit_requirement(self, tmp_path):
        with pytest.raises(prereg.PreregError, match="COMMIT"):
            prereg.check_final_prereg(tmp_path / "nope.prereg", _fake_built())


class TestRunMatrix:
    def test_cells_respect_population_and_dp_levels(self):
        cfg = _cfg()
        dev = ext.cells_for(cfg, "dev")
        assert ("exactpose", "village", "dp") in dev
        assert ("swipes", "village", "dp") in dev            # dp_levels includes village
        assert all(key == "village" for _, key, _s in dev)
        final = ext.cells_for(cfg, "final")
        assert ("swipes", "mountains", "dp") not in final    # DP swipes are DEV-only
        assert ("exactpose", "city", "dp") in final
        assert {key for _, key, _s in final} == {"mountains", "city"}
