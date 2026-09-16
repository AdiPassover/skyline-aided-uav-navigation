"""Run config, digest, and preflight (T020).

Contract: ``contracts/matcher-run-config.md``. Preflight's job is to refuse **before** anything is
written, so a run either produces a complete, trustworthy record or produces nothing at all.

The geodetic-origin cross-check has a specific history: the spec-007 query set declared an origin
~86 km from the frame its own ENU coordinates were in (research **R5**). That defect is fixed, but
the guard stays, because a wrong origin is silently wrong -- retrieval, tolerances and error metrics
all stay correct while the absolute fix lands in the wrong country.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from hsreloc.retrieval.runconfig import ConfigError, canonical_digest, load_config, parse_config, preflight


def _cfg(sky_run_config_dict, tmp_path, **overrides):
    cfg = copy.deepcopy(sky_run_config_dict)
    cfg["output"] = {"root": str(tmp_path / "skyline_runs")}
    for dotted, value in overrides.items():
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return parse_config(cfg, base_dir=tmp_path)


# --- digest ------------------------------------------------------------------------------------

def test_digest_is_stable_and_key_order_independent():
    a = {"run_id": "x", "match": {"baseline": "ncc", "recall_k": 5}}
    b = {"match": {"recall_k": 5, "baseline": "ncc"}, "run_id": "x"}
    assert canonical_digest(a) == canonical_digest(b)


def test_digest_changes_when_a_setting_changes():
    a = {"run_id": "x", "match": {"baseline": "ncc"}}
    b = {"run_id": "x", "match": {"baseline": "l1"}}
    assert canonical_digest(a) != canonical_digest(b)


def test_digest_ignores_a_previously_written_digest_field():
    a = {"run_id": "x"}
    b = {"run_id": "x", "config_digest": "stale"}
    assert canonical_digest(a) == canonical_digest(b)


# --- structural validation -----------------------------------------------------------------------

def test_config_round_trips_from_disk(sky_run_config_dict, tmp_path):
    raw = copy.deepcopy(sky_run_config_dict)
    raw["output"] = {"root": str(tmp_path / "runs")}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.run_id == raw["run_id"] and cfg.baseline == "ncc"
    assert cfg.config_path == path.resolve()


def test_missing_required_key_is_rejected(sky_run_config_dict, tmp_path):
    raw = copy.deepcopy(sky_run_config_dict)
    del raw["geodetic_origin"]
    raw["output"] = {"root": str(tmp_path)}
    with pytest.raises(ConfigError, match="geodetic_origin"):
        parse_config(raw, base_dir=tmp_path)


def test_unknown_baseline_is_rejected(sky_run_config_dict, tmp_path):
    with pytest.raises(ConfigError, match="match.baseline"):
        _cfg(sky_run_config_dict, tmp_path, **{"match.baseline": "netvlad"})


def test_mock_is_refused_on_anything_but_a_t1_run(sky_run_config_dict, tmp_path):
    """The guard is on evidence tier, not plumbing: mock must never touch real evidence."""
    with pytest.raises(ConfigError, match="mock baseline is refused"):
        _cfg(sky_run_config_dict, tmp_path, **{"match.baseline": "mock", "evidence.tier": "T3"})
    assert _cfg(sky_run_config_dict, tmp_path, **{"match.baseline": "mock"}).baseline == "mock"


def test_target_hardware_claim_is_rejected(sky_run_config_dict, tmp_path):
    with pytest.raises(ConfigError, match="is_target_hardware"):
        _cfg(sky_run_config_dict, tmp_path, **{"environment.is_target_hardware": True})


def test_unsupported_profile_units_are_rejected(sky_run_config_dict, tmp_path):
    with pytest.raises(Exception, match="elevation-angle"):
        _cfg(sky_run_config_dict, tmp_path, **{"profile.units": "elevation_angle_deg"})


# --- preflight -----------------------------------------------------------------------------------

def test_preflight_accepts_the_fixture(sky_run_config_dict, tmp_path):
    pre = preflight(_cfg(sky_run_config_dict, tmp_path))
    assert len(pre.query_set.queries) == 9
    assert len(pre.reference_set.references) == 9
    assert len(pre.query_curves_digest) == 64
    assert len(pre.consumed) == 9


def test_preflight_rejects_a_tier_that_disagrees_with_the_query_set(sky_run_config_dict, tmp_path):
    with pytest.raises(ConfigError, match="evidence tier mismatch"):
        preflight(_cfg(sky_run_config_dict, tmp_path, **{"evidence.tier": "T3"}))


def test_preflight_rejects_a_disagreeing_geodetic_origin(sky_run_config_dict, tmp_path):
    bad = {"lat_deg": 63.4401, "lon_deg": 10.45111, "alt_m": 42.0}
    with pytest.raises(ConfigError, match="geodetic origin disagreement"):
        preflight(_cfg(sky_run_config_dict, tmp_path, geodetic_origin=bad))


def test_preflight_rejects_a_provenance_the_run_did_not_declare(sky_run_config_dict, tmp_path):
    with pytest.raises(ConfigError, match="oracle run must never silently consume"):
        preflight(_cfg(sky_run_config_dict, tmp_path,
                       **{"inputs.query_curves.expect_provenance": "oracle:manual"}))


def test_preflight_rejects_a_missing_query_curve(sky_run_config_dict, tmp_path, sky_corpus):
    import shutil
    store = tmp_path / "observations"
    shutil.copytree(sky_corpus["observations"], store)
    (store / "synth_query" / "skylines_oracle" / "q_degenerate.csv").unlink()
    with pytest.raises(ConfigError, match="no oracle curve"):
        preflight(_cfg(sky_run_config_dict, tmp_path, **{"inputs.query_curves.root": str(store)}))


def test_preflight_refuses_an_existing_run_id(sky_run_config_dict, tmp_path):
    cfg = _cfg(sky_run_config_dict, tmp_path)
    (cfg.output_root / cfg.run_id).mkdir(parents=True)
    with pytest.raises(ConfigError, match="already exists"):
        preflight(cfg)


def test_preflight_refuses_lag_search_which_is_not_implemented(sky_run_config_dict, tmp_path):
    with pytest.raises(ConfigError, match="lag_search is not implemented"):
        preflight(_cfg(sky_run_config_dict, tmp_path, **{"match.lag_search": True}))


def test_preflight_rejects_a_missing_input_path(sky_run_config_dict, tmp_path):
    with pytest.raises(ConfigError, match="query set not found"):
        preflight(_cfg(sky_run_config_dict, tmp_path,
                       **{"inputs.query_set": str(tmp_path / "nope")}))


def test_preflight_can_skip_the_run_id_check_for_read_only_callers(sky_run_config_dict, tmp_path):
    """The diagnostic writes nothing, and is most useful *after* a run has produced a record."""
    cfg = _cfg(sky_run_config_dict, tmp_path)
    (cfg.output_root / cfg.run_id).mkdir(parents=True)
    with pytest.raises(ConfigError, match="already exists"):
        preflight(cfg)
    pre = preflight(cfg, check_run_id=False)
    assert len(pre.reference_set.references) == 9
