"""Shared fixtures for the SKY matcher (Stage 1) tests.

The corpus is session-scoped because building it writes a full observation store, a reference set and
a query set to disk; every Stage-1 test reads the same one, and none of them mutate it.

Nothing here touches real data. That is not an incidental property -- ``test_retrieval_stage1_isolation``
asserts it, because Stage 1's whole value is that it proves the chain without spending the one real
dataset or making any real-world claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from tests import synth_corpus  # noqa: E402


@pytest.fixture(scope="session")
def sky_corpus(tmp_path_factory):
    """The Stage-1 synthetic corpus: observation store + reference set + query set + answers."""
    root = tmp_path_factory.mktemp("sky_corpus")
    return synth_corpus.build_corpus(root)


@pytest.fixture(scope="session")
def sky_run_config_dict(sky_corpus):
    """A valid Stage-1 run config, as a dict, pointing at the synthetic corpus."""
    origin = sky_corpus["origin"]
    return {
        "schema_version": "1.1.0",
        "run_id": "sky-stage1-fixture",
        "inputs": {
            "reference_set": str(sky_corpus["reference_set"]),
            "query_set": str(sky_corpus["query_set"]),
            "query_curves": {
                "kind": "observation_store",
                "root": str(sky_corpus["observations"]),
                "expect_provenance": "oracle:sim_exact",
            },
        },
        "geodetic_origin": origin.as_dict(),
        "profile": {"n_samples": 256, "normalize_mean": True, "detrend": False,
                    "units": "image_fraction"},
        "match": {"baseline": "ncc", "recall_k": 5, "lag_search": False},
        "acceptance": {"confidence_k": 5.0, "reject_threshold": 0.5, "viable_sigma": 3.0,
                       "ambiguity_margin_sigma": 1.0, "degenerate_profile_std": 1e-6},
        "output": {"root": "unset -- each test supplies its own"},
        "evidence": {"tier": "T1", "caveat": "synthetic Stage-1 fixture; not real-world evidence"},
        "environment": {"is_target_hardware": False},
    }
