"""The three standing guarantees of Stage 1: no leakage, determinism, no real data.

Tasks T029, T030, T031. Spec FR-018, FR-024, SC-001, SC-009.

These are the tests that make a *later* Stage-2 number mean something. If the matcher could see
ground truth, a good result would prove nothing; if a run were not reproducible, a result could not
be re-derived; and if Stage 1 secretly depended on the real dataset, it would not be the independent
proof of the chain that the checkpoint discipline assumes.
"""

from __future__ import annotations

import ast
import copy
import csv
import json
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from hsreloc.retrieval import queryset as queryset_mod
from hsreloc.retrieval.queryset import (MatcherQuery, MatcherQuerySet, QuerySetError,
                                        load_matcher_query_set)
from hsreloc.retrieval.refset import Reference
from hsreloc.retrieval.run import execute
from hsreloc.retrieval.runconfig import parse_config
from hsreloc.retrieval.skyline_curve import SkylineCurve

RETRIEVAL_DIR = Path(__file__).resolve().parents[1] / "hsreloc" / "retrieval"


def _code_strings(path: Path) -> list:
    """Every string literal a module *executes*, excluding docstrings (and comments, which the AST
    drops anyway).

    The distinction matters: these modules explain at length what they must not do -- read ground
    truth, borrow a DEM-line assumption, touch Nordland -- and that prose is the documentation
    Principle II asks for. Scanning raw text would punish a module for describing its own
    constraints, which is precisely backwards. What must be clean is the code.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)                     and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


def _reads_key(path: Path, key: str) -> bool:
    """Does this module *look up* ``key`` -- ``row[key]`` or ``row.get(key)`` -- anywhere?

    Naming a field is not reading it: ``queryset.FORBIDDEN_FIELDS`` lists ``in_coverage`` precisely
    in order to forbid it, and a test that punished the mention would push the guard out of the code
    and into a comment. What must be absent is the *access*.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)                 and node.slice.value == key:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)                 and node.func.attr in ("get", "pop", "getdefault") and node.args                 and isinstance(node.args[0], ast.Constant) and node.args[0].value == key:
            return True
    return False


def _imported_names(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def _run(sky_run_config_dict, tmp_path, baseline="ncc", run_id="g1"):
    raw = copy.deepcopy(sky_run_config_dict)
    raw["match"]["baseline"] = baseline
    raw["run_id"] = run_id
    raw["output"] = {"root": str(tmp_path / "skyline_runs")}
    return execute(parse_config(raw, base_dir=tmp_path))


# --- T029: no ground truth reaches the matcher ----------------------------------------------------

@pytest.mark.parametrize("cls", [MatcherQuery, MatcherQuerySet, SkylineCurve, Reference])
def test_matcher_visible_types_carry_no_ground_truth_field(cls):
    """A leak would have to be a *field*; there is none, so there is nothing to read by accident."""
    names = {f.name for f in dataclass_fields(cls)}
    forbidden = {"in_coverage", "gt_east_m", "gt_north_m", "gt_heading_deg", "is_correct",
                 "ground_truth", "answer", "expected"}
    assert not (names & forbidden), f"{cls.__name__} exposes {sorted(names & forbidden)}"


def test_reference_position_is_not_ground_truth_but_is_present():
    """A reference's own coordinate is the data being searched, not an answer about a query."""
    names = {f.name for f in dataclass_fields(Reference)}
    assert {"east_m", "north_m"} <= names


def test_the_matcher_query_loader_never_reads_ground_truth(sky_corpus):
    """groundtruth.csv may be deleted and the matcher's view still loads."""
    import shutil

    copied = Path(str(sky_corpus["query_set"]) + "-noGT")
    if copied.exists():
        shutil.rmtree(copied)
    shutil.copytree(sky_corpus["query_set"], copied)
    (copied / "groundtruth.csv").unlink()
    qs = load_matcher_query_set(copied)
    assert len(qs.queries) == 9
    shutil.rmtree(copied)


def test_in_coverage_is_present_in_the_data_but_never_parsed(sky_corpus):
    """The column exists in the query set -- proof the loader is choosing not to read it."""
    with (Path(sky_corpus["query_set"]) / "skyline_queries.csv").open(encoding="utf-8") as f:
        assert "in_coverage" in (csv.DictReader(f).fieldnames or [])
    for path in RETRIEVAL_DIR.glob("*.py"):
        assert not _reads_key(path, "in_coverage"), f"{path.name} looks up in_coverage"


def test_no_retrieval_module_opens_groundtruth():
    """No matcher module names the ground-truth file *in code*, or imports its GT-bearing loader."""
    for path in RETRIEVAL_DIR.glob("*.py"):
        assert not any("groundtruth" in s for s in _code_strings(path)), (
            f"{path.name} names groundtruth.csv in executable code")
        assert "load_skyline_query_set" not in _imported_names(path), (
            f"{path.name} imports naveval's GT-bearing query-set loader")


def test_forbidden_field_list_is_kept_honest():
    assert "in_coverage" in queryset_mod.FORBIDDEN_FIELDS


# --- T030: determinism ---------------------------------------------------------------------------

def test_two_runs_of_the_same_config_produce_identical_results(sky_run_config_dict, tmp_path):
    a = _run(sky_run_config_dict, tmp_path, run_id="det-a")
    b = _run(sky_run_config_dict, tmp_path, run_id="det-b")

    assert a["config_digest"] != b["config_digest"]      # run_id differs, so the digest must too
    assert a["query_curves_digest"] == b["query_curves_digest"]
    assert a["outcomes"] == b["outcomes"]

    def rows(summary):
        with (summary["record_dir"] / "queries.csv").open(encoding="utf-8", newline="") as f:
            out = []
            for row in csv.DictReader(f):
                # Wall-clock timing is measured, so it legitimately varies between runs; everything
                # that describes *what the matcher decided* must not.
                row.pop("process_time_ns", None)
                out.append(row)
            return out

    assert rows(a) == rows(b)


def test_identical_configs_produce_identical_digests(sky_run_config_dict, tmp_path):
    raw = copy.deepcopy(sky_run_config_dict)
    raw["output"] = {"root": str(tmp_path / "runs")}
    first = parse_config(copy.deepcopy(raw), base_dir=tmp_path)
    second = parse_config(copy.deepcopy(raw), base_dir=tmp_path)
    assert first.config_digest == second.config_digest


def test_the_record_carries_both_digests(sky_run_config_dict, tmp_path):
    summary = _run(sky_run_config_dict, tmp_path, run_id="dig")
    manifest = json.loads((summary["record_dir"] / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["relocalizer_config"]
    assert cfg["config_digest"] == summary["config_digest"]
    assert cfg["query_curves_digest"] == summary["query_curves_digest"]
    assert len(cfg["query_curves_digest"]) == 64


def test_a_changed_curve_changes_the_curve_digest(sky_run_config_dict, sky_corpus, tmp_path):
    """The digest is the reproducibility anchor: silent input drift must be detectable."""
    import shutil

    store = tmp_path / "observations"
    shutil.copytree(sky_corpus["observations"], store)
    baseline_summary = _run(sky_run_config_dict, tmp_path, run_id="cd-a")

    target = store / "synth_query" / "skylines_oracle" / "q_unambiguous_3.csv"
    lines = target.read_text(encoding="utf-8").splitlines()
    lines[1] = "0,50.5"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = copy.deepcopy(sky_run_config_dict)
    raw["inputs"]["query_curves"]["root"] = str(store)
    raw["run_id"] = "cd-b"
    raw["output"] = {"root": str(tmp_path / "skyline_runs")}
    changed = execute(parse_config(raw, base_dir=tmp_path))
    assert changed["query_curves_digest"] != baseline_summary["query_curves_digest"]


# --- T031: Stage 1 needs no real data --------------------------------------------------------------

FORBIDDEN_REAL_ARTIFACTS = (
    "nordland", "skyquery-nordland", "refset-nordland", "observations/summer", "observations/fall",
    "Stitching-Navigation/skyline_refdb", "2026-08-22b", "2026-08-23a",
)


def _stage1_sources():
    yield from RETRIEVAL_DIR.glob("*.py")
    tests_dir = Path(__file__).resolve().parent
    for name in ("synth_corpus.py", "conftest.py", "test_retrieval_units.py",
                 "test_retrieval_acceptance.py", "test_retrieval_refset.py",
                 "test_retrieval_record.py", "test_retrieval_runconfig.py",
                 "test_retrieval_e2e_synthetic.py", "test_retrieval_stage1_guarantees.py"):
        yield tests_dir / name
    configs = Path(__file__).resolve().parents[1] / "configs"
    if configs.exists():
        for path in configs.glob("sky-stage1-*.json"):
            yield path


def test_no_stage1_source_references_a_real_dataset():
    """Stage 1's independence is asserted, not assumed -- it is the checkpoint's whole premise.

    Executable strings only. These modules discuss Nordland at length in their docstrings, because
    explaining *why* a choice was made (no detrending, no elevation-angle units) requires naming the
    dataset those choices were made against. Documentation is not a dependency.
    """
    offenders = []
    this_file = Path(__file__).resolve()
    for path in _stage1_sources():
        if not path.exists() or path.resolve() == this_file:
            continue          # this file necessarily spells out the tokens it forbids
        if path.suffix == ".json":
            literals = [path.read_text(encoding="utf-8")]
        else:
            literals = _code_strings(path)
        for literal in literals:
            for token in FORBIDDEN_REAL_ARTIFACTS:
                if token.lower() in literal.lower():
                    offenders.append(f"{path.name}: {token!r} in {literal[:60]!r}")
    assert not offenders, f"Stage-1 sources reference real artifacts in code: {offenders}"


def test_stage1_runs_entirely_inside_a_temporary_tree(sky_run_config_dict, sky_corpus, tmp_path):
    """Every path the run touches is under a temp dir -- no repository dataset is read."""
    summary = _run(sky_run_config_dict, tmp_path, run_id="iso")
    manifest = json.loads((summary["record_dir"] / "manifest.json").read_text(encoding="utf-8"))
    repo_datasets = (REPO_ROOT / "datasets").resolve()
    repo_refdb = (REPO_ROOT / "skyline_refdb").resolve()
    repo_observations = (REPO_ROOT / "observations").resolve()
    for value in manifest["relocalizer_config"]["paths"].values():
        if not value:
            continue
        resolved = Path(value).resolve()
        for forbidden in (repo_datasets, repo_refdb, repo_observations):
            assert forbidden not in resolved.parents and resolved != forbidden, (
                f"Stage 1 touched a repository data path: {resolved}")


def test_the_query_set_the_fixture_builds_is_labelled_synthetic_and_t1(sky_corpus):
    descriptor = json.loads(
        (Path(sky_corpus["query_set"]) / "dataset.json").read_text(encoding="utf-8"))
    assert descriptor["source_type"] == "synthetic"
    assert descriptor["evidence_tier"] == "T1"
    assert "not real data" in descriptor["evidence_caveat"].lower()


def test_recorded_paths_are_repo_relative_when_inside_the_repository(sky_run_config_dict, tmp_path):
    """A record is evidence and gets committed: an absolute local path identifies a machine, not an
    input, and the next reader cannot resolve it. Paths genuinely outside the repo stay absolute,
    because there the machine-specific location is the honest answer."""
    from hsreloc.retrieval.run import _portable

    summary = _run(sky_run_config_dict, tmp_path, run_id="paths")
    manifest = json.loads((summary["record_dir"] / "manifest.json").read_text(encoding="utf-8"))
    paths = manifest["relocalizer_config"]["paths"]
    # The fixture lives in a pytest tmp dir, i.e. outside the repo, so these stay absolute.
    assert all(v is None or Path(v).is_absolute() for v in paths.values())
    # A path inside the repository is emitted relative, with forward slashes.
    inside = REPO_ROOT / "skyline" / "configs"
    assert _portable(inside) == "skyline/configs"
