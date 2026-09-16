"""The versioned run configuration, its digest, and preflight.

Contract: ``contracts/matcher-run-config.md``. One config drives one run; its digest is stamped into
the emitted record so a result traces back to exactly the settings that produced it (Principle VI).

This config deliberately carries **no metric, tolerance or ground-truth field**. Those belong to the
evaluation config (``006/contracts/evaluation-config.md``), on the other side of the file boundary.
Keeping them out is not tidiness -- it is what makes it structurally impossible for the matcher to be
tuned against the numbers it is being judged by.

Preflight refuses before anything is written, rather than half-producing a record:

1. every configured path exists;
2. the reference set and the matcher's view of the query set both load;
3. the **geodetic origin agrees across all three sources** that declare one -- the run config, the
   query set's ``dataset.json`` and the reference set's ``manifest.json``. This is the guard for
   research **R5**, where the query set declared an origin ~86 km from the frame its own coordinates
   were in. That defect is fixed; the guard stays, because it is what would catch a recurrence;
4. every query resolves to a curve covering every column;
5. the ``mock`` baseline is refused on anything but a **T1** run;
6. the declared evidence tier matches the query set's own;
7. ``is_target_hardware`` is false;
8. ``run_id`` is free.

*Implementation note, 2026-08-23.* ``contracts/matcher-run-config.md`` first keyed the mock guard on
``query_curves.kind != "synthetic"``. That turned out to guard the wrong thing: Stage 1 deliberately
runs the **real** ``OracleCurveSource`` code path over a *synthetic* observation store, so that Stage 2
exercises identical code and a Stage-2 result stays attributable to the representation rather than to
a separate Stage-1-only pipeline. Keying on ``kind`` would have banned the mock from the very
comparison it exists to make. The guard is now on **evidence tier**, which is the property that
actually matters -- can this run produce a real-world claim? -- and a real query set is T3, so the mock
still cannot touch real evidence. Contract updated to 1.1.0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hsreloc.retrieval import baselines
from hsreloc.retrieval.acceptance import AcceptanceConfig
from hsreloc.retrieval.fix import GeodeticOrigin
from hsreloc.retrieval.profile import ProfileConfig
from hsreloc.retrieval.queryset import load_matcher_query_set
from hsreloc.retrieval.refset import load_reference_set
from hsreloc.retrieval.skyline_curve import CurveError
from hsreloc.retrieval.sources import OracleCurveSource

SCHEMA_VERSION = "1.1.0"
SYNTHETIC = "synthetic"
OBSERVATION_STORE = "observation_store"
T1 = "T1"


class ConfigError(Exception):
    """A run configuration is malformed, or preflight refused it."""


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    reference_set: Path
    query_set: Path
    query_curves_kind: str
    query_curves_root: Optional[Path]
    expect_provenance: str
    geodetic_origin: GeodeticOrigin
    profile: ProfileConfig
    baseline: str
    recall_k: int
    lag_search: bool
    acceptance: AcceptanceConfig
    output_root: Path
    evidence_tier: str
    evidence_caveat: str
    mock_seed: int
    raw: dict
    config_digest: str
    config_path: Optional[Path] = None


def canonical_digest(payload: dict) -> str:
    """Stable digest of a config: sorted keys, no incidental ordering, no digest of itself."""
    body = {k: v for k, v in payload.items() if k != "config_digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path | str) -> RunConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return parse_config(raw, base_dir=Path(path).resolve().parent, config_path=Path(path).resolve())


def parse_config(raw: dict, base_dir: Path, config_path: Optional[Path] = None) -> RunConfig:
    for key in ("run_id", "inputs", "geodetic_origin", "match", "output", "evidence"):
        if key not in raw:
            raise ConfigError(f"run config is missing required key {key!r}")

    inputs = raw["inputs"]
    for key in ("reference_set", "query_set", "query_curves"):
        if key not in inputs:
            raise ConfigError(f"run config 'inputs' is missing required key {key!r}")

    curves = inputs["query_curves"]
    kind = curves.get("kind")
    if kind not in (SYNTHETIC, OBSERVATION_STORE):
        raise ConfigError(
            f"inputs.query_curves.kind must be {OBSERVATION_STORE!r} or {SYNTHETIC!r}, got {kind!r}")
    root = _resolve(base_dir, curves["root"]) if curves.get("root") else None
    if kind == OBSERVATION_STORE and root is None:
        raise ConfigError("inputs.query_curves.root is required when kind is 'observation_store'")

    match = raw["match"]
    baseline = match.get("baseline")
    if baseline not in baselines.ALL_BASELINES:
        raise ConfigError(
            f"match.baseline must be one of {baselines.ALL_BASELINES}, got {baseline!r}")

    evidence = raw["evidence"]
    tier = evidence.get("tier")
    if not tier:
        raise ConfigError("evidence.tier is required -- a run cannot be unlabelled")

    if baseline == baselines.MOCK and tier != T1:
        raise ConfigError(
            f"the mock baseline is refused on a {tier} run: it ranks by identity and carries no "
            f"skyline information, so a record it produces must never be read as real-world "
            f"evidence. Mock runs are T1 only."
        )

    env = raw.get("environment", {})
    if env.get("is_target_hardware"):
        raise ConfigError(
            "environment.is_target_hardware must be false: no target UAV hardware is specified "
            "repo-wide, so no onboard or real-time claim is available (Principle IX)")

    profile_raw = raw.get("profile", {})
    profile = ProfileConfig(
        n_samples=int(profile_raw.get("n_samples", 256)),
        normalize_mean=bool(profile_raw.get("normalize_mean", True)),
        detrend=bool(profile_raw.get("detrend", False)),
        units=profile_raw.get("units", "image_fraction"),
    )
    profile.validate()

    acc_raw = raw.get("acceptance", {})
    acceptance = AcceptanceConfig(
        confidence_k=float(acc_raw.get("confidence_k", 5.0)),
        reject_threshold=float(acc_raw.get("reject_threshold", 0.5)),
        viable_sigma=float(acc_raw.get("viable_sigma", 3.0)),
        ambiguity_margin_sigma=float(acc_raw.get("ambiguity_margin_sigma", 1.0)),
        degenerate_profile_std=float(acc_raw.get("degenerate_profile_std", 1e-6)),
    )

    return RunConfig(
        run_id=raw["run_id"],
        reference_set=_resolve(base_dir, inputs["reference_set"]),
        query_set=_resolve(base_dir, inputs["query_set"]),
        query_curves_kind=kind,
        query_curves_root=root,
        expect_provenance=curves.get("expect_provenance", "oracle:manual"),
        geodetic_origin=GeodeticOrigin.from_dict(raw["geodetic_origin"]),
        profile=profile,
        baseline=baseline,
        recall_k=int(match.get("recall_k", 5)),
        lag_search=bool(match.get("lag_search", False)),
        acceptance=acceptance,
        output_root=_resolve(base_dir, raw["output"]["root"]),
        evidence_tier=tier,
        evidence_caveat=evidence.get("caveat", ""),
        mock_seed=int(match.get("mock_seed", 0)),
        raw=raw,
        config_digest=canonical_digest(raw),
        config_path=config_path,
    )


@dataclass
class PreflightResult:
    query_set: object
    reference_set: object
    curve_source: object
    origin: GeodeticOrigin
    query_curves_digest: str
    consumed: list


def preflight(config: RunConfig, curve_source=None, check_run_id: bool = True) -> PreflightResult:
    """Verify everything before a single output byte is written. Raises ``ConfigError`` on refusal.

    ``check_run_id=False`` is for callers that will not write anything -- the reference
    self-similarity diagnostic, for instance, reads reference curves and reports; refusing it
    because a *record* already exists would be guarding the wrong thing, and would make the
    diagnostic impossible to re-run on a completed experiment, which is exactly when it is wanted.
    """
    if config.lag_search:
        raise ConfigError(
            "match.lag_search is not implemented: a lag search would estimate a heading this dataset "
            "declares unsupported, and would inflate similarity exactly for a self-similar corridor "
            "(research R10). It is reserved for a labelled secondary run.")

    if check_run_id:
        record_dir = config.output_root / config.run_id
        if record_dir.exists():
            raise ConfigError(
                f"run_id {config.run_id!r} already exists at {record_dir} -- refusing to overwrite a "
                f"recorded run")

    if not config.query_set.exists():
        raise ConfigError(f"query set not found at {config.query_set}")
    if not config.reference_set.exists():
        raise ConfigError(f"reference set not found at {config.reference_set}")

    qs = load_matcher_query_set(config.query_set)

    if qs.evidence_tier != config.evidence_tier:
        raise ConfigError(
            f"evidence tier mismatch: the run config declares {config.evidence_tier!r} but the query "
            f"set {qs.dataset_id!r} is {qs.evidence_tier!r}. A config cannot relabel a run's tier.")

    rs = load_reference_set(
        config.reference_set,
        image_height_px=qs.image_height_px,
        observation_store_root=config.query_curves_root,
        profile_config=config.profile,
    )

    # --- the R5 guard: every declared origin must agree ---
    declared = {"run_config": config.geodetic_origin}
    if qs.local_frame_origin:
        declared["query_set"] = GeodeticOrigin.from_dict(qs.local_frame_origin)
    if rs.origin is not None:
        declared["reference_set"] = rs.origin
    for name, origin in declared.items():
        if not origin.matches(config.geodetic_origin):
            values = {k: v.as_dict() for k, v in declared.items()}
            raise ConfigError(
                f"geodetic origin disagreement ({name} differs from the run config): {values}. "
                f"Refusing to guess which frame the coordinates are in -- an ENU->geodetic conversion "
                f"against the wrong origin is silently wrong by kilometres (research R5).")

    if curve_source is None:
        if config.query_curves_kind != OBSERVATION_STORE:
            raise ConfigError(
                f"a {config.query_curves_kind!r} curve source must be supplied programmatically; "
                f"only {OBSERVATION_STORE!r} can be built from the config alone")
        curve_source = OracleCurveSource(
            root=config.query_curves_root,
            sessions=qs.session_map(),
            image_height_px=qs.image_height_px,
            image_width_px=qs.image_width_px,
            expect_provenance=config.expect_provenance,
        )

    consumed = []
    digests = []
    for query in qs.queries:
        try:
            curve = curve_source.get(query.observation_id)
        except CurveError as exc:
            raise ConfigError(f"query {query.query_index} ({query.observation_id}): {exc}") from exc
        # Two independent checks, because they can fail independently.
        #
        # (1) What the *dataset* declares. This is the cross-artifact one that has teeth: the query
        #     set records how each curve was obtained, and a run that declares something else is
        #     reading data it did not intend to. `skyline_queries.csv` stores the suffix only
        #     ("manual", "sim_exact"), so the suffix is what is compared -- the column cannot
        #     distinguish `oracle:foo` from `automatic:foo`, which is a limitation of that contract
        #     rather than of this check.
        declared = query.oracle_provenance
        if declared and declared != config.expect_provenance.split(":", 1)[-1]:
            raise ConfigError(
                f"query {query.observation_id}: the query set declares oracle_provenance "
                f"{declared!r} but the run declared {config.expect_provenance!r} -- an oracle run "
                f"must never silently consume curves of another kind")
        # (2) What the *source* returns. Vacuous for OracleCurveSource, which stamps what it was
        #     configured with, but not for a source that knows its own provenance -- a future
        #     automatic extractor reading its own directory is exactly that case.
        if curve.provenance != config.expect_provenance:
            raise ConfigError(
                f"query {query.observation_id}: source returned provenance {curve.provenance!r} but "
                f"the run declared {config.expect_provenance!r}")
        consumed.append({"query_index": query.query_index, "observation_id": query.observation_id,
                         "session_id": query.session_id, "source_ref": curve.source_ref,
                         "digest": curve.digest, "n_columns": int(curve.row_per_col.size)})
        digests.append(curve.digest)

    combined = hashlib.sha256("\n".join(sorted(digests)).encode("utf-8")).hexdigest()
    return PreflightResult(query_set=qs, reference_set=rs, curve_source=curve_source,
                           origin=config.geodetic_origin, query_curves_digest=combined,
                           consumed=consumed)
