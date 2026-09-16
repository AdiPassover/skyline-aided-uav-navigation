"""Offline confidence re-scoring (DEC-CONF-001 / DEC-CONF-002 as amended).

Loads a schema-2.0.0 calibration configuration, refuses what it cannot honour (FR-017 and
amendment A3's configuration binding), and re-scores a run record's persisted signals to
reproduce the online Java verdicts **bitwise** (contracts/agreement.md).

Arithmetic discipline, mirrored from ``org.boofcv.confidence.ConfidenceScorer`` and pinned by
``evaluation/tests/test_confidence_agreement.py``:

- accumulation is left-to-right with ``+=`` in configured order — never ``sum()``/``math.fsum``;
- clamping happens per normalised term, then once on the total;
- comparisons are strict (``lt`` is ``<``, ``gt`` is ``>``);
- absence short-circuits before any arithmetic;
- **no transcendental function on the shared path**: the logistic model's score is the linear
  predictor pushed through the algebraic squash ``0.5 + 0.5 * (z / (1 + |z|))`` — ``exp`` is not
  identically rounded across Java and Python, the squash is (only exactly-rounded IEEE ops).

The temporal-state replay (``replay_windowed_signals``) re-derives the windowed signals from the
persisted raw columns under the pre-registered A4 semantics, so a persisted ``relative_support`` or
``inc_log_scale_dispersion`` is independently checkable rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from naveval.errors import ContractViolationError
from naveval.runrecord import FrameEstimate, RunRecord

SCHEMA_VERSION = "2.0.0"

#: Signals a calibration may name (SignalBlock's canonical names + the derived ratio).
KNOWN_SIGNALS = frozenset({
    "track_count", "inlier_count", "inlier_ratio",
    "residual_inlier_count", "residual_mean_sq_px", "residual_rms_px",
    "residual_median_sq_px", "residual_max_sq_px", "inlier_threshold_sq_px",
    "inlier_coverage", "keyframe_age",
    "relative_support", "inc_flow_px", "inc_log_scale_dispersion",
})

#: Signals carried by every v1.0.0 record; the rest need the v1.1.0 confidence columns.
V1_0_SIGNALS = frozenset({"track_count", "inlier_count", "inlier_ratio"})

_VALID_OUTCOMES = ("usable", "degraded", "rejected", "not_produced")
_REJECTION_REASONS = frozenset({
    "not_established", "estimator_failed", "insufficient_features", "poor_inlier_support",
    "high_reprojection_error", "inconsistent_motion", "invalid_homography", "signals_unavailable",
})

_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "calibration_id", "validated", "validated_by", "probability_semantics",
    "notes", "required_signals", "configuration_binding", "temporal", "rejection_rules",
    "score_model", "usable_score_bound",
})


class CalibrationRefusalError(ContractViolationError):
    """A calibration or a run was refused rather than silently defaulted (FR-017 / A3)."""


def _refuse(message: str) -> None:
    raise CalibrationRefusalError(message)


def digest_of(text: str) -> str:
    """SHA-256 over the configuration's UTF-8 bytes with line endings normalised to LF.

    Byte-hashing, not canonical re-serialisation: Java and Python disagree on the textual form of
    some doubles, which would make a re-serialisation digest a latent cross-language mismatch.
    Mirrors ``CalibrationConfig.digestOf`` exactly.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return "sha256:" + hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _require(d: dict, key: str, context: str):
    if key not in d or d[key] is None:
        _refuse(f"Missing required field: {context}{key}")
    return d[key]


def _require_finite(value, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _refuse(f"{field} must be a number, got {value!r}")
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        _refuse(f"{field} must be finite, got {v}")
    return v


@dataclass(frozen=True)
class RejectionRule:
    signal: str
    op: str          # "lt" | "gt", strict
    threshold: float
    reason: str

    def fires(self, value: float) -> bool:
        return value < self.threshold if self.op == "lt" else value > self.threshold


@dataclass(frozen=True)
class ScoreTerm:
    signal: str
    normalise: str   # "identity" | "saturating" | "inverse_saturating"
    reference: Optional[float]
    weight: float


@dataclass(frozen=True)
class Coefficient:
    signal: str
    coefficient: float


@dataclass(frozen=True)
class ScoreModel:
    type: str                                  # "weighted_sum" | "logistic" | "isotonic"
    terms: tuple = ()                          # weighted_sum
    intercept: float = 0.0                     # logistic
    coefficients: tuple = ()                   # logistic
    signal: str = ""                           # isotonic
    thresholds: tuple = ()                     # isotonic, strictly ascending
    values: tuple = ()                         # isotonic, len == len(thresholds) + 1

    def signals(self) -> list:
        if self.type == "weighted_sum":
            return [t.signal for t in self.terms]
        if self.type == "logistic":
            return [c.signal for c in self.coefficients]
        return [self.signal]


@dataclass(frozen=True)
class CalibrationConfig:
    schema_version: str
    calibration_id: str
    validated: bool
    validated_by: Optional[str]
    probability_semantics: bool
    required_signals: tuple
    configuration_binding: dict
    window_w: int
    warmup_m: int
    rejection_rules: tuple
    score_model: ScoreModel
    usable_score_bound: float
    digest: str


def load_calibration(path: Path | str) -> CalibrationConfig:
    text = Path(path).read_text(encoding="utf-8")
    return parse_calibration(text)


def parse_calibration(text: str) -> CalibrationConfig:  # noqa: C901 - one validator, kept together
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        _refuse(f"Malformed calibration configuration: {exc}")

    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        # A typo must be an error, not a silently ignored term that changes the score.
        _refuse(f"Unknown calibration fields: {sorted(unknown)}")

    schema_version = _require(raw, "schema_version", "")
    if schema_version != SCHEMA_VERSION:
        _refuse(
            f"Unsupported calibration schema_version {schema_version!r}; this loader accepts "
            f"{SCHEMA_VERSION} only. 1.0.0 configurations predate DEC-CONF-002 and carry no "
            f"configuration binding; re-author them under 2.0.0."
        )

    calibration_id = _require(raw, "calibration_id", "")
    if "validated" not in raw:
        _refuse("Missing required field: validated")
    validated = bool(raw["validated"])
    validated_by = raw.get("validated_by")
    if "probability_semantics" not in raw:
        _refuse("Missing required field: probability_semantics")
    probability_semantics = bool(raw["probability_semantics"])
    if probability_semantics and (not validated or validated_by is None):
        _refuse("probability_semantics may be true only for a validated calibration naming its "
                "validating experiment (amendment A1)")

    required_signals = tuple(_require(raw, "required_signals", ""))
    for s in required_signals:
        if s not in KNOWN_SIGNALS:
            _refuse(f"Unknown signal name: {s!r}")

    binding = _require(raw, "configuration_binding", "")
    if not isinstance(binding, dict) or not binding:
        _refuse("configuration_binding must be a non-empty object (amendment A3): a calibration "
                "with no stated validity domain cannot refuse a mismatched run")
    for k, v in binding.items():
        if not isinstance(v, (int, float, str, bool)) or v is None:
            _refuse(f"configuration_binding values must be scalar; {k!r} is {type(v).__name__}")

    temporal = _require(raw, "temporal", "")
    if set(temporal) - {"window_w", "warmup_m"}:
        _refuse(f"Unknown temporal fields: {sorted(set(temporal) - {'window_w', 'warmup_m'})}")
    window_w = int(_require(temporal, "window_w", "temporal."))
    warmup_m = int(_require(temporal, "warmup_m", "temporal."))
    if window_w < 2 or warmup_m < 2 or warmup_m > window_w:
        _refuse(f"temporal requires 2 <= warmup_m <= window_w; got W={window_w}, m={warmup_m}")

    rules = []
    for i, r in enumerate(_require(raw, "rejection_rules", "")):
        if set(r) - {"signal", "op", "threshold", "reason"}:
            _refuse(f"Unknown rejection_rules[{i}] fields")
        signal = _require(r, "signal", f"rejection_rules[{i}].")
        if signal not in KNOWN_SIGNALS:
            _refuse(f"Unknown signal name: {signal!r}")
        op = _require(r, "op", f"rejection_rules[{i}].")
        if op not in ("lt", "gt"):
            _refuse(f"Unknown comparison op: {op!r}")
        threshold = _require_finite(_require(r, "threshold", f"rejection_rules[{i}]."),
                                    f"rejection_rules[{i}].threshold")
        reason = _require(r, "reason", f"rejection_rules[{i}].")
        if reason not in _REJECTION_REASONS:
            _refuse(f"Not a valid rejection reason: {reason!r}")
        rules.append(RejectionRule(signal, op, threshold, reason))

    model = _parse_score_model(_require(raw, "score_model", ""))

    usable = _require_finite(_require(raw, "usable_score_bound", ""), "usable_score_bound")
    if not (0.0 <= usable <= 1.0):
        _refuse(f"usable_score_bound must be in [0,1], got {usable}")

    declared = set(required_signals)
    for rule in rules:
        if rule.signal not in declared:
            _refuse(f"rejection rule reads {rule.signal!r} which is not in required_signals")
    for s in model.signals():
        if s not in declared:
            _refuse(f"score model reads {s!r} which is not in required_signals")

    return CalibrationConfig(
        schema_version=schema_version,
        calibration_id=calibration_id,
        validated=validated,
        validated_by=validated_by,
        probability_semantics=probability_semantics,
        required_signals=required_signals,
        configuration_binding=dict(binding),
        window_w=window_w,
        warmup_m=warmup_m,
        rejection_rules=tuple(rules),
        score_model=model,
        usable_score_bound=usable,
        digest=digest_of(text),
    )


def _parse_score_model(dto: dict) -> ScoreModel:  # noqa: C901
    mtype = _require(dto, "type", "score_model.")
    if mtype == "weighted_sum":
        if set(dto) - {"type", "terms"}:
            _refuse("weighted_sum accepts only 'terms'")
        terms = []
        weight_sum = 0.0
        for i, t in enumerate(_require(dto, "terms", "score_model.")):
            if set(t) - {"signal", "normalise", "reference", "weight"}:
                _refuse(f"Unknown score_model.terms[{i}] fields")
            signal = _require(t, "signal", f"score_model.terms[{i}].")
            if signal not in KNOWN_SIGNALS:
                _refuse(f"Unknown signal name: {signal!r}")
            normalise = _require(t, "normalise", f"score_model.terms[{i}].")
            if normalise not in ("identity", "saturating", "inverse_saturating"):
                _refuse(f"Unknown normalisation: {normalise!r}")
            reference = t.get("reference")
            if normalise != "identity" and reference is None:
                _refuse(f"normalisation {normalise!r} requires a reference for signal {signal!r}")
            if reference is not None:
                reference = _require_finite(reference, f"score_model.terms[{i}].reference")
                if reference == 0.0:
                    _refuse("reference must be non-zero")
            weight = _require_finite(_require(t, "weight", f"score_model.terms[{i}]."),
                                     f"score_model.terms[{i}].weight")
            if weight < 0.0:
                _refuse(f"weight must be >= 0, got {weight}")
            terms.append(ScoreTerm(signal, normalise, reference, weight))
            weight_sum += weight
        if not terms:
            _refuse("weighted_sum terms must not be empty")
        if abs(weight_sum - 1.0) > 1e-9:
            _refuse(f"weighted_sum weights must sum to 1.0, got {weight_sum}")
        return ScoreModel(type="weighted_sum", terms=tuple(terms))

    if mtype == "logistic":
        if set(dto) - {"type", "intercept", "coefficients"}:
            _refuse("logistic accepts only 'intercept' and 'coefficients'")
        intercept = _require_finite(_require(dto, "intercept", "score_model."), "score_model.intercept")
        coefficients = []
        for i, c in enumerate(_require(dto, "coefficients", "score_model.")):
            if set(c) - {"signal", "coefficient"}:
                _refuse(f"Unknown score_model.coefficients[{i}] fields")
            signal = _require(c, "signal", f"score_model.coefficients[{i}].")
            if signal not in KNOWN_SIGNALS:
                _refuse(f"Unknown signal name: {signal!r}")
            coefficient = _require_finite(
                _require(c, "coefficient", f"score_model.coefficients[{i}]."),
                f"score_model.coefficients[{i}].coefficient")
            coefficients.append(Coefficient(signal, coefficient))
        if not coefficients:
            _refuse("logistic coefficients must not be empty")
        return ScoreModel(type="logistic", intercept=intercept, coefficients=tuple(coefficients))

    if mtype == "isotonic":
        if set(dto) - {"type", "signal", "thresholds", "values"}:
            _refuse("isotonic accepts only 'signal', 'thresholds' and 'values'")
        signal = _require(dto, "signal", "score_model.")
        if signal not in KNOWN_SIGNALS:
            _refuse(f"Unknown signal name: {signal!r}")
        thresholds = [_require_finite(t, f"score_model.thresholds[{i}]")
                      for i, t in enumerate(_require(dto, "thresholds", "score_model."))]
        values = [_require_finite(v, f"score_model.values[{i}]")
                  for i, v in enumerate(_require(dto, "values", "score_model."))]
        if len(values) != len(thresholds) + 1:
            _refuse(f"isotonic values must have thresholds+1 entries, got {len(values)} for "
                    f"{len(thresholds)} thresholds")
        for a, b in zip(thresholds, thresholds[1:]):
            if b <= a:
                _refuse("isotonic thresholds must be strictly ascending")
        for v in values:
            if not (0.0 <= v <= 1.0):
                _refuse(f"isotonic values must be in [0,1], got {v}")
        return ScoreModel(type="isotonic", signal=signal,
                          thresholds=tuple(thresholds), values=tuple(values))

    _refuse(f"Unknown score_model.type: {mtype!r}")
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Signal access and scoring (the mirror of org.boofcv.confidence.ConfidenceScorer)
# ---------------------------------------------------------------------------

def carried_signals(record: RunRecord) -> frozenset:
    """The signals this record's schema carries (FR-017's refusal basis).

    Per-row absence is a *verdict* (``signals_unavailable``); schema-level absence is a *refusal*.
    A v1.0.0 record carries only the counts and their derived ratio — it has no verdict and no
    confidence columns, and emphatically does not have "low" ones.
    """
    first: FrameEstimate = record.frame_estimates[0]
    if getattr(first, "confidence", None) is not None:
        return frozenset(KNOWN_SIGNALS)
    return V1_0_SIGNALS


def signal_value(fe: FrameEstimate, name: str) -> Optional[float]:
    """One frame's signal by canonical name; ``None`` is absence, never zero. Unknown names throw."""
    if name not in KNOWN_SIGNALS:
        raise ContractViolationError(f"Unknown signal name: {name!r}")
    if name == "track_count":
        return None if fe.track_count is None else float(fe.track_count)
    if name == "inlier_count":
        return None if fe.inlier_count is None else float(fe.inlier_count)
    if name == "inlier_ratio":
        if fe.track_count is None or fe.inlier_count is None or fe.track_count == 0:
            return None
        return fe.inlier_count / fe.track_count
    c = getattr(fe, "confidence", None)
    if c is None:
        return None
    mapping = {
        "residual_inlier_count": None if c.residual_inlier_count is None else float(c.residual_inlier_count),
        "residual_mean_sq_px": c.residual_mean_sq_px,
        "residual_rms_px": c.residual_rms_px,
        "residual_median_sq_px": c.residual_median_sq_px,
        "residual_max_sq_px": c.residual_max_sq_px,
        "inlier_threshold_sq_px": c.inlier_threshold_sq_px,
        "inlier_coverage": c.inlier_coverage,
        "keyframe_age": None if c.keyframe_age is None else float(c.keyframe_age),
        "relative_support": c.relative_support,
        "inc_flow_px": c.inc_flow_px,
        "inc_log_scale_dispersion": c.inc_log_scale_dispersion,
    }
    return mapping[name]


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    return min(v, 1.0)


def _normalise(kind: str, value: float, reference: Optional[float]) -> float:
    if kind == "identity":
        return _clamp01(value)
    if kind == "saturating":
        return _clamp01(value / reference)
    return _clamp01(1.0 / (1.0 + value / reference))


def _compute_score(calibration: CalibrationConfig, values: dict) -> float:
    model = calibration.score_model
    if model.type == "weighted_sum":
        total = 0.0
        for term in model.terms:
            total += term.weight * _normalise(term.normalise, values[term.signal], term.reference)
        return _clamp01(total)
    if model.type == "logistic":
        z = model.intercept
        for c in model.coefficients:
            z += c.coefficient * values[c.signal]
        return 0.5 + 0.5 * (z / (1.0 + abs(z)))
    value = values[model.signal]
    idx = 0
    for t in model.thresholds:
        if value > t:               # strict, matching Op.GT's strictness discipline
            idx += 1
    return model.values[idx]


@dataclass(frozen=True)
class ScoredFrame:
    frame_index: int
    outcome: str
    reason: str
    score: Optional[float]


def score_frame(calibration: CalibrationConfig, fe: FrameEstimate, first_frame: bool) -> ScoredFrame:
    """One frame's verdict — the exact mirror of ``ConfidenceScorer.score``."""
    # 1. The estimator produced nothing. Its report, not our judgement.
    if not fe.success:
        return ScoredFrame(fe.frame_index, "not_produced", "estimator_failed", None)
    # 2. First frame: nothing to judge, and deliberately not not_produced.
    if first_frame:
        return ScoredFrame(fe.frame_index, "rejected", "not_established", None)
    # 3. Availability, before any arithmetic.
    values = {}
    for name in calibration.required_signals:
        v = signal_value(fe, name)
        if v is None:
            return ScoredFrame(fe.frame_index, "rejected", "signals_unavailable", None)
        values[name] = v
    # 4. Rejection rules, in configured order, first match wins.
    for rule in calibration.rejection_rules:
        if rule.fires(values[rule.signal]):
            return ScoredFrame(fe.frame_index, "rejected", rule.reason, None)
    # 5. Score, then the usable/degraded split.
    score = _compute_score(calibration, values)
    if score >= calibration.usable_score_bound:
        return ScoredFrame(fe.frame_index, "usable", "ok", score)
    return ScoredFrame(fe.frame_index, "degraded", "low_score", score)


# ---------------------------------------------------------------------------
# Run-level refusals (FR-017, amendments A3/A4) and re-scoring
# ---------------------------------------------------------------------------

def check_run_compatibility(record: RunRecord, calibration: CalibrationConfig) -> None:
    """Every refusal that must precede scoring, with the signal and the run named."""
    run_name = record.manifest.run_id or "<unnamed run>"

    carried = carried_signals(record)
    for name in calibration.required_signals:
        if name not in carried:
            _refuse(
                f"Calibration {calibration.calibration_id!r} requires signal {name!r} which run "
                f"{run_name!r} does not carry (schema {record.manifest.schema_version}). A v1.0.0 "
                f"record has no confidence signals — absent, not low — and is never defaulted."
            )

    conf_block = getattr(record.manifest, "confidence", None)

    # Amendment A4: the windowed signals were captured under a specific (W, m); scoring them under
    # a calibration expecting different temporal identity would silently compare different signals.
    windowed = {"relative_support", "inc_log_scale_dispersion"} & set(calibration.required_signals)
    if windowed:
        if not conf_block:
            _refuse(f"Calibration {calibration.calibration_id!r} reads windowed signals but run "
                    f"{run_name!r} carries no confidence block naming their capture (W, m)")
        w = conf_block.get("signal_window_w")
        m = conf_block.get("signal_warmup_m")
        if w != calibration.window_w or m != calibration.warmup_m:
            _refuse(
                f"Run {run_name!r} captured windowed signals under W={w}, m={m} but calibration "
                f"{calibration.calibration_id!r} is identified by W={calibration.window_w}, "
                f"m={calibration.warmup_m}; a different W is a different calibration (amendment A4)"
            )

    # Amendment A3: the configuration binding.
    context = dict(record.manifest.estimator_config or {})
    if conf_block and isinstance(conf_block.get("capture_context"), dict):
        context.update(conf_block["capture_context"])
    for key, bound in calibration.configuration_binding.items():
        if key not in context:
            _refuse(f"Calibration {calibration.calibration_id!r} binds {key!r} but run "
                    f"{run_name!r} does not carry it; transfer outside the binding is unvalidated "
                    f"(amendment A3)")
        actual = context[key]
        if isinstance(bound, bool) or isinstance(actual, bool):
            equal = bound is actual if isinstance(bound, bool) and isinstance(actual, bool) else False
        elif isinstance(bound, (int, float)) and isinstance(actual, (int, float)):
            equal = float(bound) == float(actual)
        else:
            equal = bound == actual
        if not equal:
            _refuse(f"Calibration {calibration.calibration_id!r} requires {key} = {bound!r} but "
                    f"run {run_name!r} has {actual!r}; transfer outside the binding is unvalidated "
                    f"(amendment A3)")


def rescore_run(record: RunRecord, calibration: CalibrationConfig) -> list:
    """Re-scores every frame offline; returns ``list[ScoredFrame]`` in file order."""
    check_run_compatibility(record, calibration)
    return [score_frame(calibration, fe, i == 0)
            for i, fe in enumerate(record.frame_estimates)]


def verify_agreement(record: RunRecord, calibration: CalibrationConfig) -> list:
    """The agreement invariant (contracts/agreement.md): re-scored == persisted, exactly.

    Returns a list of human-readable mismatch strings — empty means the invariant holds. Exact
    means string equality on outcome/reason and **bitwise** equality on the score (no tolerance:
    a tolerance would silently absorb precisely the divergence this exists to catch).
    """
    mismatches = []
    persisted_digest = None
    conf_block = getattr(record.manifest, "confidence", None)
    if conf_block:
        persisted_digest = conf_block.get("calibration_digest")
    if persisted_digest is not None and persisted_digest != calibration.digest:
        mismatches.append(
            f"calibration digest mismatch: run names {persisted_digest}, loaded file is "
            f"{calibration.digest} — this is not the calibration the run was scored under")
        return mismatches

    for i, fe in enumerate(record.frame_estimates):
        c = getattr(fe, "confidence", None)
        if c is None:
            mismatches.append(f"frame {fe.frame_index}: no persisted verdict")
            continue
        rescored = score_frame(calibration, fe, i == 0)
        if rescored.outcome != c.outcome:
            mismatches.append(f"frame {fe.frame_index}: outcome {c.outcome!r} vs re-scored "
                              f"{rescored.outcome!r}")
        if rescored.reason != c.reason:
            mismatches.append(f"frame {fe.frame_index}: reason {c.reason!r} vs re-scored "
                              f"{rescored.reason!r}")
        if (rescored.score is None) != (c.score is None):
            mismatches.append(f"frame {fe.frame_index}: score nullity {c.score!r} vs "
                              f"{rescored.score!r}")
        elif rescored.score is not None and rescored.score != c.score:
            mismatches.append(f"frame {fe.frame_index}: score {c.score!r} vs re-scored "
                              f"{rescored.score!r} (bitwise)")
    return mismatches


# ---------------------------------------------------------------------------
# Temporal-state replay (EXP-CONF-001 §Temporal signal state semantics, amendment A4)
# ---------------------------------------------------------------------------

def replay_windowed_signals(record: RunRecord, window_w: int, warmup_m: int) -> list:
    """Re-derives per-frame ``(relative_support, inc_log_scale_dispersion)`` from raw columns.

    The exact rules of ``org.boofcv.confidence.SignalExtractor``: buffers of the most recent
    ``W`` *valid* frames (success, event neither init nor restart), cleared on a reference change
    **before** the frame computes, strictly-past reference, absence below ``warmup_m``; median with
    ``(a + b) / 2.0`` for even counts; two-pass sample SD in buffer order with ``sqrt(ssq/(n-1))``.

    Returns a list of ``(Optional[float], Optional[float])``, one per frame, for bitwise
    comparison against the persisted columns.
    """
    support: list = []
    log_scale: list = []
    out = []
    for fe in record.frame_estimates:
        event = fe.event
        if event in ("restart", "recenter"):
            support.clear()
            log_scale.clear()

        rs = None
        if fe.inlier_count is not None and len(support) >= warmup_m:
            ordered = sorted(support)
            n = len(ordered)
            if n % 2 == 1:
                ref = float(ordered[n // 2])
            else:
                ref = (ordered[n // 2 - 1] + float(ordered[n // 2])) / 2.0
            rs = None if ref == 0.0 else fe.inlier_count / ref

        disp = None
        n = len(log_scale)
        if n >= warmup_m:
            total = 0.0
            for v in log_scale:
                total += v
            mean = total / n
            ssq = 0.0
            for v in log_scale:
                d = v - mean
                ssq += d * d
            disp = math.sqrt(ssq / (n - 1))

        out.append((rs, disp))

        valid = fe.success and event not in ("init", "restart")
        if valid:
            if fe.inlier_count is not None:
                if len(support) == window_w:
                    support.pop(0)
                support.append(fe.inlier_count)
            c = getattr(fe, "confidence", None)
            inc_log_scale = None if c is None else c.inc_log_scale
            if inc_log_scale is not None and math.isfinite(inc_log_scale):
                if len(log_scale) == window_w:
                    log_scale.pop(0)
                log_scale.append(inc_log_scale)
    return out
