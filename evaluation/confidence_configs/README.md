# Confidence calibration configurations

Versioned artifacts mapping stitching-VO signals to a confidence verdict. Read by both the online
Java scorer (`org.boofcv.confidence.CalibrationConfig`) and the offline Python re-scorer
(`naveval.confidence`), which must produce **bitwise-identical** results from the same file
(asserted by `ConfidenceAgreementFixtureTest` and `evaluation/tests/test_confidence_agreement.py`).

## `unvalidated` in a calibration id is load-bearing, not a placeholder

`bootstrap-unvalidated-v2` is named that way on purpose. The id is written into every result the
calibration produces — `frames.csv`, `manifest.json`, `metrics.json`, and any table exported from
them — so a reviewer cannot encounter one of its numbers without also encountering the word.

**Do not rename it to something tidier.** Feature spec FR-018 forbids presenting an unvalidated
threshold or weight as validated, and SC-009 requires the label to appear in the artifact's own
output. Putting the word in the identity is how that requirement survives copy-paste into a thesis
table; a note in a README does not.

A calibration earns a name without `unvalidated` only when an experiment record supports its
values, at which point `validated` becomes `true` and `validated_by` names the experiment.

## Adding a calibration

- **Never edit an existing file in place** once it has produced results. Add a new one with a new
  id. The digest is over the file's bytes, so an edit silently invalidates the traceability of
  every result that cited the old digest.
- Every signal named in `rejection_rules` or the `score_model` must also appear in `required_signals`.
  Both loaders refuse otherwise — that list is what drives the refusal to score a run that lacks a
  signal, rather than defaulting it (FR-017).
- Weights must sum to `1.0`. Not a mathematical necessity for a weighted sum; a set that does not
  almost always means a term was edited and another forgotten.
- Thresholds are compared strictly (`lt` is `<`, `gt` is `>`). A value sitting exactly on a
  threshold must be classified identically by both languages.

## Schema 2.0.0 (`DEC-CONF-002` as amended, 2026-08-29)

Every calibration now additionally carries, and both loaders refuse a file without:

- **`configuration_binding`** (amendment A3) — the estimator/acquisition context the calibration is
  valid for. Scoring a run whose capture context does not match every bound key is refused;
  transfer outside the binding is unvalidated by definition.
- **`temporal`** (amendment A4) — the window `W` and warm-up `m` the windowed signals were captured
  under. Part of the identity: a different `W` is a different calibration.
- **`probability_semantics`** (amendment A1) — `false` until the pre-registered dual gate
  (episode-adequacy floor + held-out reliability/Brier, `EXP-CONF-001`) passes. Structurally
  refused while `validated` is false. While false, the score is an ordinal index, never a
  probability.
- **`score_model`** — a discriminated union replacing 1.0.0's `score_terms`/`combine`:
  `weighted_sum` (the bootstrap plumbing form), `logistic` (raw-signal coefficients, squashed
  algebraically — no `exp` on the shared path, see the contract), or `isotonic` (single-signal
  piecewise-constant monotone map, the fallback form).

Schema 1.0.0 files are refused by both loaders. `bootstrap-unvalidated-v1.json` (1.0.0) was
replaced by `bootstrap-unvalidated-v2.json` before any run record ever cited it and is not
included.

## Current files

| File | Validated | Purpose |
|---|---|---|
| `bootstrap-unvalidated-v2.json` | no | Exercises the plumbing and the agreement invariant. Its thresholds restate existing behaviour or are arbitrary round numbers; see the file's own `notes` field. |
| `contrast-unvalidated-v2.json` | no | The deliberately-different second calibration for the agreement contract's two-calibrations case (FR-019); placeholder logistic coefficients, visibly unfitted. |
