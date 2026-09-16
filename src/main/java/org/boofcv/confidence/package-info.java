/**
 * Stitching-confidence estimation for the VO front end.
 *
 * <p><b>Maturity: {@code prototype}.</b> Every scoring threshold and weight currently shipped is
 * <b>unvalidated</b>: no experiment in this repository yet establishes that any of these signals
 * predicts VO error, in either direction. The calibration artifact carries the word
 * {@code unvalidated} in its own identity so that the label travels with every number produced
 * from it.
 *
 * <h2>What this package is for</h2>
 *
 * <p>It turns the per-frame diagnostics of the stitching VO into an explicit, inspectable verdict:
 * an outcome from a closed set, a reason code naming the condition that produced it, the raw
 * signals it was computed from with absence preserved, and a scalar score carrying the identity of
 * the calibration that produced it. Constitution Principle X requires that low-confidence estimates
 * not be silently consumed; {@link org.boofcv.confidence.ConfidentPose} makes that structural by
 * refusing to hand out a pose without its verdict.
 *
 * <p>The score is an <b>ordinal reliability index, not a probability</b>. It is not a calibrated
 * likelihood of correctness and must not be reported as one.
 *
 * <h2>Scope boundary — no BoofCV type may be imported here</h2>
 *
 * <p>{@code DEC-CONF-001} names the persisted signal block plus the calibration identity as this
 * subsystem's isolating interface, precisely so that a different front end could populate the same
 * signals. Importing a BoofCV type into this package would weld confidence to the current
 * estimator and break Principle VII.
 *
 * <p>The one sanctioned exception is the adapter that reads the VO's diagnostics
 * ({@code SignalExtractor}), which is the only class permitted to name a VO type — the same
 * quarantine pattern {@code VoDiagnostics} already follows for its BoofCV cast.
 *
 * <h2>Determinism</h2>
 *
 * <p>Scoring is a pure function of {@code (SignalBlock, estimatorSucceeded, CalibrationConfig)}.
 * No hidden state, no time dependence, no accumulation across frames. This is not a stylistic
 * preference: {@code DEC-CONF-001} splits computation between an online Java scorer and an offline
 * Python re-scorer, and their exact agreement — asserted bitwise, never within a tolerance — is the
 * load-bearing invariant of the whole design. Temporal signals enter through the signal block, not
 * through scorer state.
 */
package org.boofcv.confidence;
