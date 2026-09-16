package org.boofcv.confidence;

import javax.annotation.Nullable;

import java.util.Objects;

/**
 * The per-frame confidence verdict (feature spec FR-002 – FR-006).
 *
 * <p>Exactly one exists for every <em>processed</em> frame, including frames where the estimator
 * produced no pose and frames where no signals could be read (FR-001). A missing result would be
 * indistinguishable from a lost record.
 *
 * <h2>The score is an ordinal reliability index, not a probability</h2>
 *
 * <p>Range {@code [0,1]}, higher meaning more reliable. It is <b>not</b> a calibrated likelihood of
 * correctness and must not be reported, plotted, or described as one. Nothing in this repository
 * currently establishes any relationship between it and realized error, in either direction; that
 * is {@code EXP-CONF-001}'s question. "Confidence in [0,1]" invites exactly this misreading, which
 * is why the warning sits on the type rather than only in a document.
 *
 * <h2>Confidence is distinct from correctness</h2>
 *
 * <p>Constitution Principle X requires the two be treated separately, and this contract permits and
 * represents the dangerous case: a high-confidence estimate that is in fact wrong. Nothing here
 * defines confidence so that this cannot occur — a design in which it could not occur would be
 * measuring correctness, which is unavailable at runtime.
 *
 * @param outcome           never null
 * @param reason            never null; names the condition that produced {@code outcome}
 * @param score             {@code [0,1]} when the frame was scored, absent otherwise. Present
 *                          <b>exactly</b> when {@code outcome} is {@link ConfidenceOutcome#USABLE}
 *                          or {@link ConfidenceOutcome#DEGRADED} — every other outcome
 *                          short-circuits before a score is computed. Never a sentinel such as -1
 * @param signals           the raw block this verdict was computed from; never null, though every
 *                          field inside it may be absent
 * @param calibrationId     human-readable identity of the calibration used, e.g.
 *                          {@code bootstrap-unvalidated-v2}
 * @param calibrationDigest content digest of that calibration, proving two results came from
 *                          byte-identical configuration
 */
public record ConfidenceResult(
        ConfidenceOutcome outcome,
        ConfidenceReason reason,
        @Nullable Double score,
        SignalBlock signals,
        String calibrationId,
        String calibrationDigest) {

    /** Compact constructor enforcing the invariants in {@code contracts/confidence-result.md}. */
    public ConfidenceResult {
        Objects.requireNonNull(outcome, "outcome");
        Objects.requireNonNull(reason, "reason");
        Objects.requireNonNull(signals, "signals");
        Objects.requireNonNull(calibrationId, "calibrationId");
        Objects.requireNonNull(calibrationDigest, "calibrationDigest");

        // Invariant 3: a score exists exactly when the frame was scored -- that is, when the
        // outcome is USABLE or DEGRADED. Every other outcome short-circuits earlier: the estimator
        // produced nothing, a required signal was absent, or a rejection rule fired before scoring.
        //
        // Stated on OUTCOME, not on reason. Phrasing it on reason was wrong and the tests caught
        // it: it forced a score onto rejected frames, which would have meant inventing a number
        // for an estimate we had just refused -- exactly the "plausible value for something nobody
        // measured" failure this class exists to prevent.
        //
        // A biconditional on purpose: a null score with a scored outcome means a verdict was formed
        // without a value, and a non-null score with an unscored outcome means a value was invented.
        boolean scoreExpected = outcome == ConfidenceOutcome.USABLE
                || outcome == ConfidenceOutcome.DEGRADED;

        if (!scoreExpected && score != null) {
            throw new IllegalArgumentException(
                    "outcome " + outcome + " is not scored, but a score of " + score + " was supplied");
        }
        if (scoreExpected && score == null) {
            throw new IllegalArgumentException(
                    "outcome " + outcome + " requires a score, got none");
        }
        if (score != null) {
            if (Double.isNaN(score) || Double.isInfinite(score)) {
                throw new IllegalArgumentException("score must be finite, got " + score);
            }
            if (score < 0.0 || score > 1.0) {
                throw new IllegalArgumentException("score must be in [0,1], got " + score);
            }
        }

        // Invariant 5: a rejection always names a condition.
        if (outcome == ConfidenceOutcome.REJECTED && !reason.isRejectionCondition()) {
            throw new IllegalArgumentException(
                    "REJECTED requires a named rejection condition, got reason " + reason);
        }
        // Invariant 6: a degradation never borrows a condition's name.
        if (outcome == ConfidenceOutcome.DEGRADED
                && reason != ConfidenceReason.OK && reason != ConfidenceReason.LOW_SCORE) {
            throw new IllegalArgumentException(
                    "DEGRADED must not carry a rejection condition's reason, got " + reason);
        }
        if (outcome == ConfidenceOutcome.USABLE
                && reason != ConfidenceReason.OK) {
            throw new IllegalArgumentException(
                    "USABLE must carry reason OK, got " + reason);
        }
        // NOT_PRODUCED means the estimator reported no update, and nothing else. The run-record
        // contract pins this as a biconditional with success == false, so the first frame -- which
        // does produce a pose (the origin) but cannot be judged -- must not land here. It is
        // REJECTED/NOT_ESTABLISHED instead.
        if (outcome == ConfidenceOutcome.NOT_PRODUCED
                && reason != ConfidenceReason.ESTIMATOR_FAILED) {
            throw new IllegalArgumentException(
                    "NOT_PRODUCED must carry ESTIMATOR_FAILED, got " + reason);
        }
    }

    /**
     * Whether a consumer may use the accompanying pose as a measurement.
     *
     * <p>True for {@link ConfidenceOutcome#USABLE} and {@link ConfidenceOutcome#DEGRADED} — the
     * latter is a low number attached to a real estimate, not a refusal.
     */
    public boolean poseConsumable() {
        return outcome.poseExists();
    }

    /**
     * Whether this frame was scored at all.
     *
     * <p>False for {@link ConfidenceOutcome#REJECTED} and {@link ConfidenceOutcome#NOT_PRODUCED} —
     * an absent score there means "never computed", never "computed as zero".
     */
    public boolean isScored() {
        return score != null;
    }

    /** Whether the verdict was formed from at least one measured signal. */
    public boolean hasSignals() {
        return !signals.isEmpty();
    }
}
