package org.boofcv.confidence;

import javax.annotation.Nullable;

import java.util.Objects;

/**
 * Turns signals into a verdict (feature spec FR-006, FR-016).
 *
 * <p>A <b>pure function</b> of {@code (SignalBlock, estimatorSucceeded, CalibrationConfig)}. No
 * hidden state, no time dependence, no accumulation across frames. Temporal signals, where they
 * exist, enter through the signal block rather than through scorer state — otherwise the offline
 * re-scorer could not reproduce a verdict from a persisted row, and {@code DEC-CONF-001}'s whole
 * split would collapse.
 *
 * <h2>Order of evaluation, and why it is fixed</h2>
 *
 * <ol>
 *   <li><b>Estimator failure</b> — no estimate exists; nothing to judge.</li>
 *   <li><b>Not established</b> — first frame; no prior state.</li>
 *   <li><b>Signal availability</b> — any required signal absent short-circuits to
 *       {@link ConfidenceReason#SIGNALS_UNAVAILABLE} with a null score. <b>Before</b> any
 *       arithmetic, so an absent signal can never contribute a partially-computed term.</li>
 *   <li><b>Rejection rules</b>, in configured order, first match wins.</li>
 *   <li><b>Score</b>, and only then the usable/degraded split.</li>
 * </ol>
 *
 * <p>Rejections are evaluated before the score so that a rejection is never the accidental
 * consequence of a low number — it is a stated criterion with a name (Principle VIII:
 * <i>distinguish an invalid estimate from a low-confidence estimate</i>).
 *
 * <h2>Arithmetic discipline</h2>
 *
 * <p>{@code contracts/agreement.md} requires the Python re-scorer to reproduce this bitwise, so the
 * things that make floating-point arithmetic order-dependent are pinned here and must not be
 * "optimised": terms accumulate left-to-right in {@code scoreTerms} order with a simple {@code +=};
 * clamping happens per term and then once on the total; ratios are computed as plain double
 * division and never pre-rounded.
 */
public final class ConfidenceScorer {

    private final CalibrationConfig calibration;

    public ConfidenceScorer(CalibrationConfig calibration) {
        this.calibration = Objects.requireNonNull(calibration, "calibration");
    }

    public CalibrationConfig calibration() {
        return calibration;
    }

    /**
     * Scores one frame.
     *
     * @param signals            the frame's raw diagnostics; may be entirely absent
     * @param estimatorSucceeded the estimator's own frame-update result
     * @param firstFrame         whether this is the first frame of the sequence
     */
    public ConfidenceResult score(SignalBlock signals, boolean estimatorSucceeded, boolean firstFrame) {
        Objects.requireNonNull(signals, "signals");

        // 1. The estimator produced nothing. Its report, not our judgement.
        if (!estimatorSucceeded) {
            return result(ConfidenceOutcome.NOT_PRODUCED, ConfidenceReason.ESTIMATOR_FAILED, null, signals);
        }

        // 2. First frame: a pose exists (the origin, by definition) but no motion was estimated
        // against anything, so there is nothing to judge. Withheld rather than scored -- and
        // deliberately not NOT_PRODUCED, which the run-record contract pins to success == false.
        if (firstFrame) {
            return result(ConfidenceOutcome.REJECTED, ConfidenceReason.NOT_ESTABLISHED, null, signals);
        }

        // 3. Availability, before any arithmetic. An absent signal must never contribute a
        // neutral or default value: that is the mechanism by which an unmeasured frame becomes a
        // confident one (FR-004, FR-017).
        for (String required : calibration.requiredSignals()) {
            if (!signals.hasSignal(required)) {
                return result(ConfidenceOutcome.REJECTED, ConfidenceReason.SIGNALS_UNAVAILABLE, null, signals);
            }
        }

        // 4. Rejection rules, in configured order, first match wins.
        for (CalibrationConfig.RejectionRule rule : calibration.rejectionRules()) {
            Double value = signals.signal(rule.signal());
            // Cannot be null: step 3 established every required signal is present, and the
            // constructor established every rule reads a required signal.
            if (rule.op().test(Objects.requireNonNull(value), rule.threshold())) {
                return result(ConfidenceOutcome.REJECTED, rule.reason(), null, signals);
            }
        }

        // 5. Score, then the usable/degraded split.
        double score = computeScore(signals);
        if (score >= calibration.usableScoreBound()) {
            return result(ConfidenceOutcome.USABLE, ConfidenceReason.OK, score, signals);
        }
        return result(ConfidenceOutcome.DEGRADED, ConfidenceReason.LOW_SCORE, score, signals);
    }

    /**
     * Evaluates the calibration's score model.
     *
     * <p>Every accumulation is left-to-right in configured order with a simple {@code +=}. Not
     * {@code stream().sum()}, which may reorder or compensate, and not any pairwise or compensated
     * summation — the Python side must reproduce this exactly, and exact reproduction of a sum
     * requires the same order of operations. No transcendental function appears on this path
     * (see {@link CalibrationConfig}'s class javadoc): the logistic model's score is the linear
     * predictor pushed through the algebraic squash {@code 0.5 + 0.5 * (z / (1 + |z|))}, which is
     * strictly monotone, lies in {@code (0,1)}, and uses only exactly-rounded IEEE-754 operations.
     */
    private double computeScore(SignalBlock signals) {
        CalibrationConfig.ScoreModel model = calibration.scoreModel();
        if (model instanceof CalibrationConfig.WeightedSum ws) {
            double total = 0.0;
            for (CalibrationConfig.ScoreTerm term : ws.terms()) {
                Double value = signals.signal(term.signal());
                double normalised = term.normalise().apply(Objects.requireNonNull(value), term.reference());
                total += term.weight() * normalised;
            }
            return CalibrationConfig.clamp01(total);
        }
        if (model instanceof CalibrationConfig.Logistic lo) {
            double z = lo.intercept();
            for (CalibrationConfig.Coefficient c : lo.coefficients()) {
                Double value = signals.signal(c.signal());
                z += c.coefficient() * Objects.requireNonNull(value);
            }
            return 0.5 + 0.5 * (z / (1.0 + Math.abs(z)));
        }
        CalibrationConfig.Isotonic iso = (CalibrationConfig.Isotonic) model;
        double value = Objects.requireNonNull(signals.signal(iso.signal()));
        double[] thresholds = iso.thresholds();
        int idx = 0;
        for (double t : thresholds) {
            if (value > t) {          // strict, matching Op.GT's strictness discipline
                idx++;
            }
        }
        return iso.values()[idx];
    }

    private ConfidenceResult result(ConfidenceOutcome outcome, ConfidenceReason reason,
                                    @Nullable Double score, SignalBlock signals) {
        return new ConfidenceResult(outcome, reason, score, signals,
                calibration.calibrationId(), calibration.digest());
    }
}
