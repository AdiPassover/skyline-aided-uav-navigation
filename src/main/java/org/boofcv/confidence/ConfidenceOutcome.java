package org.boofcv.confidence;

/**
 * The closed set of per-frame confidence outcomes (feature spec FR-002).
 *
 * <p>Four values, and the distinctions between them are the point rather than a taxonomy. Two in
 * particular must never be collapsed:
 *
 * <ul>
 *   <li><b>{@link #REJECTED} vs {@link #NOT_PRODUCED}</b> — the first is <i>this subsystem's</i>
 *       verdict on an estimate that exists; the second is <i>the estimator's</i> report that there
 *       is nothing to judge. A consumer that cannot tell them apart cannot tell "we refused it"
 *       from "there was nothing there", which is the difference between a working rejection
 *       mechanism and a silent one.</li>
 *   <li><b>{@link #DEGRADED} vs {@link #REJECTED}</b> — the first is a low number, the second is a
 *       named condition. Constitution Principle VIII requires distinguishing an invalid estimate
 *       from a low-confidence estimate; conflating these would make every threshold look like a
 *       criterion and every criterion look arbitrary.</li>
 * </ul>
 */
public enum ConfidenceOutcome {

    /** Estimate exists, no rejection condition fired, score at or above the usable bound. */
    USABLE("usable", true),

    /**
     * Estimate exists and no rejection condition fired, but the score is below the usable bound.
     * Consumable <em>with its verdict attached</em> — this is a low number, not a named failure.
     */
    DEGRADED("degraded", true),

    /**
     * A stated rejection criterion fired. The estimate exists but must not be consumed as a
     * measurement. Always accompanied by a reason code naming the condition
     * ({@code ConfidenceResult} invariant 5).
     */
    REJECTED("rejected", false),

    /**
     * The estimator produced no new pose for this frame. There is no estimate to judge — this is
     * not a verdict about quality.
     */
    NOT_PRODUCED("not_produced", false);

    private final String wireName;
    private final boolean poseExists;

    ConfidenceOutcome(String wireName, boolean poseExists) {
        this.wireName = wireName;
        this.poseExists = poseExists;
    }

    /** Lowercase name used in {@code frames.csv} and in the Python re-scorer. */
    public String wireName() {
        return wireName;
    }

    /**
     * Whether a usable pose accompanies this outcome.
     *
     * <p>Note this is deliberately <em>not</em> named {@code isTrustworthy}: {@link #DEGRADED}
     * returns {@code true} here because a pose exists and may be consumed, while carrying a verdict
     * that says how much to believe it. Trust is the consumer's decision, informed by the score;
     * existence is this subsystem's fact.
     */
    public boolean poseExists() {
        return poseExists;
    }

    /** Parses a wire name, rejecting anything outside the closed set. */
    public static ConfidenceOutcome fromWireName(String name) {
        for (ConfidenceOutcome o : values()) {
            if (o.wireName.equals(name)) {
                return o;
            }
        }
        throw new IllegalArgumentException("Not a valid confidence outcome: '" + name + "'");
    }
}
