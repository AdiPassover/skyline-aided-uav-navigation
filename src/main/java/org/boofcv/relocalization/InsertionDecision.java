package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * The outcome of offering an observation to the reference memory (design §6.2). Every refusal
 * carries the first rule that failed, in the order the rule is written, so the log explains why
 * a frame did not become a trusted anchor.
 *
 * @param frameIndex    the offered frame
 * @param timestampS    its timestamp
 * @param outcome       what happened
 * @param reference     the inserted reference, present only for {@link Outcome#INSERTED}
 * @param noveltyDistance {@code 1 − NCC} (North) to the last trusted reference, when it was computed
 * @param framesSinceLast frames since the last trusted reference, when one exists
 */
public record InsertionDecision(int frameIndex, double timestampS, Outcome outcome,
                                @Nullable TrustedReference reference,
                                @Nullable Double noveltyDistance,
                                @Nullable Integer framesSinceLast) {

    /** The insertion rule's terms, in evaluation order. */
    public enum Outcome {
        INSERTED,
        REJECTED_SKYLINE_INVALID,
        /** The persistent position is unknown (after a hard loss or dropout, before a re-anchor). */
        REJECTED_GLOBAL_POSITION_INVALID,
        REJECTED_UNRESOLVED_INSTABILITY,
        REJECTED_EXPOSURE_BEYOND_SEARCH_BOUND,
        REJECTED_NOT_NOVEL;

        public String wireName() {
            return name().toLowerCase();
        }
    }

    public InsertionDecision {
        if (outcome == null) {
            throw new IllegalArgumentException("outcome is required");
        }
        if ((outcome == Outcome.INSERTED) != (reference != null)) {
            throw new IllegalArgumentException("reference present iff INSERTED");
        }
    }

    public boolean inserted() {
        return outcome == Outcome.INSERTED;
    }
}
