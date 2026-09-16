package org.boofcv.confidence;

import javax.annotation.Nullable;

/**
 * Why a frame received the outcome it did (feature spec FR-003, FR-020).
 *
 * <p>Exactly one code per result. Reason codes are evaluated <b>before</b> the score, in the order
 * configured in the calibration, first match wins — so a rejection is never the accidental
 * consequence of a low number, it is a stated criterion with a name.
 *
 * <h2>Coverage of Constitution Principle X's VO-side failure states</h2>
 *
 * <p>Principle X names eight failure states, five of which are VO-side. Each is mapped here to
 * either a detectable code or an explicit statement that it is not detectable online. <b>Silent
 * absence is not permitted</b> (FR-020): a state with no code and no explanation would be
 * indistinguishable from one nobody thought about.
 *
 * <table>
 *   <caption>Principle X VO-side coverage</caption>
 *   <tr><th>State</th><th>Code</th><th>Where detected</th></tr>
 *   <tr><td>insufficient tracked features</td><td>{@link #INSUFFICIENT_FEATURES}</td><td>online</td></tr>
 *   <tr><td>poor RANSAC support</td><td>{@link #POOR_INLIER_SUPPORT}</td><td>online</td></tr>
 *   <tr><td>high reprojection error</td><td>{@link #HIGH_REPROJECTION_ERROR}</td><td>online, since {@code DEC-VO-001}</td></tr>
 *   <tr><td>inconsistent motion</td><td>{@link #INCONSISTENT_MOTION}</td><td><b>offline only</b></td></tr>
 *   <tr><td>invalid homography</td><td>{@link #INVALID_HOMOGRAPHY}</td><td><b>offline only</b></td></tr>
 * </table>
 *
 * <p>The two offline-only states are a real limitation, not a formality. The incremental
 * frame-to-keyframe model does not cross the VO boundary — only its <i>residuals</i> do — so the
 * online path has no frame-to-frame transform to test. Offline detection composes consecutive
 * mosaic homographies from the run record instead. Approximating them online with something weaker
 * under the same name would be exactly the substitution FR-012 forbids.
 */
public enum ConfidenceReason {

    /** No rejection condition matched. */
    OK("ok", null, Detectability.NOT_A_FAILURE_STATE),

    /**
     * First frame of a sequence: no prior state exists, so no signal that depends on one is
     * defined. Deliberately neither maximally confident nor maximally unconfident — it is a third
     * thing, and the score is null.
     */
    NOT_ESTABLISHED("not_established", null, Detectability.NOT_A_FAILURE_STATE),

    /**
     * The estimator's own frame update returned false. This is its report, not our judgement; the
     * accompanying outcome is {@link ConfidenceOutcome#NOT_PRODUCED}.
     */
    ESTIMATOR_FAILED("estimator_failed", null, Detectability.ONLINE),

    /** Track count below the configured minimum. */
    INSUFFICIENT_FEATURES("insufficient_features", "insufficient tracked features", Detectability.ONLINE),

    /**
     * Inlier count, or the derived inlier ratio, below the configured minimum. Previously a raw
     * count with no verdict attached ({@code COMP-001} §8); now a stated criterion.
     */
    POOR_INLIER_SUPPORT("poor_inlier_support", "poor RANSAC support", Detectability.ONLINE),

    /**
     * A configured residual statistic above its bound. Previously neither detected nor reported;
     * reportable since {@code DEC-VO-001} and detected here.
     *
     * <p><b>Confounded by keyframe age</b> (feature spec FR-032): the residual is measured against
     * the frame-to-keyframe homography, which resets when the keyframe changes, so magnitude is
     * small by construction just after a change and grows as the keyframe ages regardless of
     * tracking quality. Keyframe age is not currently observable. A calibration must not be fitted
     * as though this confound were absent.
     */
    HIGH_REPROJECTION_ERROR("high_reprojection_error", "high reprojection error", Detectability.ONLINE),

    /** Frame-to-frame motion outside configured bounds. <b>Offline only</b> — see class javadoc. */
    INCONSISTENT_MOTION("inconsistent_motion", "inconsistent motion", Detectability.OFFLINE_ONLY),

    /** Composed homography degenerate or ill-conditioned. <b>Offline only</b> — see class javadoc. */
    INVALID_HOMOGRAPHY("invalid_homography", "invalid homography", Detectability.OFFLINE_ONLY),

    /**
     * A signal the calibration requires was absent, so no verdict about quality can be formed.
     *
     * <p>This is a third state, not a bad one: a frame whose signals could not be read must receive
     * neither a confident verdict nor a low one. The score is null. Substituting a "reasonable
     * default" for the absent signal here is the precise mechanism by which an unmeasured frame
     * would become a confident one.
     */
    SIGNALS_UNAVAILABLE("signals_unavailable", null, Detectability.ONLINE),

    /**
     * No named condition fired; the score alone placed the frame below the usable bound. Always
     * accompanied by {@link ConfidenceOutcome#DEGRADED}, never by
     * {@link ConfidenceOutcome#REJECTED}.
     */
    LOW_SCORE("low_score", null, Detectability.NOT_A_FAILURE_STATE);

    /** Where, if anywhere, this reason's condition can be evaluated. */
    public enum Detectability {
        /** Evaluated in the online scorer. */
        ONLINE,
        /**
         * Cannot be evaluated online — the required quantity does not cross the VO boundary.
         * Detected by the offline re-scorer from the run record instead.
         */
        OFFLINE_ONLY,
        /** Not a Principle X failure state; a bookkeeping outcome. */
        NOT_A_FAILURE_STATE
    }

    private final String wireName;
    @Nullable
    private final String principleXState;
    private final Detectability detectability;

    ConfidenceReason(String wireName, @Nullable String principleXState, Detectability detectability) {
        this.wireName = wireName;
        this.principleXState = principleXState;
        this.detectability = detectability;
    }

    /** Lowercase name used in {@code frames.csv} and in the Python re-scorer. */
    public String wireName() {
        return wireName;
    }

    /**
     * The Constitution Principle X failure state this reason reports, or {@code null} if it does
     * not correspond to one. Exposed rather than left in prose so a test can assert that every
     * named state has a code (feature spec SC-005).
     */
    @Nullable
    public String principleXState() {
        return principleXState;
    }

    public Detectability detectability() {
        return detectability;
    }

    /**
     * Whether this reason denotes a named condition rather than a low score.
     *
     * <p>True for every reason except {@link #OK} and {@link #LOW_SCORE}. Note this includes
     * {@link #NOT_ESTABLISHED} and {@link #SIGNALS_UNAVAILABLE}: "cannot be judged" is a named
     * condition and must withhold the estimate, not merely score it low. An unjudgeable estimate
     * consumed as a measurement is precisely what Principle X forbids.
     */
    public boolean isRejectionCondition() {
        return this != OK && this != LOW_SCORE;
    }

    /** Parses a wire name, rejecting anything outside the closed set. */
    public static ConfidenceReason fromWireName(String name) {
        for (ConfidenceReason r : values()) {
            if (r.wireName.equals(name)) {
                return r;
            }
        }
        throw new IllegalArgumentException("Not a valid confidence reason: '" + name + "'");
    }
}
