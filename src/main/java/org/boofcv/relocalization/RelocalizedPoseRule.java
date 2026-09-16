package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * How the persistent <b>position</b> of an accepted query is obtained from the accepted reference —
 * the one substitution point the design reserves (§8): today {@code p_reloc = p_ref}; if the SKY
 * lane ever validates a query-to-reference relative translation with a range source
 * ({@code EXP-SKY-009} outcome C, conditional), a second implementation returns
 * {@code p_ref + Δp_ref→query}. Either way the {@link NavigationAligner} action stays binary: the
 * whole returned position is taken, or the VO-derived position is kept.
 *
 * <p><b>Position only.</b> The heading is the VO's authoritative navigation heading and is never
 * part of this rule ({@code DEC-VO-010}; the skyline cameras are world-locked, so a C1 lag is never
 * a yaw — {@code DEC-INT-002}). An implementation may read the {@link MatchEvidence} (the C1 lags
 * of both views live there) but the only one that exists ignores it, deliberately: two orthogonal
 * lags observe the <em>direction</em> of a displacement, not its length ({@code EXP-SKY-011} R8),
 * and no lag → position conversion may be added here without a validated range source.
 */
@FunctionalInterface
public interface RelocalizedPoseRule {

    /**
     * @param reference  the accepted trusted reference
     * @param evidence   the match evidence, or {@code null} when acceptance came from a caller that
     *                   had none (tests, forced re-anchors)
     * @param queryLocal the segment-relative position at the query
     * @return the persistent position the query is snapped onto
     */
    PlanarPosition relocalizedPosition(TrustedReference reference, @Nullable MatchEvidence evidence,
                                       PlanarPosition queryLocal);

    /** The MVP: the stored reference position itself, exactly (design §8). */
    RelocalizedPoseRule DISCRETE_REFERENCE = (reference, evidence, queryLocal) -> reference.positionGlobal();

    default String id() {
        return this == DISCRETE_REFERENCE ? "discrete_reference" : getClass().getSimpleName();
    }
}
