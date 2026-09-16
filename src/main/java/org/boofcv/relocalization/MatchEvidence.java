package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * The raw relocalization evidence of design §9.2 for one executed search, kept separate from the
 * lineage the anchor inherits (§9.1). It controls the binary acceptance gate and is logged for later
 * analysis — including the false-accept, harmful-correction and request-attribution questions;
 * <b>none of it is mapped to a probability, a confidence, or a position weight</b>, and the lags of
 * both views are preserved without position semantics ({@code DEC-INT-002}; {@code EXP-SKY-011}
 * R8: two orthogonal lags observe a displacement's direction, not its length).
 *
 * @param queryFrameIndex       the query frame
 * @param queryTimestampS       its timestamp
 * @param queryLocalPosition    the segment-relative position at the query (metres, ENU)
 * @param queryHeadingDeg       the authoritative heading at the query, or {@code NaN}
 * @param cause                 the effective cause the search ran under (after any escalation)
 * @param originCause           the cause that originally armed the request
 * @param originFrame           the frame the request was originally armed on
 * @param escalated             the request was escalated by a later suspect-VO trigger
 * @param matcherVariant        which matcher scored
 * @param fusionRule            the configured dual-view rule
 * @param referenceRegionId     the primary best region (0 in the result it came from)
 * @param bestReferenceId       the primary best region's best reference
 * @param acceptedReferenceId   the accepted reference, or {@code null} when not accepted
 * @param topRegionScore        primary best-region score
 * @param competingRegionScore  best score outside that region, or {@code null}
 * @param regionMargin          their difference, or {@code null}
 * @param winningLagSamples     the North C1 lag of the best reference (0 for C0) — logged, not used
 * @param overlapFraction       North samples compared at that lag
 * @param scoreAtZeroLag        the North score at lag 0 when searched
 * @param northScore            the candidate's North score
 * @param northMargin           the North ranking's region-level margin
 * @param westScore             the candidate's West score, or {@code null}
 * @param westMargin            the West ranking's region-level margin, or {@code null}
 * @param westLagSamples        the West C1 lag of the candidate, or {@code null} — logged, not used
 * @param westAgreement         the two views' top-1s named one region, or {@code null}
 * @param dualStatus            {@link RetrievalResult.DualStatus} of the retrieval
 * @param dualVerdict           {@link AcceptanceGate.DualVerdict} of the decision
 * @param temporalSupportCount  consecutive eligible queries the region has persisted over
 * @param temporalRequired      whether temporal confirmation was consulted for this decision
 * @param accepted              the binary decision
 */
public record MatchEvidence(int queryFrameIndex, double queryTimestampS,
                            PlanarPosition queryLocalPosition, double queryHeadingDeg,
                            SearchScheduler.Cause cause, SearchScheduler.Cause originCause,
                            int originFrame, boolean escalated, String matcherVariant,
                            String fusionRule, int referenceRegionId, int bestReferenceId,
                            @Nullable Integer acceptedReferenceId, double topRegionScore,
                            @Nullable Double competingRegionScore, @Nullable Double regionMargin,
                            int winningLagSamples, double overlapFraction,
                            @Nullable Double scoreAtZeroLag, @Nullable Double northScore,
                            @Nullable Double northMargin, @Nullable Double westScore,
                            @Nullable Double westMargin, @Nullable Integer westLagSamples,
                            @Nullable Boolean westAgreement, String dualStatus, String dualVerdict,
                            int temporalSupportCount, boolean temporalRequired, boolean accepted) {

    public MatchEvidence {
        if (queryLocalPosition == null || cause == null || originCause == null || matcherVariant == null
                || fusionRule == null || dualStatus == null || dualVerdict == null) {
            throw new IllegalArgumentException("position, causes, matcher, rule and statuses are required");
        }
        if (accepted != (acceptedReferenceId != null)) {
            throw new IllegalArgumentException("acceptedReferenceId present iff accepted");
        }
    }
}
