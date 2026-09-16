package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * What is stored about one accepted re-anchor (design §9, §10): the raw quantities the snap-safety
 * characterisation needs, and the applied alignment delta downstream control must not read as
 * motion. <b>Translation only</b> — the persistent heading is the VO's authoritative heading before
 * and after, and no field here can describe a change to it.
 *
 * <p>No match score is converted into a weight and no field is a confidence. The retrieval
 * evidence travels alongside as {@link MatchEvidence}; the position action is what this record is
 * about. Units: metres, ENU.
 *
 * @param frameIndex              the query frame the re-anchor was applied on
 * @param timestampS              its timestamp
 * @param referenceId             the accepted trusted reference
 * @param referencePosition       the stored persistent position of that reference
 * @param relocalizedPosition     the position the query was snapped onto — equal to
 *                                {@code referencePosition} under {@link RelocalizedPoseRule#DISCRETE_REFERENCE}
 * @param queryLocalPosition      the segment-relative position at the query frame
 * @param headingDeg              the authoritative heading at the query frame, unchanged by the
 *                                re-anchor; {@code NaN} when none existed
 * @param globalBefore            the VO-derived persistent position before the snap, or {@code null}
 *                                when the alignment was unknown (re-anchor after a loss)
 * @param globalAfter             the persistent position after the snap — equal to
 *                                {@code relocalizedPosition} by construction
 * @param alignmentBefore         {@code t_e} before, or {@code null} when unknown
 * @param alignmentAfter          {@code t_e_new = p_reloc − p_query}
 * @param appliedDelta            {@code t_new − t_old} — the jump applied to every subsequent
 *                                persistent position; {@code null} when there was no previous alignment
 * @param lineageBefore           the ledger before inheritance
 * @param lineageAfter            the ledger after inheriting the reference's baseline
 * @param segmentId               the VO segment re-anchored
 * @param alignmentEpochId        the new epoch id
 * @param queryReferenceFrameGap  frames between the reference's frame and the query
 * @param queryReferenceTimeGapS  seconds between them
 * @param evidence                the match evidence, or {@code null} when none was supplied
 * @param poseRuleId              which {@link RelocalizedPoseRule} produced the position
 */
public record ReanchorEvent(int frameIndex, double timestampS, int referenceId,
                            PlanarPosition referencePosition, PlanarPosition relocalizedPosition,
                            PlanarPosition queryLocalPosition, double headingDeg,
                            @Nullable PlanarPosition globalBefore, PlanarPosition globalAfter,
                            @Nullable AlignmentTransform alignmentBefore,
                            AlignmentTransform alignmentAfter,
                            @Nullable AlignmentTransform appliedDelta,
                            AnchorLineage.Snapshot lineageBefore,
                            AnchorLineage.Snapshot lineageAfter,
                            int segmentId, int alignmentEpochId,
                            int queryReferenceFrameGap, double queryReferenceTimeGapS,
                            @Nullable MatchEvidence evidence, String poseRuleId) {

    /**
     * Magnitude of the persistent-position jump this re-anchor applied, metres, or {@code null}
     * when there was no valid position before (nothing to jump from).
     */
    @Nullable
    public Double positionJumpM() {
        return globalBefore == null ? null : globalBefore.distanceTo(globalAfter);
    }
}
