package org.boofcv.relocalization;

import org.boofcv.stitching.metric.HeadingStatus;

import javax.annotation.Nullable;

/**
 * The trusted-reference record (design §6.1), on the post-merge pose contract. Sparse appearance
 * memory, not a landmark map: descriptors, the persistent position and heading at creation, and
 * the provenance needed to inherit lineage on acceptance and to reconstruct it afterwards.
 *
 * <p>Instances are created only by {@link ReferenceMemory} through its insertion policy, so a
 * {@code TrustedReference} is trusted by construction: created with a valid North skyline, a valid
 * persistent <em>position</em>, clean effective lineage, and inside the conservative exposure bound.
 *
 * <h2>Pose semantics — explicit, and refused when they do not match</h2>
 *
 * <p>{@link #positionGlobal} is metres, East/North, in the persistent frame rooted at the run's
 * first frame; {@link #headingDeg} is the VO's authoritative navigation heading at storage —
 * degrees clockwise from North ({@code DEC-VO-010} D2), {@code NaN} when none existed. Nothing
 * pixel-valued and no start-relative visual yaw may enter this record. {@link #poseSchema} names
 * that contract and the constructor refuses any other value, so a record written under the P0
 * pixel / visual-yaw contract cannot be silently reinterpreted as a metric one.
 *
 * <h2>Provenance for the poisoning question (design §9; {@code EXP-SKY-011} follow-up)</h2>
 *
 * <p>{@link #sourceAnchorId} is the anchor reference the navigation was aligned to when this one
 * was stored ({@code null} = the root lineage, never re-anchored), and
 * {@link #sourceReanchorFrameIndex} is the frame of the accepted {@link ReanchorEvent} that
 * established that lineage. Following {@code sourceAnchorId} through the memory reconstructs the
 * whole chain; {@link #descendsFromRelocalization()} is the one-bit question. No quarantine or
 * staged-trust policy is applied here — the decision is pending.
 *
 * @param id                       memory-local sequential id, chronological
 * @param frameIndex               the frame it was observed on
 * @param timestampS               its timestamp
 * @param northDescriptor          the North skyline descriptor (required)
 * @param westDescriptor           the West skyline descriptor, or {@code null} when the West view
 *                                 was unavailable or invalid at storage
 * @param positionGlobal           the persistent position at creation (metres, ENU)
 * @param headingDeg               the authoritative heading at creation, or {@code NaN}
 * @param headingStatus            the heading freshness at creation
 * @param segmentId                the VO segment it was created in
 * @param alignmentEpochId         the alignment epoch it was created in
 * @param sourceAnchorId           the anchor reference active at creation, or {@code null} for the root
 * @param sourceReanchorFrameIndex the frame of the re-anchor that established the lineage, or {@code null}
 * @param baselineExposure         {@code E_eff} at storage — provenance, not metres of error
 * @param baselineInstability      {@code I_eff} at storage — always {@code false} under the MVP policy
 * @param poseSchema               must equal {@link #POSE_SCHEMA}
 */
public record TrustedReference(int id, int frameIndex, double timestampS,
                               SkylineDescriptor northDescriptor,
                               @Nullable SkylineDescriptor westDescriptor,
                               PlanarPosition positionGlobal, double headingDeg,
                               HeadingStatus headingStatus, int segmentId, int alignmentEpochId,
                               @Nullable Integer sourceAnchorId,
                               @Nullable Integer sourceReanchorFrameIndex,
                               long baselineExposure, boolean baselineInstability,
                               String poseSchema) {

    /**
     * The one pose contract this record accepts. Anything else — in particular the P0 layer's
     * first-frame-pixel positions with a start-relative visual yaw — is refused at construction.
     */
    public static final String POSE_SCHEMA =
            "int-reference-pose/2: position metres ENU in the persistent frame rooted at the run's "
            + "first frame; heading degrees CW from North (DEC-VO-010), NaN when unavailable";

    public TrustedReference {
        if (northDescriptor == null || positionGlobal == null || headingStatus == null) {
            throw new IllegalArgumentException("northDescriptor, positionGlobal and headingStatus are required");
        }
        if (baselineExposure < 0) {
            throw new IllegalArgumentException("baselineExposure must be >= 0");
        }
        if (!POSE_SCHEMA.equals(poseSchema)) {
            throw new IllegalArgumentException("incompatible reference pose record: schema '"
                    + poseSchema + "' is not '" + POSE_SCHEMA + "'. Legacy pixel / visual-yaw "
                    + "references are refused, never reinterpreted.");
        }
        if (headingStatus.usable() != Double.isFinite(headingDeg)) {
            throw new IllegalArgumentException("heading validity and the heading value disagree: "
                    + headingStatus + " / " + headingDeg);
        }
        if ((sourceAnchorId == null) != (sourceReanchorFrameIndex == null)) {
            throw new IllegalArgumentException("sourceAnchorId and sourceReanchorFrameIndex are set together");
        }
    }

    /** True when a West descriptor is stored. */
    public boolean hasWest() {
        return westDescriptor != null;
    }

    /** True when the stored heading is an authoritative one. */
    public boolean headingValid() {
        return headingStatus.usable() && Double.isFinite(headingDeg);
    }

    /** True when this reference was stored under a lineage that passed through an accepted re-anchor. */
    public boolean descendsFromRelocalization() {
        return sourceAnchorId != null;
    }
}
