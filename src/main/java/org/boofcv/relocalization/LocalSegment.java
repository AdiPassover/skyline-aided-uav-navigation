package org.boofcv.relocalization;

/**
 * One VO segment as the alignment layer sees it: the span of frames over which the metric VO's
 * segment-relative position is continuous, i.e. between stitching restarts ({@code DEC-INT-001};
 * design §10).
 *
 * <p>Since the post-merge rebase the segment identity is the VO's own: {@link #segmentId} <b>is</b>
 * {@code MetricNavigationState.getSegmentIndex()}, and {@link NavigationAligner} refuses to continue
 * if the two ever disagree. The metric VO already re-bases its readout at a restart — its
 * {@code segmentRelativePose()} is the position since the segment's origin frame, containing no
 * unestimated interval ({@code DEC-VO-009} D10) — so the external re-basing the P0 layer performed
 * on the pixel readout is no longer needed and no longer exists here. The two gap flags are the
 * VO's ({@code DEC-VO-010} D10): the displacement across the boundary is unknown by construction for
 * a stitching restart, while the orientation is known whenever the heading channel was up.
 *
 * @param segmentId                    the VO segment index (0 for the root)
 * @param originFrameIndex             the frame the segment starts on (the restart frame, or the first)
 * @param originTimestampS             its timestamp
 * @param unknownTranslationGapBefore  the displacement across the boundary that opened this segment
 *                                     was never estimated (always true for a restart)
 * @param headingKnownAcrossGap        an authoritative heading was available at the origin frame
 */
public record LocalSegment(int segmentId, int originFrameIndex, double originTimestampS,
                           boolean unknownTranslationGapBefore, boolean headingKnownAcrossGap) {

    public LocalSegment {
        if (segmentId < 0) {
            throw new IllegalArgumentException("segmentId must be >= 0, got " + segmentId);
        }
    }

    /** The root segment, from the first frame's sample. */
    public static LocalSegment root(LocalPoseSample first) {
        return new LocalSegment(0, first.frameIndex(), first.timestampS(),
                first.unknownTranslationGapBeforeSegment(), first.headingKnownAcrossGap());
    }
}
