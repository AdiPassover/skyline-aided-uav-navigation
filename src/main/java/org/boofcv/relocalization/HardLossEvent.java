package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * A hard VO loss: the frozen estimator failed and restarted on this frame, and the metric VO opened
 * a new segment whose preceding displacement was never estimated (design §10; {@code DEC-VO-010}
 * D10).
 *
 * <p>The translation across the gap is UNKNOWN and nothing here may be read as a bridge. The
 * closed segment's last valid persistent position is kept for the record only. Whether the
 * <em>heading</em> is known across the gap is a separate fact — it is the external channel's,
 * not the visual pipeline's — and is carried here so a reader never infers one from the other.
 *
 * @param frameIndex              the restart frame (also the new segment's origin)
 * @param timestampS              its timestamp
 * @param closedSegmentId         the VO segment that ended
 * @param newSegmentId            the VO segment that begins on this frame
 * @param lastValidGlobalPosition the closed segment's persistent position at its last valid frame,
 *                                or {@code null} if it never had one
 * @param lastValidFrameIndex     the frame that position belongs to, or {@code -1}
 * @param headingKnownAcrossGap   the authoritative heading was available at the restart frame
 */
public record HardLossEvent(int frameIndex, double timestampS, int closedSegmentId,
                            int newSegmentId, @Nullable PlanarPosition lastValidGlobalPosition,
                            int lastValidFrameIndex, boolean headingKnownAcrossGap) {
}
