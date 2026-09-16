package org.boofcv.relocalization;

import org.boofcv.stitching.metric.HeadingStatus;
import org.boofcv.stitching.metric.HeightStatus;

import javax.annotation.Nullable;

/**
 * The VO accepted a frame but could produce no metric increment for it — no usable height or no
 * authoritative heading at the reference instant ({@code MetricNavigationState.lastFrameUsable()}
 * false) — while the alignment was known. The segment-relative position now hides an unobserved
 * displacement, so the persistent position becomes UNKNOWN exactly as after a hard loss, although
 * the VO segment itself continues. Recorded once per known→unknown transition; further unusable
 * frames while already unknown are counted, not re-recorded.
 *
 * <p>Under the merged VO this arises in practice only before the heading channel's first sample
 * ({@code HeadingStatus.UNAVAILABLE}); a height {@code UNAVAILABLE} needs a non-positive AGL.
 *
 * @param frameIndex              the frame
 * @param timestampS              its timestamp
 * @param segmentId               the VO segment (unchanged by this event)
 * @param heightStatus            the height reading at this frame
 * @param headingStatus           the heading reading at this frame
 * @param lastValidGlobalPosition the persistent position at the last valid frame, if any
 * @param lastValidFrameIndex     that frame, or {@code -1}
 */
public record TranslationDropoutEvent(int frameIndex, double timestampS, int segmentId,
                                      HeightStatus heightStatus, HeadingStatus headingStatus,
                                      @Nullable PlanarPosition lastValidGlobalPosition,
                                      int lastValidFrameIndex) {
}
