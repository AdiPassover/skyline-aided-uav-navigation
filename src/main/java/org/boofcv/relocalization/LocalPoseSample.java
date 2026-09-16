package org.boofcv.relocalization;

import org.boofcv.stitching.metric.HeadingStatus;
import org.boofcv.stitching.metric.HeightStatus;
import org.boofcv.stitching.metric.MetricNavigationState;
import org.boofcv.util.structs.Pose3D;

/**
 * One frame of the VO → INT handoff — <b>the</b> input of the relocalization layer after the
 * post-merge rebase ({@code DEC-INT-001} amendment 2026-09-07).
 *
 * <pre>
 *   L_t = ( x_segment [m, East], y_segment [m, North], psi_nav [deg CW from North] ) + validity
 * </pre>
 *
 * <p>Everything here is read from {@link MetricNavigationState} through {@link #fromMetric} and
 * nothing is derived: the position is {@code segmentRelativePose()} (metres, ENU, relative to the
 * current VO segment's origin — the pose containing no unestimated interval), the heading is
 * {@code navHeadingDeg()} (a sensor measurement, {@code NaN} until the first heading sample), and
 * the validity fields are the VO's own ({@code HeadingStatus}, {@code HeightStatus},
 * {@code lastFrameUsable()}, the segment index and the two gap flags of {@code DEC-VO-010} D10).
 *
 * <p><b>There is no pixel path.</b> The layer refuses a metric state whose direction is the
 * visually integrated yaw ({@code isHeadingAuthoritative() == false}): under that arm the East/North
 * axes drift with the visual heading and a translation-only alignment is undefined. The visual yaw
 * is carried as a diagnostic only.
 *
 * @param frameIndex                         the frame, strictly increasing
 * @param timestampS                         its timestamp on the channels' clock
 * @param voSuccess                          the {@code processFrame} boolean
 * @param voSegmentIndex                     {@code MetricNavigationState.getSegmentIndex()}
 * @param unknownTranslationGapBeforeSegment {@code isUnknownTranslationGapBeforeSegment()}
 * @param headingKnownAcrossGap              {@code isHeadingKnownAcrossGap()}
 * @param segmentPosition                    segment-relative East/North, metres
 * @param translationUsable                  {@code lastFrameUsable()}: this frame's increment was
 *                                           produced (always true on a reference/origin frame with
 *                                           a usable height); false means the segment position now
 *                                           hides an unobserved displacement
 * @param heightStatus                       the height reading at this frame's own time
 * @param headingDeg                         {@code navHeadingDeg()}: degrees CW from North in
 *                                           {@code [0, 360)}, or {@code NaN} when unavailable
 * @param headingStatus                      {@code currentHeadingStatus()}
 * @param visualYawDeg                       the raw rigid readout's yaw, start-frame datum —
 *                                           diagnostic only, never a navigation heading
 */
public record LocalPoseSample(int frameIndex, double timestampS, boolean voSuccess,
                              int voSegmentIndex, boolean unknownTranslationGapBeforeSegment,
                              boolean headingKnownAcrossGap, PlanarPosition segmentPosition,
                              boolean translationUsable, HeightStatus heightStatus,
                              double headingDeg, HeadingStatus headingStatus, double visualYawDeg) {

    public LocalPoseSample {
        if (segmentPosition == null || heightStatus == null || headingStatus == null) {
            throw new IllegalArgumentException("segmentPosition, heightStatus and headingStatus are required");
        }
        if (voSegmentIndex < 0) {
            throw new IllegalArgumentException("voSegmentIndex must be >= 0");
        }
        if (headingStatus.usable() && !Double.isFinite(headingDeg)) {
            throw new IllegalArgumentException("heading status " + headingStatus
                    + " says usable but the heading is " + headingDeg);
        }
        if (!headingStatus.usable()) {
            // No authoritative heading exists for this frame. Whatever number came with the
            // status is not one, and must not survive into a pose.
            headingDeg = Double.NaN;
        }
    }

    /**
     * Reads one frame from the estimator's metric state, after {@code processFrame}.
     *
     * @throws IllegalStateException when the state's direction is the visual yaw rather than an
     *                               external heading — the layer has no contract for that arm
     */
    public static LocalPoseSample fromMetric(int frameIndex, double timestampS, boolean voSuccess,
                                             MetricNavigationState m, Pose3D rigidPose) {
        if (m == null) {
            throw new IllegalArgumentException("a MetricNavigationState is required: the "
                    + "relocalization layer consumes the metric segment-relative pose and nothing else");
        }
        if (!m.isHeadingAuthoritative()) {
            throw new IllegalStateException("the metric state rotates increments by the VISUAL yaw "
                    + "(no heading channel); the relocalization layer requires the authoritative "
                    + "external heading of DEC-VO-010, without which its East/North axes are not "
                    + "North-referenced and a translation-only alignment is undefined");
        }
        Pose3D seg = m.segmentRelativePose();
        org.boofcv.stitching.metric.HeightReading h = m.getReferenceHeight();
        return new LocalPoseSample(frameIndex, timestampS, voSuccess, m.getSegmentIndex(),
                m.isUnknownTranslationGapBeforeSegment(), m.isHeadingKnownAcrossGap(),
                new PlanarPosition(seg.x, seg.y), m.lastFrameUsable(),
                h == null ? HeightStatus.UNAVAILABLE : h.status(),
                m.navHeadingDeg(), m.currentHeadingStatus(),
                rigidPose == null ? Double.NaN : rigidPose.yaw);
    }

    /** True when an authoritative navigation heading exists for this frame. */
    public boolean headingValid() {
        return headingStatus.usable() && Double.isFinite(headingDeg);
    }
}
