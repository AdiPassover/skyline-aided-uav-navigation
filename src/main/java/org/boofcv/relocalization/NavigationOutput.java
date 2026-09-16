package org.boofcv.relocalization;

import org.boofcv.stitching.metric.HeadingStatus;
import org.boofcv.util.structs.Pose3D;

import javax.annotation.Nullable;

import java.util.Optional;

/**
 * The alignment layer's per-frame output — the navigation interface of design §10, with
 * <b>position validity and heading validity kept separate</b> (post-merge rebase, 2026-09-07).
 *
 * <p>Two independent facts describe a frame:
 * <ul>
 *   <li>{@link #globalPositionValid()} — the persistent East/North position is known
 *       ({@code p_segment + t_e} with {@code t_e} known). False from a hard loss or a translation
 *       dropout until a trusted reference is accepted.</li>
 *   <li>{@link #headingValid()} — the VO's authoritative navigation heading exists for this frame
 *       ({@code DEC-VO-010}). Independent of the visual pipeline: a hard loss does not touch it,
 *       and no heading loss ever promotes the visual yaw.</li>
 * </ul>
 * {@link #globalPoseValid()} is their conjunction, kept for consumers that need the pair.
 *
 * <p>The accessors are structural: {@link #globalPosition()} and {@link #globalPose()} are
 * {@link Optional}s that are empty exactly when the corresponding validity is false, so a consumer
 * cannot take the segment-local position, or a {@code NaN} heading, for a persistent pose by
 * accident. Units: metres, ENU ({@link PlanarPosition}); degrees clockwise from North.
 *
 * @param frameIndex           the frame this output describes
 * @param timestampS           its timestamp
 * @param voSuccess            the VO's {@code processFrame} result for this frame
 * @param localPosition        the segment-relative position (metres, ENU)
 * @param translationUsable    the VO produced this frame's metric increment
 * @param headingDeg           the authoritative heading, or {@code NaN} when {@code headingStatus} is not usable
 * @param headingStatus        the VO's heading freshness for this frame
 * @param globalPositionOrNull {@code p_segment + t_e} when {@code t_e} is known, else {@code null}
 * @param segmentId            the VO segment index
 * @param alignmentEpochId     increments at every re-anchor, hard loss and translation dropout
 * @param lineage              the ledger at this frame
 * @param event                what happened on this frame
 * @param reanchorEvent        the re-anchor applied on this frame, if any
 */
public record NavigationOutput(int frameIndex, double timestampS, boolean voSuccess,
                               PlanarPosition localPosition, boolean translationUsable,
                               double headingDeg, HeadingStatus headingStatus,
                               @Nullable PlanarPosition globalPositionOrNull,
                               int segmentId, int alignmentEpochId,
                               AnchorLineage.Snapshot lineage, FrameEvent event,
                               @Nullable ReanchorEvent reanchorEvent) {

    /** What happened on a frame, for the per-frame log. */
    public enum FrameEvent {
        /** The first frame: root segment opened, alignment identity, position valid. */
        INIT,
        /** A usable metric increment composed normally. */
        NONE,
        /** The VO failed and restarted: new VO segment, alignment UNKNOWN. */
        HARD_LOSS,
        /**
         * The VO accepted the frame but produced no metric increment (no usable height or heading
         * at the reference instant): the segment position now hides an unobserved displacement, so
         * the alignment is UNKNOWN. The VO segment is unchanged.
         */
        TRANSLATION_DROPOUT,
        /** A trusted reference was accepted on this frame: alignment replaced. */
        REANCHOR;

        public String wireName() {
            return name().toLowerCase();
        }
    }

    public NavigationOutput {
        if (localPosition == null || headingStatus == null) {
            throw new IllegalArgumentException("localPosition and headingStatus are required");
        }
        if (lineage == null || event == null) {
            throw new IllegalArgumentException("lineage and event are required");
        }
        if (event == FrameEvent.REANCHOR && reanchorEvent == null) {
            throw new IllegalArgumentException("a REANCHOR frame must carry its ReanchorEvent");
        }
        if (headingStatus.usable() != Double.isFinite(headingDeg)) {
            throw new IllegalArgumentException("heading validity and the heading value disagree: status "
                    + headingStatus + ", heading " + headingDeg);
        }
    }

    /** The persistent East/North position, present only while the alignment is known. */
    public Optional<PlanarPosition> globalPosition() {
        return Optional.ofNullable(globalPositionOrNull);
    }

    /** False from a hard loss or dropout until a trusted reference is accepted. */
    public boolean globalPositionValid() {
        return globalPositionOrNull != null;
    }

    /** True when an authoritative navigation heading exists for this frame. */
    public boolean headingValid() {
        return headingStatus.usable() && Double.isFinite(headingDeg);
    }

    /** The authoritative heading, present only when valid. */
    public Optional<Double> heading() {
        return headingValid() ? Optional.of(headingDeg) : Optional.empty();
    }

    /** {@code global_position_valid AND heading_valid} — the aggregate compatibility fact. */
    public boolean globalPoseValid() {
        return globalPositionValid() && headingValid();
    }

    /**
     * The persistent pose (east, north, 0, heading), present only when <em>both</em> the position
     * and the heading are valid. There is no accessor that pairs a valid position with an invalid
     * heading, or vice versa.
     */
    public Optional<Pose3D> globalPose() {
        return globalPoseValid() ? Optional.of(globalPositionOrNull.toPose(headingDeg)) : Optional.empty();
    }

    /** The segment-local pose with the authoritative heading, present only when the heading is valid. */
    public Optional<Pose3D> localPose() {
        return headingValid() ? Optional.of(localPosition.toPose(headingDeg)) : Optional.empty();
    }
}
