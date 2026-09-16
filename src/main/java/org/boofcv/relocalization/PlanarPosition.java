package org.boofcv.relocalization;

import org.boofcv.util.structs.Pose3D;

/**
 * A horizontal position in <b>metres</b>, <b>East / North</b> axes (ENU), in whichever frame the
 * owner declares — the segment-relative frame for a local position, the persistent frame (rooted at
 * the run's first frame) for a global one. The one unit/axis convention of the relocalization layer
 * after the post-merge rebase ({@code DEC-INT-001} amendment 2026-09-07): there is no pixel position
 * anywhere in this package.
 *
 * <p>Deliberately not a {@link Pose3D}: a position carries no heading, and the heading of the
 * persistent pose is the VO's authoritative navigation heading ({@code DEC-VO-010}), which the
 * alignment never touches. {@link #toPose(double)} attaches one only at the boundary where a
 * consumer needs the pair.
 */
public record PlanarPosition(double eastM, double northM) {

    public static final PlanarPosition ORIGIN = new PlanarPosition(0.0, 0.0);

    public PlanarPosition {
        if (!Double.isFinite(eastM) || !Double.isFinite(northM)) {
            throw new IllegalArgumentException("a position must be finite: east=" + eastM
                    + " north=" + northM);
        }
        // Record equality is Double.compare's, under which -0.0 != 0.0; a segment origin read as
        // -(0.0 - 0.0) must still be the origin.
        eastM = eastM + 0.0;
        northM = northM + 0.0;
    }

    /** The {@code x}/{@code y} of a metric pose read as east/north ({@code MetricNavigationState}'s convention). */
    public static PlanarPosition of(Pose3D metricPose) {
        if (metricPose == null) {
            throw new IllegalArgumentException("pose is required");
        }
        return new PlanarPosition(metricPose.x, metricPose.y);
    }

    public PlanarPosition plus(PlanarPosition d) {
        return new PlanarPosition(eastM + d.eastM, northM + d.northM);
    }

    public PlanarPosition minus(PlanarPosition o) {
        return new PlanarPosition(eastM - o.eastM, northM - o.northM);
    }

    public double distanceTo(PlanarPosition o) {
        return Math.hypot(eastM - o.eastM, northM - o.northM);
    }

    public double norm() {
        return Math.hypot(eastM, northM);
    }

    /**
     * {@code (east, north, 0, heading)} as a {@link Pose3D}. {@code headingDeg} may be {@code NaN}
     * when no authoritative heading exists — {@code Pose3D} passes it through unchanged — so a
     * caller that needs a heading must check validity first; this class does not invent one.
     */
    public Pose3D toPose(double headingDeg) {
        return new Pose3D(eastM, northM, 0.0, headingDeg);
    }

    @Override
    public String toString() {
        return String.format("(E=%.3f m, N=%.3f m)", eastM, northM);
    }
}
