package org.boofcv.stitching;

/**
 * The length unit of a navigation pose's {@code x}/{@code y}, made explicit so it cannot be assumed.
 *
 * <p>{@link org.boofcv.util.structs.Pose3D} carries four doubles and no unit. That was harmless
 * while every pose in the system was in the same arbitrary unit; it stopped being harmless when
 * {@code DEC-VO-007}'s metric readout was ported into the runtime, because from that point two
 * poses with identical field names can differ by a factor of the ground sampling distance — roughly
 * x12 on the geometry {@code EXP-VO-014} was captured at. A waypoint tolerance of {@code 2.0} means
 * "about two pixels" under one and "two metres" under the other, and nothing in the type system
 * would have said so.
 *
 * <p>This enum is that missing statement. It is declared by the producer
 * ({@link NavigationSource#units()}) and by every consumer that holds a distance constant
 * ({@code MissionStrategy.declaredUnits()}), and a mismatch is refused rather than run.
 *
 * <p>There is deliberately no conversion function here. Converting requires a height and a focal
 * length, which is the entire content of {@code DEC-VO-007} D3; offering a one-liner would invite
 * exactly the silent rescaling this type exists to prevent.
 */
public enum NavigationUnits {

    /**
     * Pixels of the run's first frame. What {@code MOSAIC_LEGACY}, {@code LOGICAL_FRAME} and
     * {@code RIGID_MOTION} publish, and what every pre-2026-09-06 controller gain, mission
     * threshold and committed run record is expressed in ({@code COMP-001} section 7:
     * "Every position number the system currently produces is in arbitrary units").
     */
    IMAGE_PIXELS,

    /**
     * Metres on the ground, east/north, relative to the run's first frame. What
     * {@link NavigationSource#METRIC_LOCAL} publishes. Local only — not geodetic, not a global
     * origin, not north-referenced ({@code DEC-VO-007} D9).
     */
    METRES;

    /** Short stable identifier for logs, headers and provenance: {@code px} / {@code m}. */
    public String symbol() {
        return this == METRES ? "m" : "px";
    }
}
