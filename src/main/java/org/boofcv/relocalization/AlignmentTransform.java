package org.boofcv.relocalization;

import org.boofcv.util.structs.Pose3D;

/**
 * The alignment {@code t_e} between a VO segment's metric frame and the persistent frame — a
 * <b>pure horizontal translation in metres, East/North</b> ({@code DEC-INT-001}, amendment of
 * 2026-09-07).
 *
 * <pre>
 *   p_global(k) = p_segment(k) + t_e
 *   re-anchor:   t_e_new = p_reference − p_segment_query        so that  p_global_query = p_reference
 * </pre>
 *
 * <h2>Why translation only</h2>
 *
 * <p>The merged VO publishes the segment-relative position in metres (height owns the scale,
 * {@code DEC-VO-009}) rotated into ENU by an <b>authoritative North-referenced heading</b> that is
 * measured, not integrated, and survives a visual restart ({@code DEC-VO-010} D10). After an
 * ordinary hard loss the only unknown that remains in the persistent frame is therefore the
 * horizontal displacement across the failed interval; neither scale nor rotation is an alignment
 * unknown, and a one-correspondence snap determines exactly the two parameters this class holds.
 * The SE(2) container the P0 layer used against the pixel / visual-yaw pose was retired rather than
 * kept with a rotation forced to identity: a degree of freedom that must never move is safest when
 * it does not exist.
 *
 * <p>Consequences pinned by test: {@link #apply(Pose3D)} never changes {@code yaw} or {@code z};
 * {@link #reanchor} cannot express a rotation; {@code reanchor(ref, q).apply(q) == ref} exactly.
 *
 * <p>Immutable. Units: metres, ENU, always.
 */
public final class AlignmentTransform {

    /** The identity: persistent frame ≡ segment frame. The root segment's alignment by convention. */
    public static final AlignmentTransform IDENTITY = new AlignmentTransform(0.0, 0.0);

    private final double tE;
    private final double tN;

    private AlignmentTransform(double tE, double tN) {
        if (!Double.isFinite(tE) || !Double.isFinite(tN)) {
            throw new IllegalArgumentException("AlignmentTransform components must be finite: "
                    + tE + ", " + tN);
        }
        this.tE = tE;
        this.tN = tN;
    }

    /** From an East/North translation in metres. */
    public static AlignmentTransform of(double tE, double tN) {
        return new AlignmentTransform(tE, tN);
    }

    /**
     * {@code p_reference − p_query}: the alignment under which the query's segment position maps
     * exactly onto the reference's persistent position (design §3; {@code DEC-INT-001}).
     */
    public static AlignmentTransform reanchor(PlanarPosition referencePersistent, PlanarPosition queryLocal) {
        if (referencePersistent == null || queryLocal == null) {
            throw new IllegalArgumentException("both positions are required");
        }
        return new AlignmentTransform(referencePersistent.eastM() - queryLocal.eastM(),
                referencePersistent.northM() - queryLocal.northM());
    }

    /** The persistent position of a segment-relative position. */
    public PlanarPosition apply(PlanarPosition local) {
        if (local == null) {
            throw new IllegalArgumentException("local position is required");
        }
        return new PlanarPosition(local.eastM() + tE, local.northM() + tN);
    }

    /**
     * Shifts {@code x}/{@code y} (east/north metres) and leaves {@code z} and {@code yaw}
     * <b>untouched</b> — the heading is the VO's authoritative navigation heading and is not part of
     * the alignment.
     */
    public Pose3D apply(Pose3D localPose) {
        if (localPose == null) {
            throw new IllegalArgumentException("local pose is required");
        }
        return new Pose3D(localPose.x + tE, localPose.y + tN, localPose.z, localPose.yaw);
    }

    /** {@code this ∘ inner}: translations add. */
    public AlignmentTransform compose(AlignmentTransform inner) {
        if (inner == null) {
            throw new IllegalArgumentException("inner is required");
        }
        return new AlignmentTransform(tE + inner.tE, tN + inner.tN);
    }

    public AlignmentTransform inverse() {
        return new AlignmentTransform(-tE, -tN);
    }

    /**
     * {@code this ∘ other⁻¹}: the jump a re-anchor applies to every subsequent persistent position
     * — what downstream control must not read as motion (design §10).
     */
    public AlignmentTransform deltaFrom(AlignmentTransform other) {
        return compose(other.inverse());
    }

    public double tE() {
        return tE;
    }

    public double tN() {
        return tN;
    }

    /** Magnitude of the translation, metres. */
    public double norm() {
        return Math.hypot(tE, tN);
    }

    /** Whether this is the identity to within {@code tol} metres. */
    public boolean isIdentity(double tol) {
        return Math.abs(tE) <= tol && Math.abs(tN) <= tol;
    }

    /**
     * Wraps an angle difference into {@code (-180, 180]}. Retained for reporting heading
     * differences (snap safety, diagnostics): {@link Pose3D}'s constructor normalises every yaw into
     * {@code [0, 360)}, which turns a −2° difference into 358°, so a signed delta must never be
     * routed through a {@code Pose3D}.
     */
    public static double wrapDegrees(double deg) {
        double d = ((deg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
        return d == -180.0 ? 180.0 : d;
    }

    @Override
    public String toString() {
        return String.format("AlignmentTransform[tE=%.6f m tN=%.6f m]", tE, tN);
    }
}
