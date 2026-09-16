package org.boofcv.stitching;

import georegression.struct.InvertibleTransform;
import georegression.struct.point.Point2D_F64;
import lombok.Getter;
import org.boofcv.util.structs.Pose3D;

/**
 * Navigation state built from the <b>rigid part of each frame's image motion only</b>
 * ({@code DEC-VO-004} Alternative E).
 *
 * <p>Where {@link LogicalNavigationState} composes the estimator's transform in full — exactly, and
 * therefore including the scale, shear and perspective the estimator gets wrong on real imagery
 * ({@code EXP-VO-004}) — this class integrates only the rotation and translation, and <b>observes</b>
 * everything else without applying it:
 *
 * <pre>
 *   per-frame transform D_k : C_{k-1} → C_k
 *         ├─ rigid rotation + centre translation ──► logical x, y, yaw      (navigation state)
 *         ├─ uniform scale √|det J|              ──► accumulated diagnostic (NOT applied)
 *         ├─ anisotropy / stretch axis           ──► diagnostic             (NOT applied)
 *         └─ perspective magnitude               ──► diagnostic             (NOT applied)
 * </pre>
 *
 * <h2>The update, and why the pivot is the image centre</h2>
 *
 * <p>The state is the rigid map {@code M_k : C_k → L} of the current camera frame into the fixed
 * logical frame, held as a heading {@code θ_k} and the logical displacement {@code T_k} of the
 * <b>image centre</b> {@code c}:
 *
 * <pre>
 *   M_k(p) = R(θ_k)·(p − c) + c + T_k
 * </pre>
 *
 * <p>Given the inverse per-frame transform {@code D_k⁻¹ : C_k → C_{k-1}} (built from the estimator's
 * own accumulation, see {@link #observe}), the update is
 *
 * <pre>
 *   q     = D_k⁻¹(c)                      the previous-frame pixel now under the optical axis
 *   T_k   = R(θ_{k-1})·(q − c) + T_{k-1}
 *   θ_k   = θ_{k-1} + polarRotation(J_{D_k⁻¹}(c))
 * </pre>
 *
 * <p><b>Both halves pivot on the image centre, and that is the correctness requirement, not a
 * detail.</b> Under a rotation about a point, only that point is fixed; taking the translation
 * column of a matrix (which describes the motion of the image <i>origin</i>) manufactures
 * translation out of pure rotation — measured at 117 px for a 30° rotation of a 320 px frame
 * ({@code DEC-VO-003}). Absent calibration the image centre is the standard approximation to the
 * principal point, i.e. the ground point the optical axis is looking at, so {@code T_k} is the
 * displacement of the point the camera is actually pointing at. The same reasoning applies to the
 * Jacobian: for a homography it varies across the image, and the centre is where it is evaluated.
 *
 * <h2>Reference changes</h2>
 *
 * <p>The state never sees an absolute transform, only increments, so it has <b>no anchor to fold and
 * nothing to reset</b>. When the estimator zeroes its accumulation (a mosaic canvas re-origin, or a
 * restart), the caller passes an identity previous-transform for the next frame and the increment
 * is simply the new accumulation — the arithmetic is identical either way. That is why this readout
 * is independent of the canvas schedule by construction rather than by compensation.
 *
 * <p><b>Rejected estimates are never integrated:</b> the caller only calls {@link #observe} on a
 * frame the stitcher accepted, and passes the last <i>good</i> accumulation as the previous
 * transform ({@code EXP-VO-004} R0 — BoofCV writes {@code worldToCurr} before
 * {@code checkLargeMotion} can reject the frame).
 *
 * <h2>What is deliberately not done here</h2>
 *
 * <p>The accumulated uniform scale is maintained and reported but <b>never applied to x/y</b>, and is
 * <b>not</b> altitude: on real imagery it carries a scene-induced bias of ≈0.03 %/frame that swamps
 * any height signal ({@code EXP-VO-004}, {@code EXP-VO-005}). Converting it to height, fusing it with
 * an altitude source, and metric scale are all out of scope by {@code DEC-VO-004}.
 *
 * @param <IT> the 2D motion model
 */
public final class RigidNavigationState<IT extends InvertibleTransform<IT>> {

    private final MotionModelSupport<IT> motionModel;

    /** {@code D_k⁻¹ : C_k → C_{k-1}}, rebuilt every frame. */
    private final IT incrementCurrentToPrevious;
    private final IT scratchInverse;

    private final RigidMotionDecomposition decomposition = new RigidMotionDecomposition();
    private final Point2D_F64 mappedCentre = new Point2D_F64();
    private final double[] jacobian = new double[4];

    // --- navigation state ---
    private double headingRad;
    private double centreX;      // T_k, image convention (y down), logical pixels
    private double centreY;
    private Pose3D pose = new Pose3D();

    // --- observed-but-not-applied geometric state ---
    @Getter private double accumulatedLogScale;          // Σ log √|det J| over accepted frames
    @Getter private double incrementScale = 1.0;         // this frame's √|det J|
    @Getter private double incrementLogScale;            // this frame's log √|det J|
    @Getter private double incrementAnisotropy = 1.0;    // this frame's σ₁/σ₂
    @Getter private double incrementDeformation;         // this frame's |log anisotropy|/√2
    @Getter private double incrementStretchAxisRad;
    @Getter private double incrementPerspective;         // 0 for affine, by construction
    @Getter private double incrementFlowPx;              // |D_k⁻¹(c) − c|, the raw per-frame flow
    @Getter private double incrementRotationDeg;

    /**
     * {@code (q_k − c).x} — the signed per-frame displacement of the image centre, in
     * <b>reference-frame ({@code k−1}) pixels</b>, before the heading rotation is applied.
     *
     * <p>Exposed because it is the exact quantity {@code DEC-VO-007} D3's metric conversion
     * multiplies by the ground sampling distance, and the metric layer must consume <i>this</i>
     * number rather than re-derive it — a second derivation is a second chance to disagree.
     * {@link #getIncrementFlowPx()} is its magnitude and loses the direction, so it cannot serve.
     */
    @Getter private double incrementDqX;

    /** {@code (q_k − c).y}, reference-frame pixels, image convention (y down). See {@link #getIncrementDqX()}. */
    @Getter private double incrementDqY;

    /**
     * This frame's polar rotation in <b>radians</b> — the unrounded value
     * {@link #getIncrementRotationDeg()} reports in degrees, and the one the metric integrator adds
     * to its own heading so that both states integrate bit-identical yaw.
     */
    @Getter private double incrementRotationRad;
    @Getter private boolean incrementProper = true;      // false ⇒ det ≤ 0, a mirrored/singular fit
    @Getter private long improperIncrementCount;

    private boolean observed;

    public RigidNavigationState(MotionModelSupport<IT> motionModel, IT prototype) {
        if (motionModel == null || prototype == null) {
            throw new IllegalArgumentException("motionModel and prototype are required");
        }
        this.motionModel = motionModel;
        this.incrementCurrentToPrevious = prototype.createInstance();
        this.scratchInverse = prototype.createInstance();
        reset();
    }

    /** Returns to the state of a fresh run: origin and heading zero at the next observed frame. */
    public void reset() {
        incrementCurrentToPrevious.reset();
        scratchInverse.reset();
        headingRad = 0.0;
        centreX = 0.0;
        centreY = 0.0;
        accumulatedLogScale = 0.0;
        incrementScale = 1.0;
        incrementLogScale = 0.0;
        incrementAnisotropy = 1.0;
        incrementDeformation = 0.0;
        incrementStretchAxisRad = 0.0;
        incrementPerspective = 0.0;
        incrementFlowPx = 0.0;
        incrementRotationDeg = 0.0;
        incrementDqX = 0.0;
        incrementDqY = 0.0;
        incrementRotationRad = 0.0;
        incrementProper = true;
        improperIncrementCount = 0;
        pose = new Pose3D();
        observed = false;
    }

    /**
     * Integrates one <b>accepted</b> frame.
     *
     * @param previousFirstToCurrent {@code F_{k-1}} — the estimator's accumulation after the last
     *                               accepted frame, or identity immediately after any operation
     *                               that zeroed it (canvas re-origin, restart, first frame)
     * @param currentFirstToCurrent  {@code F_k} — the accumulation for this frame
     * @param frameWidth             current frame width in pixels
     * @param frameHeight            current frame height in pixels
     */
    public void observe(IT previousFirstToCurrent, IT currentFirstToCurrent,
                        int frameWidth, int frameHeight) {
        double cx = frameWidth / 2.0, cy = frameHeight / 2.0;

        // D_k⁻¹ : C_k → C_{k-1}   =   p ↦ F_{k-1}(F_k⁻¹(p))
        // georegression: a.concat(b, r) ⇒ r(p) = b(a(p)).
        IT currentInverse = currentFirstToCurrent.invert(scratchInverse);
        currentInverse.concat(previousFirstToCurrent, incrementCurrentToPrevious);

        motionModel.apply(incrementCurrentToPrevious, cx, cy, mappedCentre);
        motionModel.jacobian(incrementCurrentToPrevious, cx, cy, jacobian);
        decomposition.set(jacobian[0], jacobian[1], jacobian[2], jacobian[3]);

        // --- navigation: rigid part only ---
        double qx = mappedCentre.x - cx, qy = mappedCentre.y - cy;
        double cos = Math.cos(headingRad), sin = Math.sin(headingRad);
        centreX += cos * qx - sin * qy;
        centreY += sin * qx + cos * qy;
        headingRad += decomposition.getRotationRad();

        // --- observed, not applied ---
        incrementScale = decomposition.getUniformScale();
        incrementLogScale = decomposition.logUniformScale();
        incrementAnisotropy = decomposition.getAnisotropy();
        incrementDeformation = decomposition.getDeformationMagnitude();
        incrementStretchAxisRad = decomposition.getStretchAxisRad();
        incrementRotationRad = decomposition.getRotationRad();
        incrementRotationDeg = Math.toDegrees(incrementRotationRad);
        incrementDqX = qx;
        incrementDqY = qy;
        incrementPerspective = motionModel.perspectiveMagnitude(
                incrementCurrentToPrevious, frameWidth, frameHeight);
        incrementFlowPx = Math.hypot(qx, qy);
        incrementProper = decomposition.isProperRotation();
        if (!incrementProper) {
            improperIncrementCount++;
        } else {
            accumulatedLogScale += incrementLogScale;
        }

        // y is negated for Pose3D's image-down → pose-forward convention (COMP-001 §5).
        pose = new Pose3D(centreX, -centreY, 0.0, headingDegrees());
        observed = true;
    }

    /** Records the first frame as the logical origin without integrating any motion. */
    public void observeOrigin() {
        pose = new Pose3D(centreX, -centreY, 0.0, headingDegrees());
        observed = true;
    }

    private double headingDegrees() {
        return (Math.toDegrees(headingRad) % 360.0 + 360.0) % 360.0;
    }

    /** Pose in the logical frame: {@code x}, {@code y} in logical pixels, {@code z} always 0. */
    public Pose3D pose() {
        return pose;
    }

    /** Accumulated uniform visual scale, {@code exp(Σ log increments)}. Diagnostic; never applied. */
    public double accumulatedScale() {
        return Math.exp(accumulatedLogScale);
    }

    /** Heading in radians, unwrapped (not normalised to a turn) — the raw accumulated rotation. */
    public double headingRadUnwrapped() {
        return headingRad;
    }

    /** False until the first {@link #observe} or {@link #observeOrigin}. */
    public boolean hasObservation() {
        return observed;
    }
}
