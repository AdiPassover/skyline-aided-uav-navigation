package org.boofcv.stitching;

import georegression.struct.InvertibleTransform;
import georegression.struct.point.Point2D_F64;
import georegression.struct.shapes.Quadrilateral_F64;
import org.boofcv.util.structs.Pose3D;

/**
 * Continuous logical navigation state, maintained from the motion estimator alone
 * ({@code DEC-VO-003}).
 *
 * <p>This class deliberately knows nothing about mosaics, rasters, canvas sizes or rendering. Its
 * only input is the motion model's own accumulated transform, so navigation no longer depends on
 * where the current frame happens to land inside a finite stitched image.
 *
 * <h2>Coordinate frames</h2>
 *
 * <pre>
 *   C_k  camera image frame at video frame k   (input pixels, origin top-left, x right, y down)
 *   E_j  epoch reference frame j               — the camera frame at which the estimator last
 *                                                zeroed its accumulation; BoofCV calls it "first"
 *   L    logical navigation frame              — C_0, the first processed frame. FIXED for the run.
 * </pre>
 *
 * <h2>Algebra</h2>
 *
 * <p>Point-map convention, matching georegression: {@code a.concat(b, r)} yields
 * {@code r(p) = b(a(p))} — "apply {@code a}, then {@code b}".
 *
 * <pre>
 *   F_k : E_j → C_k     the estimator's accumulated transform (ImageMotion2D.getFirstToCurrent())
 *   A_j : L   → E_j     the anchor held by this class; A_0 = identity because E_0 = C_0 = L
 *   G_k : L   → C_k     the logical transform  =  A_j ∘ F_k
 * </pre>
 *
 * <p>When the estimator is about to discard {@code F} — which BoofCV 0.44 does in exactly one place
 * reachable in normal running, {@code StitchingFromMotion2D.setOriginToCurrent()} via
 * {@code motion.setToFirst()} — the caller must first invoke {@link #foldEpoch}, which performs
 *
 * <pre>
 *   A_{j+1} := A_j ∘ F_k
 * </pre>
 *
 * <p>After the reset {@code F = I}, so {@code G = A_{j+1} ∘ I = A_j ∘ F_k}: <b>identical to its
 * value immediately before the reset.</b> The composition is exact and lossless in the full
 * transform, including scale, shear and perspective — unlike the reduced SE(2) re-anchoring it
 * replaces ({@code COMP-001} §3.6, §4).
 *
 * <p><b>Estimator keyframe changes need no fold.</b> {@code ImageMotionPointTrackerKey.changeKeyFrame()}
 * executes {@code worldToKey.setTo(worldToCurr); keyToCurr.reset()}, leaving the accumulated
 * transform numerically unchanged, so SmartRespawn's tracking-robustness keyframe changes are
 * invisible here and must not be compensated for. Only a genuine accumulation reset is folded.
 *
 * <h2>Readout</h2>
 *
 * <p>Position is the <b>image centre</b> mapped into {@code L}, not an arbitrary image point: under a
 * rotation about a point only that point is fixed, so any other fixed point manufactures translation
 * out of pure rotation. Absent calibration the image centre is the standard approximation to the
 * principal point, i.e. the ground point the optical axis is looking at. The four canonical corners
 * are retained as the <b>footprint</b> readout — orientation, size and deformation — not as position.
 *
 * <p><b>Not interpreted here:</b> the footprint's scale is readable but is deliberately never
 * converted to height or used for scale compensation. That is out of scope by {@code DEC-VO-003}'s
 * scope exclusions and is reserved for a separate investigation.
 *
 * @param <IT> the 2D motion model
 */
public final class LogicalNavigationState<IT extends InvertibleTransform<IT>> {

    private final MotionModelSupport<IT> motionModel;

    /** {@code A_j : L → E_j}. Identity until the first epoch fold. */
    private final IT anchorLogicalToEpoch;

    /** {@code G_k : L → C_k}, recomputed on every {@link #observe}. */
    private final IT logicalToCurrent;

    /** {@code G_k⁻¹ : C_k → L} — the map used for every readout. */
    private IT currentToLogical;

    private final RigidMotionDecomposition accumulated = new RigidMotionDecomposition();
    private final double[] jacobian = new double[4];
    private final Quadrilateral_F64 logicalFootprint = new Quadrilateral_F64();
    private final Point2D_F64 logicalCentre = new Point2D_F64();
    private Pose3D pose = new Pose3D();
    private boolean observed = false;

    public LogicalNavigationState(MotionModelSupport<IT> motionModel, IT prototype) {
        if (motionModel == null || prototype == null) {
            throw new IllegalArgumentException("motionModel and prototype are required");
        }
        this.motionModel = motionModel;
        this.anchorLogicalToEpoch = prototype.createInstance();
        this.anchorLogicalToEpoch.reset();
        this.logicalToCurrent = prototype.createInstance();
        this.logicalToCurrent.reset();
        this.currentToLogical = prototype.createInstance();
        this.currentToLogical.reset();
    }

    /** Returns to the state of a fresh run: logical origin at the next observed frame. */
    public void reset() {
        anchorLogicalToEpoch.reset();
        logicalToCurrent.reset();
        currentToLogical.reset();
        logicalFootprint.setTo(new Quadrilateral_F64());
        logicalCentre.setTo(0, 0);
        pose = new Pose3D();
        observed = false;
    }

    /**
     * Recomputes the logical transform and every readout for the current frame.
     *
     * @param firstToCurrent {@code F_k}, read from {@code ImageMotion2D.getFirstToCurrent()}. Not
     *                       retained — its value is composed immediately.
     * @param frameWidth     current frame width in pixels
     * @param frameHeight    current frame height in pixels
     */
    public void observe(IT firstToCurrent, int frameWidth, int frameHeight) {
        anchorLogicalToEpoch.concat(firstToCurrent, logicalToCurrent);   // G = A ∘ F
        currentToLogical = logicalToCurrent.invert(currentToLogical);

        double w = frameWidth, h = frameHeight;
        motionModel.apply(currentToLogical, 0, 0, logicalFootprint.a);
        motionModel.apply(currentToLogical, w, 0, logicalFootprint.b);
        motionModel.apply(currentToLogical, w, h, logicalFootprint.c);
        motionModel.apply(currentToLogical, 0, h, logicalFootprint.d);
        motionModel.apply(currentToLogical, w / 2.0, h / 2.0, logicalCentre);

        // Position: displacement of the image centre in the logical frame. The y negation converts
        // image-down to pose-forward, matching Pose3D's documented convention (COMP-001 §5).
        double x = logicalCentre.x - w / 2.0;
        double y = -(logicalCentre.y - h / 2.0);
        pose = new Pose3D(x, y, 0.0, footprintYawDegrees());
        observed = true;
    }

    /**
     * Folds the estimator's accumulated transform into the anchor. <b>Must be called immediately
     * before</b> any operation that zeroes {@code getFirstToCurrent()} — in this codebase that is
     * {@code StitchingFromMotion2D.setOriginToCurrent()} and {@code StitchingFromMotion2D.reset()}.
     *
     * <p>Calling it at any other time silently redefines the logical origin, which is precisely the
     * defect {@code DEC-VO-003} removes; calling it too late loses the epoch's motion entirely.
     *
     * @param firstToCurrentBeforeReset {@code F_k} as it stands before the reset
     */
    public void foldEpoch(IT firstToCurrentBeforeReset) {
        IT folded = anchorLogicalToEpoch.concat(firstToCurrentBeforeReset, null);
        anchorLogicalToEpoch.setTo(folded);
    }

    /** {@code atan2} of the footprint's top edge, degrees, normalised to [0, 360). */
    private double footprintYawDegrees() {
        double dx = logicalFootprint.b.x - logicalFootprint.a.x;
        double dy = logicalFootprint.b.y - logicalFootprint.a.y;
        double angle = Math.toDegrees(Math.atan2(dy, dx));
        return (angle + 360.0) % 360.0;
    }

    /** Pose in the logical frame: {@code x}, {@code y} in logical pixels, {@code z} always 0. */
    public Pose3D pose() {
        return pose;
    }

    /**
     * The camera's image rectangle expressed in the logical frame — corners {@code a,b,c,d}
     * corresponding to image {@code (0,0), (w,0), (w,h), (0,h)}. Carries orientation, size and
     * deformation. Returns internal storage; copy before retaining.
     */
    public Quadrilateral_F64 logicalFootprint() {
        return logicalFootprint;
    }

    /** The image centre mapped into the logical frame. Returns internal storage. */
    public Point2D_F64 logicalCentre() {
        return logicalCentre;
    }

    /** {@code G_k : L → C_k}. Returns internal storage; copy before retaining. */
    public IT logicalToCurrent() {
        return logicalToCurrent;
    }

    /** {@code G_k⁻¹ : C_k → L}. Returns internal storage; copy before retaining. */
    public IT currentToLogical() {
        return currentToLogical;
    }

    /** {@code A_j : L → E_j}. Returns internal storage; copy before retaining. */
    public IT anchorLogicalToEpoch() {
        return anchorLogicalToEpoch;
    }

    /**
     * Decomposition of the <b>accumulated</b> transform's Jacobian at the image centre — the
     * deformation this path has integrated since the logical origin. Diagnostic: it is what
     * {@link NavigationSource#RIGID_MOTION} declines to integrate. Returns internal storage.
     */
    public RigidMotionDecomposition accumulatedDecomposition(int frameWidth, int frameHeight) {
        motionModel.jacobian(currentToLogical, frameWidth / 2.0, frameHeight / 2.0, jacobian);
        return accumulated.set(jacobian[0], jacobian[1], jacobian[2], jacobian[3]);
    }

    /** Perspective magnitude of the accumulated transform. 0 for the affine model, by construction. */
    public double accumulatedPerspective(int frameWidth, int frameHeight) {
        return motionModel.perspectiveMagnitude(logicalToCurrent, frameWidth, frameHeight);
    }

    /** False until the first {@link #observe}. */
    public boolean hasObservation() {
        return observed;
    }
}
