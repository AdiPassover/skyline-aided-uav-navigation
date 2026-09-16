package org.boofcv.stitching;

import georegression.struct.InvertibleTransform;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;
import org.boofcv.util.structs.Pose3D;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Known-answer tests for {@link RigidNavigationState} — {@code DEC-VO-004} Alternative E.
 *
 * <p>The contract under test, in one sentence: <b>the navigation readout integrates the rigid part
 * of each frame's image motion and nothing else</b>, uniform scale is observed but never applied,
 * and anisotropy/shear/perspective reach the diagnostics but never the trajectory.
 *
 * <p>Every case is driven through both motion models, with the increments constructed exactly so the
 * answer is known before the run rather than compared against another implementation. The frame is
 * deliberately non-square (400 × 300) so an x/y or width/height transposition cannot pass.
 *
 * <p>Evidence tier: T1 (analytical). Says nothing about VO accuracy on any imagery.
 */
public class RigidNavigationStateTest {

    private static final int W = 400, H = 300;
    private static final double CX = W / 2.0, CY = H / 2.0;
    private static final double EPS = 1e-9;

    // ---------------------------------------------------------------- increment builders
    // All are C_{k-1} -> C_k, i.e. how image CONTENT moves between consecutive frames.

    private static Homography2D_F64 hTranslate(double tx, double ty) {
        return new Homography2D_F64(1, 0, tx, 0, 1, ty, 0, 0, 1);
    }

    /** Linear map about the image centre: p ↦ c + M(p − c). */
    private static Homography2D_F64 hAboutCentre(double m11, double m12, double m21, double m22) {
        return new Homography2D_F64(
                m11, m12, CX - (m11 * CX + m12 * CY),
                m21, m22, CY - (m21 * CX + m22 * CY),
                0, 0, 1);
    }

    private static Homography2D_F64 hRotate(double deg) {
        double r = Math.toRadians(deg), c = Math.cos(r), s = Math.sin(r);
        return hAboutCentre(c, -s, s, c);
    }

    private static Homography2D_F64 hScale(double s) {
        return hAboutCentre(s, 0, 0, s);
    }

    private static Homography2D_F64 hAnisotropic(double sx, double sy) {
        return hAboutCentre(sx, 0, 0, sy);
    }

    private static Homography2D_F64 hShear(double k) {
        return hAboutCentre(1, k, 0, 1);
    }

    /** Perspective about the image centre: leaves c fixed, bends everything else. */
    private static Homography2D_F64 hPerspective(double px, double py) {
        Homography2D_F64 toOrigin = hTranslate(-CX, -CY);
        Homography2D_F64 persp = new Homography2D_F64(1, 0, 0, 0, 1, 0, px, py, 1);
        Homography2D_F64 back = hTranslate(CX, CY);
        Homography2D_F64 tmp = new Homography2D_F64();
        toOrigin.concat(persp, tmp);
        Homography2D_F64 out = new Homography2D_F64();
        tmp.concat(back, out);
        return out;
    }

    private static Affine2D_F64 toAffine(Homography2D_F64 h) {
        assertEquals(0.0, h.a31, 0, "not affine");
        assertEquals(0.0, h.a32, 0, "not affine");
        return new Affine2D_F64(h.a11 / h.a33, h.a12 / h.a33, h.a21 / h.a33, h.a22 / h.a33,
                h.a13 / h.a33, h.a23 / h.a33);
    }

    // ---------------------------------------------------------------- driver

    /** Composes increments into the estimator's absolute {@code F} and feeds both navigation paths. */
    private static final class Driver<IT extends InvertibleTransform<IT>> {
        final RigidNavigationState<IT> rigid;
        final LogicalNavigationState<IT> logical;
        final IT current;
        final IT previous;

        Driver(MotionModelSupport<IT> model, IT prototype) {
            rigid = new RigidNavigationState<>(model, prototype);
            logical = new LogicalNavigationState<>(model, prototype);
            current = prototype.createInstance();
            previous = prototype.createInstance();
            current.reset();
            previous.reset();
            rigid.observeOrigin();
            logical.observe(current, W, H);
        }

        /** One accepted frame whose motion relative to the previous frame is {@code increment}. */
        void step(IT increment) {
            previous.setTo(current);
            IT next = current.concat(increment, null);   // F_k = D_k ∘ F_{k-1}
            current.setTo(next);
            rigid.observe(previous, current, W, H);
            logical.observe(current, W, H);
        }

        /** Simulates BoofCV zeroing its accumulation (canvas re-origin / restart). */
        void zeroAccumulation() {
            previous.reset();
            current.reset();
        }

        Pose3D pose() { return rigid.pose(); }
    }

    private static Driver<Homography2D_F64> homography() {
        return new Driver<>(MotionModelSupport.HOMOGRAPHY, new Homography2D_F64());
    }

    private static Driver<Affine2D_F64> affine() {
        return new Driver<>(MotionModelSupport.AFFINE, new Affine2D_F64());
    }

    // ================================================================ 1. exact similarity

    @Test
    public void pureTranslationGivesTheExpectedXy() {
        Driver<Homography2D_F64> h = homography();
        h.step(hTranslate(15, -25));
        assertEquals(-15.0, h.pose().x, EPS);
        assertEquals(-25.0, h.pose().y, EPS);
        assertEquals(0.0, h.pose().yaw, EPS);
        assertEquals(1.0, h.rigid.getIncrementScale(), EPS);

        Driver<Affine2D_F64> a = affine();
        a.step(toAffine(hTranslate(15, -25)));
        assertEquals(-15.0, a.pose().x, EPS);
        assertEquals(-25.0, a.pose().y, EPS);
    }

    @Test
    public void translationAccumulatesOverManyFrames() {
        Driver<Homography2D_F64> h = homography();
        for (int i = 0; i < 50; i++) {
            h.step(hTranslate(2, 3));
        }
        assertEquals(-100.0, h.pose().x, 1e-8);
        // y = +Σty, x = -Σtx: Pose3D's image-down → pose-forward convention, unchanged from the
        // legacy and full-logical readouts (StitchingEstimatorTest asserts the same signs).
        assertEquals(150.0, h.pose().y, 1e-8);
        assertEquals(1.0, h.rigid.accumulatedScale(), 1e-12);
    }

    @Test
    public void yawScaleAndTranslationCombinedAreSeparatedExactly() {
        // content: rotate 20 deg about the centre, scale 1.10, then translate (8, -4)
        Homography2D_F64 rs = new Homography2D_F64();
        hRotate(20).concat(hScale(1.10), rs);
        Homography2D_F64 d = new Homography2D_F64();
        rs.concat(hTranslate(8, -4), d);

        Driver<Homography2D_F64> h = homography();
        h.step(d);
        assertEquals(340.0, h.pose().yaw, 1e-8, "content rotating +20° means the camera rotated −20°");
        assertEquals(1.0 / 1.10, h.rigid.getIncrementScale(), 1e-12);
        assertEquals(1.0, h.rigid.getIncrementAnisotropy(), 1e-12);
        assertEquals(0.0, h.rigid.getIncrementPerspective(), 0.0);

        Driver<Affine2D_F64> a = affine();
        a.step(toAffine(d));
        assertEquals(h.pose().x, a.pose().x, 1e-9, "models must agree on an exact similarity");
        assertEquals(h.pose().y, a.pose().y, 1e-9);
        assertEquals(h.pose().yaw, a.pose().yaw, 1e-9);
    }

    /**
     * On a sequence that is an exact <b>rigid</b> motion — rotations and translations, no scale — the
     * rigid readout and the full logical composition must agree exactly, because there is nothing
     * for the full path to accumulate that the rigid path drops. Any disagreement here would mean
     * the rigid path is doing something beyond discarding the non-rigid part.
     */
    @Test
    public void onAnExactlyRigidSequenceRigidAndFullLogicalAgree() {
        Driver<Homography2D_F64> h = homography();
        double[][] steps = {{3, 1, 2}, {-2, 4, -1}, {5, -3, 0.5}, {1, 1, -2}};
        for (int rep = 0; rep < 10; rep++) {
            for (double[] s : steps) {
                Homography2D_F64 d = new Homography2D_F64();
                hRotate(s[2]).concat(hTranslate(s[0], s[1]), d);
                h.step(d);
            }
        }
        Pose3D rigid = h.rigid.pose();
        Pose3D full = h.logical.pose();
        assertEquals(full.x, rigid.x, 1e-6, "x");
        assertEquals(full.y, rigid.y, 1e-6, "y");
        assertEquals(full.yaw, rigid.yaw, 1e-6, "yaw");
        assertEquals(1.0, h.rigid.accumulatedScale(), 1e-9);
    }

    /**
     * <b>The trade-off {@code DEC-VO-004} accepts, pinned as a test rather than left in prose.</b>
     * Once the sequence carries a genuine uniform scale change — a real height change, under the
     * ideal model — the two paths necessarily diverge: the full path rescales each epoch's
     * displacement into logical units and is metric-consistent ({@code EXP-VO-005} H5), while the
     * rigid path integrates displacements in each frame's own pixel scale and is not. The rigid
     * readout is therefore <i>not</i> a strictly better reading of the same state; it trades the
     * height-normalisation of XY (which the estimator cannot currently be trusted to supply on real
     * imagery) for immunity to the deformation it accumulates instead.
     */
    @Test
    public void underGenuineScaleChangeTheRigidAndFullPathsDivergeByDesign() {
        Driver<Homography2D_F64> h = homography();
        for (int i = 0; i < 200; i++) {
            Homography2D_F64 d = new Homography2D_F64();
            hScale(1.002).concat(hTranslate(3, 0), d);   // climbing/descending while translating
            h.step(d);
        }
        double accumulated = h.rigid.accumulatedScale();
        assertTrue(accumulated < 0.7, "a real scale change accumulated: " + accumulated);
        assertTrue(Math.abs(h.logical.pose().x - h.rigid.pose().x) > 0.1 * Math.abs(h.rigid.pose().x),
                "full " + h.logical.pose().x + " vs rigid " + h.rigid.pose().x);
        assertEquals(0.0, h.rigid.pose().y, 1e-6, "but neither manufactures cross-track motion");
        assertEquals(0.0, h.logical.pose().y, 1e-6);
    }

    // ================================================================ 2. pure yaw

    @Test
    public void pureYawProducesNoTranslationInEitherModel() {
        Driver<Homography2D_F64> h = homography();
        for (int i = 0; i < 36; i++) {
            h.step(hRotate(10));
        }
        assertEquals(0.0, h.pose().x, 1e-9, "a full turn must not move the camera");
        assertEquals(0.0, h.pose().y, 1e-9);
        assertEquals(1.0, h.rigid.accumulatedScale(), 1e-9);
        assertEquals(-360.0, Math.toDegrees(h.rigid.headingRadUnwrapped()), 1e-9);

        Driver<Affine2D_F64> a = affine();
        for (int i = 0; i < 36; i++) {
            a.step(toAffine(hRotate(10)));
        }
        assertEquals(0.0, a.pose().x, 1e-9);
        assertEquals(0.0, a.pose().y, 1e-9);
    }

    @Test
    public void yawIsMeasuredAboutTheImageCentreNotTheImageOrigin() {
        // The failure mode DEC-VO-003 identified: a rotation about the centre moves the image
        // ORIGIN a long way. A readout that used the transform's translation column would report it.
        Homography2D_F64 d = hRotate(30);
        assertTrue(Math.hypot(d.a13, d.a23) > 50, "the origin really does move under this rotation");
        Driver<Homography2D_F64> h = homography();
        h.step(d);
        assertEquals(0.0, h.pose().x, 1e-9);
        assertEquals(0.0, h.pose().y, 1e-9);
        assertEquals(330.0, h.pose().yaw, 1e-9);
    }

    // ================================================================ 3. pure uniform scale

    @Test
    public void pureUniformScaleMovesNothingAndIsReportedAsScale() {
        Driver<Homography2D_F64> h = homography();
        for (int i = 0; i < 20; i++) {
            h.step(hScale(1.05));           // content expands: the camera is descending
        }
        assertEquals(0.0, h.pose().x, 1e-9, "a height change must not create XY motion");
        assertEquals(0.0, h.pose().y, 1e-9);
        assertEquals(0.0, h.pose().yaw, 1e-9);
        assertEquals(Math.pow(1.0 / 1.05, 20), h.rigid.accumulatedScale(), 1e-9);
        assertEquals(1.0, h.rigid.getIncrementAnisotropy(), 1e-12);

        Driver<Affine2D_F64> a = affine();
        for (int i = 0; i < 20; i++) {
            a.step(toAffine(hScale(1.05)));
        }
        assertEquals(0.0, a.pose().x, 1e-9);
        assertEquals(0.0, a.pose().y, 1e-9);
        assertEquals(h.rigid.accumulatedScale(), a.rigid.accumulatedScale(), 1e-9);
    }

    // ================================================================ 4. non-rigid contamination

    @Test
    public void anisotropicScaleAboutTheCentreCreatesNoTranslationAndIsReported() {
        Driver<Affine2D_F64> a = affine();
        a.step(toAffine(hAnisotropic(1.20, 1.00)));
        assertEquals(0.0, a.pose().x, 1e-9, "deformation about the optical axis is not translation");
        assertEquals(0.0, a.pose().y, 1e-9);
        assertEquals(1.20, a.rigid.getIncrementAnisotropy(), 1e-9, "reported as anisotropy");
        assertEquals(1.0 / Math.sqrt(1.20), a.rigid.getIncrementScale(), 1e-9);
        assertEquals(0.0, a.rigid.getIncrementPerspective(), 0.0);
    }

    /**
     * A shear is <b>partly</b> interpreted as rotation by any rigid readout — the nearest rotation to
     * a shear is not the identity. This is a property of the projection, not a defect, and it is
     * asserted here so the magnitude is on the record: the diagnostics are what reveal that the
     * rotation is not trustworthy.
     */
    @Test
    public void shearProducesNoTranslationButDoesLeakIntoRotationAndIsFlagged() {
        Driver<Affine2D_F64> a = affine();
        a.step(toAffine(hShear(0.10)));
        assertEquals(0.0, a.pose().x, 1e-9);
        assertEquals(0.0, a.pose().y, 1e-9);
        assertEquals(1.0, a.rigid.getIncrementScale(), 1e-12, "a shear has unit determinant");
        assertTrue(a.rigid.getIncrementAnisotropy() > 1.10,
                "anisotropy flags it: " + a.rigid.getIncrementAnisotropy());
        // The readout reports the CAMERA's rotation, i.e. the polar rotation of D⁻¹, so the sign is
        // opposite to the content shear's.
        assertEquals(Math.toDegrees(Math.atan2(0.10, 2.0)), a.rigid.getIncrementRotationDeg(), 1e-9);
    }

    @Test
    public void perspectiveAboutTheCentreCreatesNoTranslationAndIsReported() {
        Driver<Homography2D_F64> h = homography();
        h.step(hPerspective(2e-4, -1e-4));
        assertEquals(0.0, h.pose().x, 1e-6, "perspective about the axis is not translation");
        assertEquals(0.0, h.pose().y, 1e-6);
        assertTrue(h.rigid.getIncrementPerspective() > 0.01,
                "perspective magnitude reported: " + h.rigid.getIncrementPerspective());
        assertTrue(h.rigid.getIncrementAnisotropy() > 1.0);
    }

    // ================================================================ 5. long accumulation

    /**
     * <b>The failure mechanism {@code EXP-VO-004}/{@code EXP-VO-005} measured, reproduced in
     * miniature.</b> A tiny per-frame scale bias, shear bias and projective bias are repeated for
     * thousands of frames on top of a real translation. The full logical composition accumulates all
     * three — it is supposed to — and its XY is corrupted by them. The rigid readout must integrate
     * the same translation without any of that, and its diagnostics must show the rejected state
     * growing.
     */
    @Test
    public void tinyPerFrameNonRigidBiasesCorruptFullLogicalButNotRigidXy() {
        final int frames = 4000;
        final double perFrameScale = 1.0002;       // +0.02 %/frame, the order EXP-VO-004 measured
        final double perFrameShear = 2e-5;
        final double perFramePersp = 2e-8;

        Driver<Homography2D_F64> h = homography();
        for (int i = 0; i < frames; i++) {
            Homography2D_F64 d = new Homography2D_F64();
            hScale(perFrameScale).concat(hShear(perFrameShear), d);
            Homography2D_F64 d2 = new Homography2D_F64();
            d.concat(hPerspective(perFramePersp, 0), d2);
            Homography2D_F64 d3 = new Homography2D_F64();
            d2.concat(hTranslate(3, 0), d3);
            h.step(d3);
        }

        // The rigid path integrates the translation and only the translation. Each frame's centre
        // shift is -M^-1 t with M the increment's linear part, so the total is not exactly -3*frames;
        // what matters is that it stays the right order and does not diverge.
        Pose3D rigid = h.rigid.pose();
        Pose3D full = h.logical.pose();
        assertTrue(Math.abs(rigid.x) > 1000 && Math.abs(rigid.x) < 12000,
                "rigid x stays the right order of magnitude: " + rigid.x);
        assertTrue(Math.abs(rigid.y) < 0.05 * Math.abs(rigid.x),
                "no cross-track drift is manufactured: " + rigid.y);

        // The full path accumulated the deformation, by design, and its XY differs grossly. No
        // closed form is asserted for the accumulated scale: with a perspective term present, the
        // Jacobian varies across the image, so the chain's centre determinant is not the product of
        // the per-frame centre determinants. Direction and order of magnitude are what matter.
        RigidMotionDecomposition acc = h.logical.accumulatedDecomposition(W, H);
        double scaleOnlyPrediction = Math.pow(1.0 / perFrameScale, frames);   // ≈ 0.449
        assertTrue(acc.getUniformScale() < 0.7,
                "the full path really did accumulate the scale bias: " + acc.getUniformScale());
        assertTrue(acc.getAnisotropy() > 1.05,
                "and the shear bias: accumulated anisotropy " + acc.getAnisotropy());
        assertTrue(Math.abs(full.x - rigid.x) > 0.25 * Math.abs(rigid.x),
                "the two paths must differ grossly: full " + full.x + " vs rigid " + rigid.x);

        // The rigid path's own diagnostics record what it declined to integrate.
        assertTrue(h.rigid.accumulatedScale() < 0.7,
                "rejected scale state is recorded: " + h.rigid.accumulatedScale()
                        + " (scale-bias-only prediction " + scaleOnlyPrediction + ")");
        assertTrue(h.rigid.getIncrementAnisotropy() > 1.0);
        assertTrue(h.rigid.getIncrementPerspective() > 0.0);
    }

    // ================================================================ 6. reference resets

    /**
     * Zeroing the estimator's accumulation mid-run — what a mosaic canvas re-origin does — must not
     * change the rigid trajectory at all. There is no anchor to fold: the readout only ever consumes
     * within-epoch increments, and after a reset the increment is the new accumulation itself.
     */
    @Test
    public void zeroingTheAccumulationMidRunLeavesTheRigidTrajectoryUnchanged() {
        Homography2D_F64[] steps = new Homography2D_F64[60];
        for (int i = 0; i < steps.length; i++) {
            Homography2D_F64 rs = new Homography2D_F64();
            hRotate(0.7).concat(hScale(1.003), rs);
            Homography2D_F64 d = new Homography2D_F64();
            rs.concat(hTranslate(2 + 0.01 * i, -1), d);
            steps[i] = d;
        }

        Driver<Homography2D_F64> never = homography();
        for (Homography2D_F64 d : steps) never.step(d);

        for (int period : new int[]{2, 7, 13, 25}) {
            Driver<Homography2D_F64> often = homography();
            for (int i = 0; i < steps.length; i++) {
                often.step(steps[i]);
                if (i % period == period - 1) {
                    often.zeroAccumulation();
                }
            }
            assertEquals(never.pose().x, often.pose().x, 1e-9, "period " + period);
            assertEquals(never.pose().y, often.pose().y, 1e-9, "period " + period);
            assertEquals(never.pose().yaw, often.pose().yaw, 1e-9, "period " + period);
            assertEquals(never.rigid.accumulatedScale(), often.rigid.accumulatedScale(), 1e-9);
        }
    }

    /** The same intervention destroys the full logical path unless its anchor is folded — the
     *  contrast that shows the rigid path's invariance is structural, not incidental. */
    @Test
    public void theFullLogicalPathWouldNeedItsAnchorFoldedWhereTheRigidPathNeedsNothing() {
        Driver<Homography2D_F64> d = homography();
        for (int i = 0; i < 10; i++) d.step(hTranslate(5, 0));
        double beforeRigid = d.pose().x;
        double beforeFull = d.logical.pose().x;
        d.zeroAccumulation();
        d.logical.observe(d.current, W, H);        // no fold: what DEC-VO-003 corrected
        assertEquals(beforeRigid, d.pose().x, 1e-12, "rigid readout is untouched by the reset");
        assertNotEquals(beforeFull, d.logical.pose().x, "full path loses its origin without a fold");
    }

    // ================================================================ 8. rejected estimates

    /**
     * The {@code EXP-VO-004} R0 failure, at the state level: a wild estimate must not enter the
     * trajectory. The state integrates only what it is given, so the contract is that the caller
     * never calls {@code observe} for a rejected frame and passes the last <i>accepted</i>
     * accumulation. Here the wild transform is written into the estimator's accumulation (as BoofCV
     * does before {@code checkLargeMotion} runs) and then discarded.
     */
    @Test
    public void aRejectedIncrementIsNotIntegratedWhenTheCallerSkipsIt() {
        Driver<Homography2D_F64> h = homography();
        for (int i = 0; i < 5; i++) h.step(hTranslate(4, 0));
        Pose3D good = h.pose();

        // BoofCV writes the rejected estimate into F; the caller does NOT call observe.
        Homography2D_F64 wild = new Homography2D_F64();
        h.current.concat(hTranslate(500, -700), wild);
        h.current.setTo(wild);

        assertEquals(good.x, h.pose().x, 0.0, "pose must not move on a rejected frame");
        assertEquals(good.y, h.pose().y, 0.0);

        // After the restart the accumulation is identity again and the previous is identity too.
        h.zeroAccumulation();
        h.step(hTranslate(4, 0));
        assertEquals(good.x - 4.0, h.pose().x, 1e-9, "and motion resumes from the last good pose");
    }
}
