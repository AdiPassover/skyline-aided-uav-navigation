package org.boofcv.stitching;

import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.struct.image.ImageBase;
import georegression.struct.InvertibleTransform;
import georegression.struct.shapes.Quadrilateral_F64;
import lombok.Getter;
import org.boofcv.estimation.OpticalMotionEstimator;
import org.boofcv.util.structs.Pose3D;

/**
 * Navigation without a mosaic: the estimator reduced to what navigation actually needs
 * ({@code DEC-VO-003}).
 *
 * <p>This exists to make the architectural claim concrete and testable — that after separating
 * navigation state from the mosaic canvas, the navigation path depends on {@link ImageMotion2D}
 * alone. No {@code StitchingFromMotion2D}, no raster is allocated, nothing is rendered, and no
 * canvas re-origin ever occurs, so there is no epoch to fold.
 *
 * <p><b>This is not a drop-in replacement for {@link MotionModelStitchingEstimator}, and the
 * differences are behavioural, not cosmetic.</b> Relative to the mosaic-backed path it loses:
 *
 * <ul>
 *   <li><b>the area-change fault check.</b> {@code StitchingFromMotion2D.process} runs
 *       {@code checkLargeMotion} after a successful motion estimate and reports a fault on a large
 *       area jump; here, {@code success} is the motion estimator's own boolean only. That check is
 *       the component's only geometric sanity test ({@code COMP-001} §8), blind though it is to
 *       pure translation and rotation jumps.</li>
 *   <li><b>the periodic track re-anchoring that canvas re-origin caused.</b> Each
 *       {@code setOriginToCurrent()} calls {@code setToFirst()} → {@code changeKeyFrame()}, which
 *       re-anchors every active track and spawns new ones. Without it, keyframe changes happen only
 *       when SmartRespawn decides. Track lifetimes therefore differ, RANSAC sees different
 *       correspondence sets, and the trajectory <b>will</b> diverge from a mosaic-backed run over
 *       time. Neither is "the correct" trajectory a priori; they are different estimators.</li>
 *   <li><b>restart recovery.</b> There is no mosaic to reset, so a failed frame is reported and the
 *       motion estimator is left to recover on its own.</li>
 * </ul>
 *
 * <p>Use it for headless evaluation where rendering is pure overhead, and when the differences above
 * are acceptable and stated. The mosaic-backed path remains the default and the one every committed
 * baseline was produced with.
 *
 * @param <T>  image type
 * @param <IT> 2D motion model type
 */
public class MotionOnlyNavigationEstimator<T extends ImageBase<T>, IT extends InvertibleTransform<IT>>
        implements OpticalMotionEstimator<T> {

    @Getter
    private final ImageMotion2D<T, IT> motion;

    @Getter
    private final LogicalNavigationState<IT> navigation;

    /** The rigid-only state ({@code DEC-VO-004} Alternative E), maintained alongside. */
    @Getter
    private final RigidNavigationState<IT> rigidNavigation;

    /** {@code F} after the last accepted frame; identity until one exists. */
    private final IT lastGoodFirstToCurrent;

    @Getter @lombok.Setter
    private NavigationSource navigationSource = NavigationSource.LOGICAL_FRAME;

    private final MotionModelSupport<IT> motionModel;

    public MotionOnlyNavigationEstimator(ImageMotion2D<T, IT> motion,
                                         MotionModelSupport<IT> motionModel) {
        if (motion == null || motionModel == null) {
            throw new IllegalArgumentException("motion and motionModel are required");
        }
        this.motion = motion;
        this.motionModel = motionModel;
        this.navigation = new LogicalNavigationState<>(
                motionModel, motion.getFirstToCurrent().createInstance());
        this.rigidNavigation = new RigidNavigationState<>(
                motionModel, motion.getFirstToCurrent().createInstance());
        this.lastGoodFirstToCurrent = motion.getFirstToCurrent().createInstance();
        this.lastGoodFirstToCurrent.reset();
    }

    /** The motion model this estimator was built with, e.g. {@code "homography"}, {@code "affine"}. */
    public String motionModelId() {
        return motionModel.id();
    }

    @Override
    public void init() {
        reset();
    }

    @Override
    public boolean processFrame(T frame) {
        if (frame == null) {
            return false;
        }
        boolean success = motion.process(frame);
        // Observe regardless: on a failed frame the accumulated transform holds its last good value,
        // so the reported pose stays put rather than jumping — the same continuity intent the
        // mosaic-backed path achieves through restart, without a mosaic to restart.
        navigation.observe(motion.getFirstToCurrent(), frame.width, frame.height);
        if (success) {
            rigidNavigation.observe(lastGoodFirstToCurrent, motion.getFirstToCurrent(),
                    frame.width, frame.height);
            lastGoodFirstToCurrent.setTo(motion.getFirstToCurrent());
        }
        return success;
    }

    @Override
    public void reset() {
        motion.reset();
        navigation.reset();
        rigidNavigation.reset();
        lastGoodFirstToCurrent.reset();
    }

    @Override
    public Pose3D getCurrentPose() {
        return navigationSource == NavigationSource.RIGID_MOTION
                ? rigidNavigation.pose() : navigation.pose();
    }

    /** The camera's image rectangle in the fixed logical frame. Returns internal storage. */
    public Quadrilateral_F64 getLogicalFootprint() {
        return navigation.logicalFootprint();
    }
}
