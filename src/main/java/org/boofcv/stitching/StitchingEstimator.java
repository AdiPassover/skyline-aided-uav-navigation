package org.boofcv.stitching;

import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.ImageBase;
import georegression.struct.homography.Homography2D_F64;

/**
 * The shipped stitching VO pose estimator: {@link MotionModelStitchingEstimator} fixed to BoofCV's
 * 8-DoF projective {@code Homography2D_F64} motion model.
 *
 * <p>Tracks the drone's position and orientation relative to the initial frame using 2D homography
 * tracking. This is the production path and the {@code EXP-002} baseline; its constructor and
 * generic shape are unchanged, so every existing call site is unaffected.
 *
 * <p><b>Why the body now lives in the superclass.</b> {@code DEC-VO-002} generalised the pose logic
 * over the motion-model type so {@code EXP-VO-002} could evaluate a 6-DoF affine alternative
 * without duplicating it. The homography behaviour is identical — {@code MotionModelEquivalenceTest}
 * asserts bitwise-identical poses, and {@code EXP-VO-002} reproduces {@code EXP-002}'s committed run
 * record exactly. Use {@link MotionModelStitchingEstimator} directly, with
 * {@link MotionModelSupport#AFFINE}, only in experiment code.
 *
 * @param <T> Image type.
 */
public class StitchingEstimator<T extends ImageBase<T>>
        extends MotionModelStitchingEstimator<T, Homography2D_F64> {

    public StitchingEstimator(StitchingFromMotion2D<T, Homography2D_F64> stitch) {
        super(stitch, MotionModelSupport.HOMOGRAPHY);
    }
}
