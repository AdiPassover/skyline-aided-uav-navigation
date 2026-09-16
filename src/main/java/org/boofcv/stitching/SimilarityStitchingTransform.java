package org.boofcv.stitching;

import boofcv.alg.distort.PixelTransformAffine_F32;
import boofcv.alg.sfm.d2.StitchingTransform;
import boofcv.struct.distort.PixelTransform;
import georegression.struct.affine.Affine2D_F32;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F32;

import javax.annotation.Nullable;

/**
 * The {@link StitchingTransform} for {@link Sim2_F64}, so a similarity-model motion estimator can
 * drive {@code StitchingFromMotion2D} exactly as the homography and affine models do.
 *
 * <p>BoofCV's {@code FactoryStitchingTransform} supplies affine and homography variants only, and
 * {@code FactoryMotion2D.createVideoStitch} selects between them by testing the transform type —
 * falling through to the homography branch for anything unrecognised, which would fail on a
 * {@code Sim2_F64}. This class fills the gap; it is a direct transliteration of
 * {@code FactoryStitchingTransform.createAffine_F64()} with the similarity's linear block
 * substituted, and introduces no approximation in either direction:
 *
 * <pre>
 *   [ a  −b  tx ]
 *   [ b   a  ty ]     is an affine matrix, and a homography with last row (0, 0, 1)
 *   [ 0   0   1 ]
 * </pre>
 *
 * <p><b>The {@code F32} narrowing in {@link #convertPixel} is BoofCV's, not ours</b> — the affine
 * path does exactly the same, because {@code PixelTransformAffine_F32} is what the image distorter
 * consumes. It affects <em>rendering only</em>. Navigation never reads a rendered pixel: it reads
 * the double-precision model through {@code MotionModelSupport}, and {@link #convertH} — the path
 * the run record and every pose consumer use — is exact.
 */
public final class SimilarityStitchingTransform implements StitchingTransform<Sim2_F64> {

    private final Affine2D_F32 scratch = new Affine2D_F32();

    @Override
    public PixelTransform<Point2D_F32> convertPixel(Sim2_F64 input,
                                                    @Nullable PixelTransform<Point2D_F32> output) {
        scratch.setTo((float) input.a, (float) -input.b,
                      (float) input.b, (float) input.a,
                      (float) input.tx, (float) input.ty);

        if (output != null) {
            ((PixelTransformAffine_F32) output).setTo(scratch);
        } else {
            PixelTransformAffine_F32 t = new PixelTransformAffine_F32();
            t.setTo(scratch);
            output = t;
        }
        return output;
    }

    @Override
    public Homography2D_F64 convertH(Sim2_F64 input, @Nullable Homography2D_F64 output) {
        if (output == null) output = new Homography2D_F64();
        output.setTo(input.a, -input.b, input.tx,
                     input.b, input.a, input.ty,
                     0, 0, 1);
        return output;
    }
}
