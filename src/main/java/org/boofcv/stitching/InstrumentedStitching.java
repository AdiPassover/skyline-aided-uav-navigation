package org.boofcv.stitching;

import boofcv.alg.sfm.d2.StitchingFromMotion2D;
import boofcv.struct.image.ImageBase;
import georegression.struct.homography.Homography2D_F64;

/**
 * A stitching engine paired with the residual diagnostics of the very estimator inside it.
 *
 * <p>The pairing is the point: handing back the two together makes it impossible to accidentally
 * read a {@link MotionResidualDiagnostics} belonging to one estimator while reporting the pose of
 * another. The feature spec's FR-002 — diagnostics must come from the SAME estimator that produces
 * the shipped pose — is enforced by this type's shape rather than by convention.
 *
 * <p>Read {@code diagnostics} after each {@code stitch.process(frame)} call and before the next
 * one; see {@link MotionResidualDiagnostics} on timing.
 *
 * @param stitch      the stitching engine, behaviourally identical to one built by the
 *                    non-instrumented factory methods
 * @param diagnostics residuals from {@code stitch}'s own motion estimator
 * @param <T>         image type
 */
public record InstrumentedStitching<T extends ImageBase<T>>(
        StitchingFromMotion2D<T, Homography2D_F64> stitch,
        MotionResidualDiagnostics diagnostics) {
}
