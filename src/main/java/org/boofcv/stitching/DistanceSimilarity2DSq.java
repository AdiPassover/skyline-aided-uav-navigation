package org.boofcv.stitching;

import boofcv.struct.geo.AssociatedPair;
import org.ddogleg.fitting.modelset.DistanceFromModel;

import java.util.List;

/**
 * Squared Euclidean pixel reprojection error of a correspondence under a {@link Sim2_F64}:
 * {@code ‖p₂ − (s R p₁ + t)‖²}.
 *
 * <p><b>The metric is the load-bearing part of this class, not the arithmetic.</b> BoofCV's
 * {@code DistanceHomographySq} and {@code DistanceAffine2DSq} both return squared Euclidean pixel
 * error after mapping {@code p1} through the model, and so does this. That is what makes
 * {@code inlierThresholdSq = 3.0} mean an <em>identical</em> thing in all three arms of
 * {@code EXP-VO-009} — a correspondence is an inlier when its reprojection error is ≤ √3 ≈ 1.73 px —
 * and without it the model comparison would not be interpretable at all. {@code LIT-VO-001} made the
 * same check first for the affine arm; {@code LIT-VO-005} §5 records that BoofCV's own
 * {@code DistanceScaleTranslateRotate2DSq} computes exactly this expression, so the semantics are
 * BoofCV's even though the class cannot be reused (it is typed to {@code ScaleTranslateRotate2D},
 * which is not an {@code InvertibleTransform}).
 */
public class DistanceSimilarity2DSq implements DistanceFromModel<Sim2_F64, AssociatedPair> {

    private double a, b, tx, ty;

    @Override
    public void setModel(Sim2_F64 model) {
        this.a = model.a;
        this.b = model.b;
        this.tx = model.tx;
        this.ty = model.ty;
    }

    @Override
    public double distance(AssociatedPair pt) {
        double dx = pt.p2.x - (a * pt.p1.x - b * pt.p1.y + tx);
        double dy = pt.p2.y - (b * pt.p1.x + a * pt.p1.y + ty);
        return dx * dx + dy * dy;
    }

    @Override
    public void distances(List<AssociatedPair> obs, double[] distance) {
        final int n = obs.size();
        for (int i = 0; i < n; i++) {
            distance[i] = distance(obs.get(i));
        }
    }

    @Override
    public Class<AssociatedPair> getPointType() {
        return AssociatedPair.class;
    }

    @Override
    public Class<Sim2_F64> getModelType() {
        return Sim2_F64.class;
    }
}
