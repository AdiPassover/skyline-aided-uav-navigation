package org.boofcv.stitching;

import boofcv.abst.tracker.PointTracker;
import boofcv.alg.geo.robust.DistanceHomographySq;
import boofcv.alg.sfm.d2.ImageMotionPointTrackerKey;
import boofcv.struct.geo.AssociatedPair;
import boofcv.struct.image.ImageBase;
import georegression.struct.homography.Homography2D_F64;
import org.ddogleg.fitting.modelset.ModelFitter;
import org.ddogleg.fitting.modelset.ModelMatcher;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;

/**
 * BoofCV's {@link ImageMotionPointTrackerKey}, observed. Subclasses it purely to read out
 * reprojection residuals at the one instant they are readable, and changes nothing about how
 * motion is estimated.
 *
 * <p>Package-private on purpose: consumers depend on {@link MotionResidualDiagnostics}, which is
 * this class's only public surface. Nothing outside this package needs to name a BoofCV
 * motion-estimation type to read residuals.
 *
 * <h2>Why a subclass rather than an after-the-fact read</h2>
 *
 * <p>Reading residuals after {@code stitch.process(frame)} returns is <b>not</b> correct, and would
 * fail silently rather than loudly. BoofCV's {@code ImageMotionPtkSmartRespawn.process} calls
 * {@code motion.changeKeyFrame()} <em>after</em> the motion is estimated whenever the inlier set
 * has shrunk or feature coverage has dropped. {@code changeKeyFrame} rewrites every active track's
 * {@code p1} to its current pixel and resets {@code keyToCurr} to identity. A residual computed
 * after that point maps a point onto itself through the identity transform and reads as a
 * near-perfect fit — precisely on the frames where tracking is degrading, which are the frames the
 * signal exists to characterise.
 *
 * <p>Overriding {@code process} places the observation between the two: after
 * {@code super.process} has fitted the model and marked the match set, and before the respawn
 * logic that would destroy the evidence. The override calls {@code super.process} exactly once,
 * mutates no estimator state, and returns {@code super}'s result unchanged, so the estimator's
 * behaviour is identical with and without it.
 *
 * <p>A second reason a snapshot is needed rather than a lazy read: {@code AssociatedPairTrack.p2}
 * is a live alias of the {@code PointTrack}'s pixel field (BoofCV sets {@code p.p2 = l.pixel} when
 * spawning), so the match set's observed coordinates move as later frames are tracked. Copying the
 * residuals out at capture time makes them describe the frame they belong to.
 *
 * @param <I> input image type
 */
final class ResidualSnapshotTrackerKey<I extends ImageBase<I>>
        extends ImageMotionPointTrackerKey<I, Homography2D_F64>
        implements MotionResidualDiagnostics {

    private final double inlierThresholdSq;

    /** Our own distance instance — never the estimator's, so its model field is not shared state. */
    private final DistanceHomographySq distance = new DistanceHomographySq();

    /** Defensive copy of the shipped model, so {@code distance} never aliases live estimator state. */
    private final Homography2D_F64 modelSnapshot = new Homography2D_F64();

    private Summary summary = Summary.UNAVAILABLE;

    @Nullable
    private double[] residuals = null;

    ResidualSnapshotTrackerKey(PointTracker<I> tracker,
                               ModelMatcher<Homography2D_F64, AssociatedPair> modelMatcher,
                               @Nullable ModelFitter<Homography2D_F64, AssociatedPair> modelRefiner,
                               Homography2D_F64 model,
                               int thresholdOutlierPrune,
                               double inlierThresholdSq) {
        super(tracker, modelMatcher, modelRefiner, model, thresholdOutlierPrune);
        this.inlierThresholdSq = inlierThresholdSq;
    }

    @Override
    public boolean process(I frame) {
        boolean estimated = super.process(frame);
        capture(estimated);
        return estimated;
    }

    @Override
    public void reset() {
        super.reset();
        clear();
    }

    @Override
    public double inlierThresholdSquaredPx() {
        return inlierThresholdSq;
    }

    @Override
    public Summary summarize() {
        return summary;
    }

    @Override
    @Nullable
    public double[] inlierSquaredResidualsPx() {
        return residuals == null ? null : residuals.clone();
    }

    private void clear() {
        summary = Summary.UNAVAILABLE;
        residuals = null;
    }

    /**
     * Reads residuals out of the estimator's retained match set and retained model.
     *
     * <p>Residuals are measured against {@link #getKeyToCurr()} — the model the estimator actually
     * ships as its motion, and therefore the model the reported pose derives from. With refinement
     * disabled (the repository default) this is the identical object RANSAC selected its inliers
     * with; with refinement enabled it is the refined model, which is the honest choice here since
     * a residual should describe the motion that was used, not one that was discarded.
     */
    private void capture(boolean estimated) {
        if (!estimated) {
            // No motion was estimated against the current keyframe. The match set still holds the
            // previous frame's inliers; reporting them here would attribute them to the wrong
            // frame, so absence is reported instead.
            clear();
            return;
        }

        List<AssociatedPair> matchSet = getModelMatcher().getMatchSet();
        if (matchSet == null) {
            clear();
            return;
        }

        int count = matchSet.size();
        if (count == 0) {
            // Estimation reported success over an empty match set. The count is a real measurement
            // of zero; the statistics genuinely do not exist. Both facts are reported, separately.
            summary = new Summary(0, null, null, null, null);
            residuals = null;
            return;
        }

        modelSnapshot.setTo(getKeyToCurr());
        distance.setModel(modelSnapshot);

        double[] values = new double[count];
        double sum = 0.0;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            double d = distance.distance(matchSet.get(i));
            values[i] = d;
            sum += d;
            if (d > max) {
                max = d;
            }
        }

        double mean = sum / count;

        double[] sorted = values.clone();
        Arrays.sort(sorted);
        // Lower median for an even-sized set, so the reported value is one the estimator actually
        // produced rather than an interpolation between two of them.
        double median = sorted[(count - 1) / 2];

        residuals = values;
        summary = new Summary(count, mean, Math.sqrt(mean), median, max);
    }
}
