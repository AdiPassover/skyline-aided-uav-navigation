package org.boofcv.stitching;

import boofcv.abst.tracker.PointTracker;
import boofcv.alg.sfm.d2.ImageMotionPointTrackerKey;
import boofcv.struct.geo.AssociatedPair;
import boofcv.struct.image.ImageBase;
import georegression.struct.InvertibleTransform;
import georegression.struct.homography.Homography2D_F64;
import org.ddogleg.fitting.modelset.ModelFitter;
import org.ddogleg.fitting.modelset.ModelMatcher;

import javax.annotation.Nullable;

/**
 * BoofCV's {@link ImageMotionPointTrackerKey}, observed — the model before and after RANSAC's
 * optional final refinement ({@code EXP-VO-010} Phase 4).
 *
 * <p>Directly parallel to {@code ResidualSnapshotTrackerKey}, and for the same reason: the values
 * are readable at exactly one instant, and reading them later reads something else. It overrides
 * {@code process} only to copy two transforms and an integer out after {@code super.process} has
 * fitted (and possibly refined) the model, and before {@code ImageMotionPtkSmartRespawn} can call
 * {@code changeKeyFrame()} and reset {@code keyToCurr} to the identity. The override calls
 * {@code super.process} exactly once, mutates no estimator state, and returns {@code super}'s result
 * unchanged.
 *
 * <p>Unlike the residual probe this one is <b>generic over the motion model</b>, because
 * {@code EXP-VO-010}'s primary ablation runs on both the affine and the projective model and the two
 * must be observed by the same code. The model-specific part is confined to the supplied
 * {@link MotionModelSupport}, whose {@code asHomography} is exact for both.
 *
 * <p>Package-private: consumers depend on {@link RefinementDiagnostics}, so nothing outside this
 * package has to name a BoofCV motion-estimation type.
 *
 * @param <I>  input image type
 * @param <IT> the 2D motion model
 */
final class RefinementSnapshotTrackerKey<I extends ImageBase<I>, IT extends InvertibleTransform<IT>>
        extends ImageMotionPointTrackerKey<I, IT>
        implements RefinementDiagnostics {

    private final MotionModelSupport<IT> support;
    private final boolean refinementEnabled;

    /**
     * Observer-only refiner ({@code EXP-CONF-004}): never handed to {@code super}, so the
     * estimator's own refiner stays exactly what the configuration built (null in the frozen
     * configuration). At snapshot time it is applied as a pure function to the live match set,
     * writing into probe-owned scratch — {@code fitModel} mutates neither its input list nor
     * the initial model, the same contract {@code RefinementMonteCarlo} relies on.
     */
    @Nullable
    private final ModelFitter<IT, AssociatedPair> diagnosticRefiner;
    @Nullable
    private final IT refitScratch;

    private final Homography2D_F64 minimal = new Homography2D_F64();
    private final Homography2D_F64 shipped = new Homography2D_F64();
    private final Homography2D_F64 refit = new Homography2D_F64();
    private int inliers = -1;
    private boolean populated = false;
    private boolean refitPopulated = false;
    private long refitTimeNs = -1L;

    RefinementSnapshotTrackerKey(PointTracker<I> tracker,
                                 ModelMatcher<IT, AssociatedPair> modelMatcher,
                                 @Nullable ModelFitter<IT, AssociatedPair> modelRefiner,
                                 IT model,
                                 int thresholdOutlierPrune,
                                 MotionModelSupport<IT> support) {
        this(tracker, modelMatcher, modelRefiner, model, thresholdOutlierPrune, support, null);
    }

    RefinementSnapshotTrackerKey(PointTracker<I> tracker,
                                 ModelMatcher<IT, AssociatedPair> modelMatcher,
                                 @Nullable ModelFitter<IT, AssociatedPair> modelRefiner,
                                 IT model,
                                 int thresholdOutlierPrune,
                                 MotionModelSupport<IT> support,
                                 @Nullable ModelFitter<IT, AssociatedPair> diagnosticRefiner) {
        super(tracker, modelMatcher, modelRefiner, model, thresholdOutlierPrune);
        this.support = support;
        // Read from the constructor argument rather than from configuration, so the diagnostic can
        // never disagree with the estimator it is attached to.
        this.refinementEnabled = modelRefiner != null;
        this.diagnosticRefiner = diagnosticRefiner;
        this.refitScratch = diagnosticRefiner != null ? model.createInstance() : null;
    }

    @Override
    public boolean process(I frame) {
        boolean estimated = super.process(frame);
        if (estimated) {
            // getModelParameters() is RANSAC's `bestFitParam` and is untouched by refinement, so it
            // is still the minimal-sample winner here even when the refiner has already run.
            support.asHomography(modelMatcher.getModelParameters(), minimal);
            support.asHomography(getKeyToCurr(), shipped);
            inliers = modelMatcher.getMatchSet().size();
            populated = true;
            if (diagnosticRefiner != null) {
                long t0 = System.nanoTime();
                boolean ok = diagnosticRefiner.fitModel(
                        modelMatcher.getMatchSet(), modelMatcher.getModelParameters(),
                        refitScratch);
                refitTimeNs = System.nanoTime() - t0;
                refitPopulated = ok;
                if (ok) {
                    support.asHomography(refitScratch, refit);
                }
            }
        } else {
            clear();
        }
        return estimated;
    }

    @Override
    public void reset() {
        super.reset();
        clear();
    }

    private void clear() {
        // Null-guarded because ImageMotionPointTrackerKey's constructor calls the overridable
        // reset(), which reaches here *before* this subclass's field initialisers have run. The
        // guard is not defensive padding: without it the probe throws on construction, which is
        // how RefinementSemanticsTest first found this. Returning early is correct because the
        // field initialisers that run immediately afterwards establish exactly the cleared state.
        if (minimal == null) {
            return;
        }
        minimal.reset();
        shipped.reset();
        refit.reset();
        inliers = -1;
        populated = false;
        refitPopulated = false;
        refitTimeNs = -1L;
    }

    @Override
    public boolean hasModels() {
        return populated;
    }

    @Override
    public Homography2D_F64 minimalSampleModel() {
        return new Homography2D_F64().setTo(minimal);
    }

    @Override
    public Homography2D_F64 shippedModel() {
        return new Homography2D_F64().setTo(shipped);
    }

    @Override
    public int inlierCount() {
        return inliers;
    }

    @Override
    public boolean refinementEnabled() {
        return refinementEnabled;
    }

    @Override
    public boolean diagnosticRefitEnabled() {
        return diagnosticRefiner != null;
    }

    @Override
    public boolean hasDiagnosticRefit() {
        return refitPopulated;
    }

    @Override
    public Homography2D_F64 diagnosticRefitModel() {
        return new Homography2D_F64().setTo(refit);
    }

    @Override
    public long diagnosticRefitTimeNs() {
        return refitTimeNs;
    }
}
