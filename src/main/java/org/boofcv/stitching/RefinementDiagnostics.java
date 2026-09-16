package org.boofcv.stitching;

import georegression.struct.homography.Homography2D_F64;

/**
 * The frame-to-keyframe model <b>before and after</b> RANSAC's optional final refinement, plus the
 * size of the inlier set the refinement was given ({@code EXP-VO-010} Phase 4).
 *
 * <h2>What this exists to measure</h2>
 *
 * <p>With {@code refineEstimate = false} — this repository's default since the estimator was written
 * — {@code ImageMotionPointTrackerKey.process} ships {@code modelMatcher.getModelParameters()}, the
 * model generated from the winning <b>3- or 4-point minimal sample</b>, even though RANSAC has
 * identified hundreds or thousands of inliers. With it true, that model is replaced by a
 * least-squares fit over the whole inlier set. `EXP-VO-010` needs to establish not only that the
 * trajectory changed but <b>how much the model parameters themselves changed</b>, which requires
 * seeing both.
 *
 * <p>So this interface reports, per frame:
 *
 * <ul>
 *   <li>{@link #minimalSampleModel()} — what RANSAC's winning hypothesis was, always available
 *       whether or not refinement is enabled;</li>
 *   <li>{@link #shippedModel()} — what the estimator actually used, which is the refined model when
 *       refinement is on and is <em>the same object's value</em> as the minimal-sample model when it
 *       is off;</li>
 *   <li>{@link #inlierCount()} — the size of {@code getMatchSet()}, i.e. exactly the number of
 *       correspondences handed to the refiner.</li>
 * </ul>
 *
 * <p><b>With refinement off the two models are identical by construction</b>, which makes every
 * {@code refine = false} arm a free correctness check on the probe itself.
 *
 * <h2>Provenance</h2>
 *
 * <p>Everything here is <b>retained state read out</b>, in the sense {@code DEC-VO-001} established
 * for residuals: no model is re-estimated, no second RANSAC or pyramid pass runs, and the estimator
 * is not mutated. Both models are values the estimator already computed; this reads them at the one
 * instant both are simultaneously valid.
 *
 * <h2>Timing — this one bites, for the same reason it did for residuals</h2>
 *
 * <p>Read the values after each {@code processFrame} and before the next. BoofCV's
 * {@code ImageMotionPtkSmartRespawn.process} calls {@code changeKeyFrame()} <em>after</em> the motion
 * is estimated whenever the inlier set shrinks or coverage drops, and {@code changeKeyFrame} does
 * {@code keyToCurr.reset()} — so the shipped model read after {@code process()} returns would be the
 * identity on exactly the frames where the respawn logic fired. The implementation snapshots inside
 * {@code process}, before that can happen; callers do not need to do anything about it, but must not
 * "optimise" the read to a later point.
 *
 * <p>Both models are reported in <b>homography form</b> ({@code MotionModelSupport.asHomography}),
 * which is exact for the affine model (last row {@code (0, 0, 1)}) and a copy for the projective one.
 * That keeps the diagnostic's schema model-independent, exactly as {@code logical_transform.csv}
 * does, so one analysis reads both arms.
 */
public interface RefinementDiagnostics {

    /** True once a frame has been processed and both models are populated. */
    boolean hasModels();

    /**
     * RANSAC's winning minimal-sample hypothesis — {@code modelMatcher.getModelParameters()}.
     * Meaningless when {@link #hasModels()} is false.
     *
     * @return a snapshot; the returned instance is not live estimator state
     */
    Homography2D_F64 minimalSampleModel();

    /**
     * The model the estimator actually used for this frame — {@code keyToCurr}. Equal to
     * {@link #minimalSampleModel()} when {@code refineEstimate} is false.
     *
     * @return a snapshot; the returned instance is not live estimator state
     */
    Homography2D_F64 shippedModel();

    /**
     * Size of {@code modelMatcher.getMatchSet()} — the number of correspondences the refiner was
     * given, or would have been given had refinement been enabled. {@code -1} before the first
     * successful frame.
     */
    int inlierCount();

    /** Whether this estimator was built with refinement enabled. Constant for a run. */
    boolean refinementEnabled();

    // ------------------------------------------------------------------------------------------
    // Diagnostic-only refit (EXP-CONF-004). With refineEstimate = false the estimator ships the
    // minimal-sample model; when the probe is additionally built with a *diagnostic* refiner —
    // an observer instance that is never handed to the estimator — it also computes what the
    // all-inlier refit WOULD have produced from the same production match set, as a pure
    // function into probe-owned scratch. Nothing feeds back: the shipped model, pose
    // accumulation, restart/recenter logic, feature lifecycle and reference selection are
    // untouched (DiagnosticRefitEquivalenceTest asserts bit-identity with the unobserved stack).
    // ------------------------------------------------------------------------------------------

    /** Whether this probe was built with the diagnostic (observer-only) refiner. */
    default boolean diagnosticRefitEnabled() { return false; }

    /**
     * True when the last frame produced a diagnostic refit. False when the probe has no
     * diagnostic refiner, the frame failed, or the refit itself failed — absence is explicit,
     * never written as zeros.
     */
    default boolean hasDiagnosticRefit() { return false; }

    /**
     * The all-inlier refit of the last frame's production match set, in homography form.
     * Meaningless when {@link #hasDiagnosticRefit()} is false.
     *
     * @return a snapshot; the returned instance is not live estimator state
     */
    default Homography2D_F64 diagnosticRefitModel() { return new Homography2D_F64(); }

    /**
     * Wall time of the diagnostic {@code fitModel} call alone (the refit operation, nothing
     * else), or {@code -1} when no refit ran this frame.
     */
    default long diagnosticRefitTimeNs() { return -1L; }

    /** Nothing observed yet, or the last frame's estimation failed. */
    RefinementDiagnostics UNAVAILABLE = new RefinementDiagnostics() {
        @Override public boolean hasModels() { return false; }
        @Override public Homography2D_F64 minimalSampleModel() { return new Homography2D_F64(); }
        @Override public Homography2D_F64 shippedModel() { return new Homography2D_F64(); }
        @Override public int inlierCount() { return -1; }
        @Override public boolean refinementEnabled() { return false; }
    };
}
