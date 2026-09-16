package org.boofcv.stitching;

import javax.annotation.Nullable;

/**
 * VO-owned, estimator-oriented view of the reprojection residuals of the <em>same</em> motion
 * estimator that produces the shipped VO pose.
 *
 * <p><b>Scope.</b> This interface carries raw observational quantities only. It deliberately
 * contains no threshold comparison, no score, no calibration, and no ACCEPT/REJECT verdict —
 * interpretation belongs to the estimator-health layer (feature spec FR-003).
 * Consumers outside this package must depend on this interface and {@link Summary}, never on
 * BoofCV's motion-estimation internals.
 *
 * <p><b>Maturity:</b> {@code research-grade}. Validated by executable equivalence and
 * known-answer tests on synthetic imagery (evidence tier T1); no flight data has exercised it.
 *
 * <h2>What the numbers are, and where they come from</h2>
 *
 * <p>The estimator fits a frame-to-keyframe homography to KLT track correspondences with RANSAC.
 * For each correspondence the estimator accepted (its <em>match set</em>, i.e. the inliers), the
 * residual reported here is BoofCV's own {@code DistanceHomographySq}: the <b>squared</b>
 * Euclidean distance, in <b>input-image pixels</b>, between the keyframe point mapped through the
 * estimated homography and where the tracker actually observed it in the current frame.
 *
 * <p>Units are squared pixels of the frame handed to the estimator — <em>not</em> mosaic pixels
 * and <em>not</em> metres. No metric conversion is offered: the VO front end never estimates
 * {@code z} ({@code COMP-001}), so no scale exists to convert with.
 *
 * <h2>Provenance of each quantity (feature spec FR-013)</h2>
 *
 * <p>Three categories are distinguished, because the difference matters for what may be claimed:
 *
 * <table>
 *   <caption>Quantity provenance</caption>
 *   <tr><th>Quantity</th><th>Category</th></tr>
 *   <tr><td>{@link Summary#inlierCount()}</td>
 *       <td><b>(a) retained by the shipped estimator.</b> It is exactly the size of the match set
 *       RANSAC keeps; ddogleg's {@code Ransac.getFitQuality()} returns this same count.</td></tr>
 *   <tr><td>{@link Summary#meanSquaredPx()}, {@link Summary#rmsPx()},
 *           {@link Summary#medianSquaredPx()}, {@link Summary#maxSquaredPx()}</td>
 *       <td><b>(b) recomputed from retained state, without re-running estimation.</b> RANSAC does
 *       compute these per-point distances internally while selecting its match set, but discards
 *       them — in ddogleg 0.23 {@code Ransac.selectMatchSet} holds each distance in a local
 *       variable and keeps only the point. They are therefore <em>not</em> retained, and are
 *       recomputed here by evaluating the same public distance function over the estimator's own
 *       retained match set and retained model. No sampling, no model fitting, and no image
 *       processing is repeated.</td></tr>
 *   <tr><td>{@link #inlierThresholdSquaredPx()}</td>
 *       <td><b>(a)</b> — an echo of the configured RANSAC threshold, so a consumer can interpret
 *       magnitudes without re-deriving the estimator's configuration (FR-005).</td></tr>
 * </table>
 *
 * <p>Nothing exposed here falls in category (c) — nothing required changing or duplicating the
 * estimation path.
 *
 * <h2>Timing</h2>
 *
 * <p>A {@link Summary} describes the <b>most recently processed frame</b> and is captured during
 * that frame's processing, so it is safe to read at any point before the next frame is submitted.
 * Capturing during processing is not an optimisation but a correctness requirement: BoofCV's
 * respawn logic may reset the keyframe reference immediately after the motion is estimated, which
 * would make a residual computed afterwards read as a spurious near-perfect fit.
 *
 * <h2>Absence</h2>
 *
 * <p>When no motion was estimated against the current keyframe — the first frame of a sequence, or
 * a frame whose estimation failed — {@link Summary#UNAVAILABLE} is returned, whose fields are all
 * {@code null}. Absence is never reported as a measured zero (FR-004). A genuine perfect fit is a
 * legitimate measurement and is reported as {@code 0.0}.
 */
public interface MotionResidualDiagnostics {

    /**
     * The configured RANSAC inlier threshold, in squared input-image pixels — the same
     * {@code inlierThresholdSq} the estimator was built with.
     *
     * <p>This is configuration, not a measurement, so it is always known and is not nullable.
     * Every residual in a {@link Summary} was, by construction, below this value at the moment
     * RANSAC selected its match set. (With model refinement enabled the shipped model is refined
     * after selection, so an individual residual measured against it may exceed the threshold;
     * refinement is off by default in this repository.)
     */
    double inlierThresholdSquaredPx();

    /**
     * Residual summary for the most recently processed frame, or {@link Summary#UNAVAILABLE} if no
     * motion was estimated for it. Never {@code null}.
     */
    Summary summarize();

    /**
     * Per-inlier squared reprojection residuals for the most recently processed frame, in
     * input-image pixels squared, in the estimator's own match-set order.
     *
     * <p>Returns {@code null} when unavailable — never an empty array standing in for absence.
     * The returned array is a fresh copy; mutating it does not affect the estimator.
     *
     * <p>The <em>identity</em> of the inliers (which tracks, and where they sit in the image) is
     * not duplicated here: it is already reachable through BoofCV's public {@code AccessPointTracks}
     * view of the motion estimator, which the existing per-frame count diagnostics already use.
     */
    @Nullable
    double[] inlierSquaredResidualsPx();

    /**
     * Immutable per-frame residual record. Every field is independently nullable so that absence is
     * always representable and never collides with a measured value.
     *
     * @param inlierCount     size of the match set the residuals were computed over, or {@code null}
     *                        if no motion was estimated. Reported so that a residual computed over a
     *                        handful of correspondences is not mistaken for one computed over
     *                        thousands (FR-006).
     * @param meanSquaredPx   arithmetic mean of the squared residuals, in px².
     * @param rmsPx           root-mean-square residual, in <b>px</b> (the square root of
     *                        {@code meanSquaredPx}). Provided because a linear-pixel quantity is the
     *                        one most readily compared against image scale.
     * @param medianSquaredPx median squared residual, in px². For an even-sized match set this is
     *                        the lower of the two central values, not their average, so the reported
     *                        value is always one the estimator actually produced.
     * @param maxSquaredPx    largest squared residual in the match set, in px².
     */
    record Summary(
            @Nullable Integer inlierCount,
            @Nullable Double meanSquaredPx,
            @Nullable Double rmsPx,
            @Nullable Double medianSquaredPx,
            @Nullable Double maxSquaredPx) {

        /** No motion was estimated for the frame. Distinct from a measured value of zero. */
        public static final Summary UNAVAILABLE = new Summary(null, null, null, null, null);

        /**
         * Whether residual statistics were measured for this frame.
         *
         * <p>Note this is false both when no motion was estimated and when motion was estimated
         * over an empty match set; {@link #inlierCount()} distinguishes those two (it is
         * {@code null} in the first case and {@code 0} in the second).
         */
        public boolean hasResiduals() {
            return meanSquaredPx != null;
        }
    }
}
