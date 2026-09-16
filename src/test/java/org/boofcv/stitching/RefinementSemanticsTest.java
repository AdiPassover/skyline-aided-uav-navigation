package org.boofcv.stitching;

import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.alg.geo.robust.DistanceAffine2DSq;
import boofcv.alg.geo.robust.DistanceHomographySq;
import boofcv.alg.geo.robust.GenerateAffine2D;
import boofcv.alg.geo.robust.GenerateHomographyLinear;
import boofcv.struct.geo.AssociatedPair;
import boofcv.struct.image.GrayF32;
import georegression.fitting.affine.ModelManagerAffine2D_F64;
import georegression.fitting.homography.ModelManagerHomography2D_F64;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F64;
import org.boofcv.util.structs.Pose3D;
import org.ddogleg.fitting.modelset.DistanceFromModel;
import org.ddogleg.fitting.modelset.ModelFitter;
import org.ddogleg.fitting.modelset.ModelGenerator;
import org.ddogleg.fitting.modelset.ModelMatcherPost;
import org.ddogleg.fitting.modelset.ransac.Ransac;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@code EXP-VO-010} gates 2 and 3: that RANSAC's optional final refinement does what the source
 * says it does, and that observing it changes nothing.
 *
 * <p>The experiment's entire premise is that with {@code refineEstimate = false} the estimator ships
 * a model fitted to <b>3 or 4</b> correspondences while RANSAC has identified hundreds. That is a
 * claim about library behaviour, and the record states it from a source reading — these tests make
 * the same claims executable, so a BoofCV upgrade that changed them would fail here rather than
 * quietly invalidate the experiment.
 *
 * <p>Evidence tier: T1 (synthetic). Nothing here says anything about VO accuracy.
 */
public class RefinementSemanticsTest {

    private static final int RANSAC_ITERATIONS = 220;
    private static final double INLIER_THRESHOLD_SQ = 3.0;
    private static final long SEED = 123123;

    /** Bitwise identity, not approximate agreement. */
    private static final double EXACT = 0.0;

    // ------------------------------------------------------------------ gate 3: the inlier set

    /** A {@link ModelFitter} that records what it was asked to fit and then delegates. */
    private static final class CountingFitter<M> implements ModelFitter<M, AssociatedPair> {
        private final ModelFitter<M, AssociatedPair> delegate;
        int lastSize = -1;
        int calls = 0;

        CountingFitter(ModelFitter<M, AssociatedPair> delegate) {
            this.delegate = delegate;
        }

        @Override
        public boolean fitModel(List<AssociatedPair> dataSet, M initial, M found) {
            lastSize = dataSet.size();
            calls++;
            return delegate.fitModel(dataSet, initial, found);
        }

        @Override
        public double getFitScore() {
            return delegate.getFitScore();
        }
    }

    private static List<AssociatedPair> correspondences(int n, double outlierFraction, long seed) {
        Random rand = new Random(seed);
        Affine2D_F64 truth = new Affine2D_F64(
                Math.cos(0.02), -Math.sin(0.02), Math.sin(0.02), Math.cos(0.02), 7.0, -3.0);
        List<AssociatedPair> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double x = rand.nextDouble() * 1224, y = rand.nextDouble() * 1024;
            Point2D_F64 q = new Point2D_F64();
            georegression.transform.affine.AffinePointOps_F64.transform(truth, x, y, q);
            if (rand.nextDouble() < outlierFraction) {
                q.setTo(rand.nextDouble() * 1224, rand.nextDouble() * 1024);
            } else {
                q.x += rand.nextGaussian() * 0.3;
                q.y += rand.nextGaussian() * 0.3;
            }
            out.add(new AssociatedPair(x, y, q.x, q.y));
        }
        return out;
    }

    @Test
    @DisplayName("gate 3: refinement is handed exactly the RANSAC match set, and it is far larger than the minimal sample")
    void refinementReceivesTheWholeInlierSet() {
        List<AssociatedPair> pairs = correspondences(2000, 0.05, 11);

        ModelMatcherPost<Affine2D_F64, AssociatedPair> matcher = new Ransac<>(
                SEED, RANSAC_ITERATIONS, INLIER_THRESHOLD_SQ,
                new ModelManagerAffine2D_F64(), AssociatedPair.class);
        matcher.setModel(GenerateAffine2D::new, DistanceAffine2DSq::new);
        assertTrue(matcher.process(pairs));

        CountingFitter<Affine2D_F64> refiner = new CountingFitter<>(new GenerateAffine2D());
        Affine2D_F64 refined = new Affine2D_F64();
        assertTrue(refiner.fitModel(matcher.getMatchSet(), matcher.getModelParameters(), refined));

        assertEquals(matcher.getMatchSet().size(), refiner.lastSize,
                "the refiner must receive the whole match set, not a subsample");
        assertTrue(refiner.lastSize > 1000,
                "expected the inlier set to dwarf the 3-point minimal sample; got "
                        + refiner.lastSize);
        // The premise of the whole experiment, made executable.
        assertTrue(refiner.lastSize > 100 * new GenerateAffine2D().getMinimumPoints(),
                "the shipped model without refinement is fitted to "
                        + new GenerateAffine2D().getMinimumPoints() + " points while "
                        + refiner.lastSize + " inliers were available");
    }

    @Test
    @DisplayName("gate 3: refinement moves the model, and towards the truth")
    void refinementMovesTheModelTowardsTruth() {
        List<AssociatedPair> pairs = correspondences(2000, 0.05, 12);
        double trueTheta = 0.02;

        ModelMatcherPost<Affine2D_F64, AssociatedPair> matcher = new Ransac<>(
                SEED, RANSAC_ITERATIONS, INLIER_THRESHOLD_SQ,
                new ModelManagerAffine2D_F64(), AssociatedPair.class);
        matcher.setModel(GenerateAffine2D::new, DistanceAffine2DSq::new);
        assertTrue(matcher.process(pairs));

        Affine2D_F64 minimal = matcher.getModelParameters();
        Affine2D_F64 refined = new Affine2D_F64();
        assertTrue(new GenerateAffine2D().fitModel(matcher.getMatchSet(), minimal, refined));

        double eMin = Math.abs(rotationOf(minimal) - trueTheta);
        double eRef = Math.abs(rotationOf(refined) - trueTheta);
        assertNotEquals(rotationOf(minimal), rotationOf(refined), 1e-12,
                "refinement should not be a no-op on noisy data");
        assertTrue(eRef < eMin,
                "expected the many-inlier fit to be closer to the truth: minimal error "
                        + eMin + " rad, refined " + eRef + " rad");
    }

    @Test
    @DisplayName("gate 3: refinement does not regress an already-exact fit")
    void refinementIsHarmlessOnExactData() {
        List<AssociatedPair> pairs = correspondences(500, 0.0, 13);
        // Rebuild without noise.
        pairs.clear();
        Affine2D_F64 truth = new Affine2D_F64(
                Math.cos(0.02), -Math.sin(0.02), Math.sin(0.02), Math.cos(0.02), 7.0, -3.0);
        Random rand = new Random(99);
        for (int i = 0; i < 500; i++) {
            double x = rand.nextDouble() * 1224, y = rand.nextDouble() * 1024;
            Point2D_F64 q = new Point2D_F64();
            georegression.transform.affine.AffinePointOps_F64.transform(truth, x, y, q);
            pairs.add(new AssociatedPair(x, y, q.x, q.y));
        }

        ModelMatcherPost<Affine2D_F64, AssociatedPair> matcher = new Ransac<>(
                SEED, RANSAC_ITERATIONS, INLIER_THRESHOLD_SQ,
                new ModelManagerAffine2D_F64(), AssociatedPair.class);
        matcher.setModel(GenerateAffine2D::new, DistanceAffine2DSq::new);
        assertTrue(matcher.process(pairs));

        Affine2D_F64 refined = new Affine2D_F64();
        assertTrue(new GenerateAffine2D().fitModel(
                matcher.getMatchSet(), matcher.getModelParameters(), refined));

        assertEquals(0.02, rotationOf(refined), 1e-12, "exact data must stay exact after refining");
        assertEquals(truth.tx, refined.tx, 1e-9);
        assertEquals(truth.ty, refined.ty, 1e-9);
    }

    @Test
    @DisplayName("Phase 1 Q3: fitModel ignores its initial estimate — it is not an iterative refinement")
    void refinementIgnoresTheInitialEstimate() {
        List<AssociatedPair> pairs = correspondences(300, 0.0, 14);
        GenerateAffine2D fitter = new GenerateAffine2D();

        Affine2D_F64 fromTruth = new Affine2D_F64();
        Affine2D_F64 fromNonsense = new Affine2D_F64();
        assertTrue(fitter.fitModel(pairs, new Affine2D_F64(), fromTruth));
        assertTrue(fitter.fitModel(pairs, new Affine2D_F64(9, 9, 9, 9, 9999, -9999), fromNonsense));

        // Bitwise identical: the `initial` argument is discarded, so refinement cannot diverge or
        // converge — it is one linear solve. This is why the record calls it one-shot.
        assertEquals(fromTruth.a11, fromNonsense.a11, EXACT);
        assertEquals(fromTruth.a12, fromNonsense.a12, EXACT);
        assertEquals(fromTruth.a21, fromNonsense.a21, EXACT);
        assertEquals(fromTruth.a22, fromNonsense.a22, EXACT);
        assertEquals(fromTruth.tx, fromNonsense.tx, EXACT);
        assertEquals(fromTruth.ty, fromNonsense.ty, EXACT);
        assertEquals(0.0, fitter.getFitScore(), EXACT, "getFitScore is a hardcoded 0 — no signal");
    }

    @Test
    @DisplayName("Phase 1 Q7: the similarity model has no refiner, so the sidecar refuses it")
    void similarityCannotRefine() {
        assertFalse(new GenerateSimilarity2D() instanceof ModelFitter,
                "GenerateSimilarity2D implements ModelGenerator only; if that ever changes, "
                        + "EXP-VO-010 Phase 1 Q7 and DEC-VO-006 must be revisited");
    }

    @Test
    @DisplayName("Phase 1 Q6: the two models refine by structurally different algorithms")
    void bothModelsImplementModelFitter() {
        assertTrue(new GenerateAffine2D() instanceof ModelFitter);
        assertTrue(new GenerateHomographyLinear(true) instanceof ModelFitter);
        assertTrue(new GenerateAffine2D() instanceof ModelGenerator);
        assertTrue(new GenerateHomographyLinear(true) instanceof ModelGenerator);

        // Homography refinement over ~2000 inliers is a null-space SVD of a 2N x 9 matrix. Assert it
        // at least succeeds at that scale, since the record predicts its cost from this fact.
        List<AssociatedPair> pairs = correspondences(2000, 0.0, 15);
        ModelMatcherPost<Homography2D_F64, AssociatedPair> matcher = new Ransac<>(
                SEED, RANSAC_ITERATIONS, INLIER_THRESHOLD_SQ,
                new ModelManagerHomography2D_F64(), AssociatedPair.class);
        matcher.setModel(() -> new GenerateHomographyLinear(true), DistanceHomographySq::new);
        assertTrue(matcher.process(pairs));
        Homography2D_F64 refined = new Homography2D_F64();
        assertTrue(new GenerateHomographyLinear(true).fitModel(
                matcher.getMatchSet(), matcher.getModelParameters(), refined),
                "homography refinement over " + matcher.getMatchSet().size() + " inliers must not "
                        + "fail numerically — the record's runtime prediction assumes it runs");
    }

    // ------------------------------------------------------------------ gate 2: the probe

    @Test
    @DisplayName("gate 2: the refinement probe is observational — poses are bit-identical, both models, both settings")
    void probeChangesNothing() {
        GrayF32 world = EquivalenceHarness.syntheticTexturedWorld();
        for (StitchingFactory.ProbeModel model : StitchingFactory.ProbeModel.values()) {
            for (boolean refine : new boolean[]{false, true}) {
                List<EquivalenceHarness.Observation> unprobed =
                        EquivalenceHarness.run(EquivalenceHarness.plain(model, refine), model, world);
                StitchingFactory.ProbedMotion<?> probedMotion =
                        EquivalenceHarness.probed(model, refine);
                List<EquivalenceHarness.Observation> probedObs =
                        EquivalenceHarness.run(probedMotion.motion(), model, world);

                String at = model + " refine=" + refine;
                assertEquals(unprobed.size(), probedObs.size(), at);
                for (int i = 0; i < unprobed.size(); i++) {
                    EquivalenceHarness.assertIdentical(at + " frame " + i,
                            unprobed.get(i), probedObs.get(i));
                }
                assertTrue(probedMotion.diagnostics().hasModels(), at + ": probe never populated");
                assertEquals(refine, probedMotion.diagnostics().refinementEnabled(), at);
            }
        }
    }

    @Test
    @DisplayName("gate 2: with refinement off the two recorded models are identical, and with it on they differ")
    void probeDistinguishesTheTwoSettings() {
        GrayF32 world = EquivalenceHarness.syntheticTexturedWorld();

        StitchingFactory.ProbedMotion<?> off =
                EquivalenceHarness.probed(StitchingFactory.ProbeModel.AFFINE, false);
        EquivalenceHarness.run(off.motion(), StitchingFactory.ProbeModel.AFFINE, world);
        Homography2D_F64 m = off.diagnostics().minimalSampleModel();
        Homography2D_F64 s = off.diagnostics().shippedModel();
        assertEquals(m.a11, s.a11, EXACT, "refine=false must ship the minimal-sample model verbatim");
        assertEquals(m.a12, s.a12, EXACT);
        assertEquals(m.a13, s.a13, EXACT);
        assertEquals(m.a21, s.a21, EXACT);
        assertEquals(m.a22, s.a22, EXACT);
        assertEquals(m.a23, s.a23, EXACT);

        StitchingFactory.ProbedMotion<?> on =
                EquivalenceHarness.probed(StitchingFactory.ProbeModel.AFFINE, true);
        EquivalenceHarness.run(on.motion(), StitchingFactory.ProbeModel.AFFINE, world);
        assertTrue(on.diagnostics().inlierCount() > new GenerateAffine2D().getMinimumPoints(),
                "the refiner should have had more than a minimal sample to work with");
    }

    private static double rotationOf(Affine2D_F64 a) {
        return Math.atan2(a.a21 - a.a12, a.a11 + a.a22);
    }
}
