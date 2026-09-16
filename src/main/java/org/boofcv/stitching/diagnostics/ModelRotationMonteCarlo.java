package org.boofcv.stitching.diagnostics;

import boofcv.alg.geo.robust.DistanceAffine2DSq;
import boofcv.alg.geo.robust.DistanceHomographySq;
import boofcv.alg.geo.robust.GenerateAffine2D;
import boofcv.alg.geo.robust.GenerateHomographyLinear;
import boofcv.struct.geo.AssociatedPair;
import georegression.fitting.affine.ModelManagerAffine2D_F64;
import georegression.fitting.homography.ModelManagerHomography2D_F64;
import georegression.struct.affine.Affine2D_F64;
import georegression.struct.homography.Homography2D_F64;
import org.boofcv.stitching.DistanceSimilarity2DSq;
import org.boofcv.stitching.GenerateSimilarity2D;
import org.boofcv.stitching.ModelManagerSim2_F64;
import org.boofcv.stitching.MotionModelSupport;
import org.boofcv.stitching.RigidMotionDecomposition;
import org.boofcv.stitching.Sim2_F64;
import org.ddogleg.fitting.modelset.DistanceFromModel;
import org.ddogleg.fitting.modelset.ModelGenerator;
import org.ddogleg.fitting.modelset.ModelManager;
import org.ddogleg.fitting.modelset.ModelMatcherPost;
import org.ddogleg.fitting.modelset.ransac.Ransac;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * {@code EXP-VO-009} Phase 5: rotation bias and variance of the three motion models, measured on
 * <b>identical</b> synthetic correspondence sets through the <b>real</b> estimators.
 *
 * <h2>Why this is in Java and not in the Python tooling</h2>
 *
 * <p>Because the question is about the estimators the pipeline actually runs, not about their
 * mathematical idealisations. This probe drives BoofCV 0.44's own {@code GenerateHomographyLinear}
 * and {@code GenerateAffine2D} and this repository's {@link GenerateSimilarity2D} through ddogleg
 * 0.23's own {@code Ransac}, configured with the same hardcoded seed 123123, the same 220 iterations
 * and the same squared-pixel inlier threshold the flight runs use. The rotation is then read exactly
 * as {@code RigidNavigationState} reads it: the model's Jacobian at the image centre, through
 * {@link RigidMotionDecomposition}'s polar decomposition. A Python re-implementation would be
 * measuring a second implementation's variance, which is precisely the confound {@code DEC-VO-002}
 * was written to avoid.
 *
 * <h2>What it separates</h2>
 *
 * <p>{@code LIT-VO-005} §4 gives the closed form
 * {@code Var(θ_affine)/Var(θ_sim) = (2 + κ + 1/κ)/4} and bounds the realised gain between ≈ 1.03
 * (fitting the full inlier set, κ ≈ 1.43) and ≈ 100 (blind 3-point sampling), without predicting
 * where RANSAC's best-of-220 selection lands. Three fitting modes are therefore reported side by
 * side:
 *
 * <ul>
 *   <li>{@code RANSAC} — what ships: the winning minimal-sample hypothesis, {@code refineEstimate}
 *       false.</li>
 *   <li>{@code REFINED} — the same RANSAC, then the generator re-run over the whole inlier set.
 *       This is the {@code refineEstimate = true} arm, measured here on tier-1 data <b>only</b>;
 *       {@code EXP-VO-009} deliberately does not enable it on any flight run.</li>
 *   <li>{@code ALL} — the generator over every correspondence, no RANSAC. The noise-floor
 *       reference.</li>
 * </ul>
 *
 * <p>A fourth arm, {@code similarity-2pt}, uses the model's true 2-point minimum instead of the 3
 * that keeps it sample-matched to affine — so the effect of over-determination is measured rather
 * than left as a hidden confound ({@code DEC-VO-006} <i>Rationale</i>).
 *
 * <p>Emits one JSON array on stdout. <b>Tier 1 (synthetic).</b> No imagery, no flight data, no
 * ground truth beyond the transform that generated the correspondences.
 */
public final class ModelRotationMonteCarlo {

    /** BoofCV's hardcoded RANSAC seed, replicated so this probe samples as the pipeline does. */
    private static final long RANSAC_SEED = 123123;

    /** How the model is fitted once RANSAC has (or has not) chosen an inlier set. */
    public enum FitMode { RANSAC, REFINED, ALL }

    public record Arm(String name, ModelKind kind, int minimumPoints) {}

    public enum ModelKind { HOMOGRAPHY, AFFINE, SIMILARITY }

    /** One measured configuration. */
    public record Result(String arm, String fitMode, double sigmaPx, int nPoints,
                         double outlierFraction, double anisotropy, double perspective,
                         int trials, int failures,
                         double meanRotationErrorDeg, double sdRotationErrorDeg,
                         double rmsRotationErrorDeg, double p95AbsRotationErrorDeg,
                         double meanScaleError, double meanInliers, double meanKappa) {}

    private final int width;
    private final int height;

    public ModelRotationMonteCarlo(int width, int height) {
        this.width = width;
        this.height = height;
    }

    // ------------------------------------------------------------------ correspondence synthesis

    /**
     * A frame's worth of correspondences from a known similarity, optionally violated.
     *
     * @param anisotropy  {@code σ₁/σ₂} of an extra symmetric stretch applied along x — 1.0 for none.
     *                    {@code LIT-VO-003} §4 gives {@code cos θ_tilt}, i.e. 1.004–1.037 for the
     *                    tilts measured on these flights.
     * @param perspective magnitude of an extra perspective row, in the units of
     *                    {@code MotionModelSupport.perspectiveMagnitude} (max edge displacement, px)
     */
    private List<AssociatedPair> synthesise(Random rand, int n, double scale, double thetaRad,
                                            double tx, double ty, double sigmaPx,
                                            double outlierFraction, double anisotropy,
                                            double perspective) {
        double cx = width / 2.0, cy = height / 2.0;
        // Build the truth as a homography about the image centre so that every violation is
        // expressed in one place and the similarity case is exactly recovered when both are off.
        double sx = scale * Math.sqrt(anisotropy);
        double sy = scale / Math.sqrt(anisotropy);
        double c = Math.cos(thetaRad), s = Math.sin(thetaRad);
        double h31 = perspective / width;
        double h32 = -perspective / height;

        List<AssociatedPair> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double px = rand.nextDouble() * width;
            double py = rand.nextDouble() * height;
            double ux = px - cx, uy = py - cy;
            double lx = sx * (c * ux - s * uy);
            double ly = sy * (s * ux + c * uy);
            double w = 1.0 + h31 * ux + h32 * uy;
            double qx = cx + lx / w + tx;
            double qy = cy + ly / w + ty;
            if (rand.nextDouble() < outlierFraction) {
                qx = rand.nextDouble() * width;
                qy = rand.nextDouble() * height;
            } else {
                qx += rand.nextGaussian() * sigmaPx;
                qy += rand.nextGaussian() * sigmaPx;
            }
            out.add(new AssociatedPair(px, py, qx, qy));
        }
        return out;
    }

    /** Condition number of the centred second-moment matrix — {@code κ} of {@code LIT-VO-005} (8). */
    static double kappa(List<AssociatedPair> pairs) {
        double mx = 0, my = 0;
        for (AssociatedPair p : pairs) { mx += p.p1.x; my += p.p1.y; }
        mx /= pairs.size();
        my /= pairs.size();
        double sxx = 0, syy = 0, sxy = 0;
        for (AssociatedPair p : pairs) {
            double dx = p.p1.x - mx, dy = p.p1.y - my;
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
        double tr = sxx + syy;
        double disc = Math.sqrt(Math.max(0.0, (sxx - syy) * (sxx - syy) + 4 * sxy * sxy));
        double l1 = 0.5 * (tr + disc), l2 = 0.5 * (tr - disc);
        return l2 <= 0 ? Double.POSITIVE_INFINITY : l1 / l2;
    }

    // ------------------------------------------------------------------ per-arm estimation

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static ModelMatcherPost buildRansac(Arm arm, int iterations, double thresholdSq) {
        ModelManager manager;
        switch (arm.kind()) {
            case HOMOGRAPHY -> manager = new ModelManagerHomography2D_F64();
            case AFFINE -> manager = new ModelManagerAffine2D_F64();
            default -> manager = new ModelManagerSim2_F64();
        }
        ModelMatcherPost m = new Ransac<>(RANSAC_SEED, iterations, thresholdSq, manager,
                AssociatedPair.class);
        final int min = arm.minimumPoints();
        switch (arm.kind()) {
            case HOMOGRAPHY -> m.setModel(
                    (org.ddogleg.struct.Factory<ModelGenerator<Homography2D_F64, AssociatedPair>>)
                            () -> new GenerateHomographyLinear(true),
                    (org.ddogleg.struct.Factory<DistanceFromModel<Homography2D_F64, AssociatedPair>>)
                            DistanceHomographySq::new);
            case AFFINE -> m.setModel(
                    (org.ddogleg.struct.Factory<ModelGenerator<Affine2D_F64, AssociatedPair>>)
                            GenerateAffine2D::new,
                    (org.ddogleg.struct.Factory<DistanceFromModel<Affine2D_F64, AssociatedPair>>)
                            DistanceAffine2DSq::new);
            default -> m.setModel(
                    (org.ddogleg.struct.Factory<ModelGenerator<Sim2_F64, AssociatedPair>>)
                            () -> new FixedMinimumSimilarity(min),
                    (org.ddogleg.struct.Factory<DistanceFromModel<Sim2_F64, AssociatedPair>>)
                            DistanceSimilarity2DSq::new);
        }
        return m;
    }

    /**
     * {@link GenerateSimilarity2D} with the reported minimum overridden — used only for the
     * {@code similarity-2pt} control arm, which measures the effect of the over-determination that
     * the shipped 3-point choice buys. The fit itself is unchanged.
     */
    private static final class FixedMinimumSimilarity extends GenerateSimilarity2D {
        private final int minimum;

        FixedMinimumSimilarity(int minimum) {
            this.minimum = minimum;
        }

        @Override
        public int getMinimumPoints() {
            return minimum;
        }
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static ModelGenerator generatorFor(Arm arm) {
        return switch (arm.kind()) {
            case HOMOGRAPHY -> new GenerateHomographyLinear(true);
            case AFFINE -> new GenerateAffine2D();
            case SIMILARITY -> new GenerateSimilarity2D();
        };
    }

    /** Reads rotation and scale exactly as {@code RigidNavigationState} does: Jacobian at centre. */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private double[] readout(Arm arm, Object model, RigidMotionDecomposition dec) {
        MotionModelSupport support = switch (arm.kind()) {
            case HOMOGRAPHY -> MotionModelSupport.HOMOGRAPHY;
            case AFFINE -> MotionModelSupport.AFFINE;
            case SIMILARITY -> MotionModelSupport.SIMILARITY;
        };
        double[] j = support.jacobian((georegression.struct.InvertibleTransform) model,
                width / 2.0, height / 2.0, null);
        dec.set(j[0], j[1], j[2], j[3]);
        return new double[]{dec.getRotationRad(), dec.getUniformScale()};
    }

    // ------------------------------------------------------------------ the sweep

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Result measure(Arm arm, FitMode mode, int trials, long seed, int n, double sigmaPx,
                          double outlierFraction, double anisotropy, double perspective,
                          double scale, double thetaRad, int iterations, double thresholdSq) {

        Random rand = new Random(seed);
        ModelMatcherPost matcher = mode == FitMode.ALL ? null
                : buildRansac(arm, iterations, thresholdSq);
        ModelGenerator generator = generatorFor(arm);
        RigidMotionDecomposition dec = new RigidMotionDecomposition();

        double sumErr = 0, sumSq = 0, sumScaleErr = 0, sumInliers = 0, sumKappa = 0;
        int ok = 0, failures = 0;
        double[] absErrs = new double[trials];

        for (int t = 0; t < trials; t++) {
            List<AssociatedPair> pairs = synthesise(rand, n, scale, thetaRad, 3.7, -2.1,
                    sigmaPx, outlierFraction, anisotropy, perspective);
            sumKappa += kappa(pairs);

            Object model;
            int inliers;
            if (mode == FitMode.ALL) {
                model = switch (arm.kind()) {
                    case HOMOGRAPHY -> new Homography2D_F64();
                    case AFFINE -> new Affine2D_F64();
                    case SIMILARITY -> new Sim2_F64();
                };
                if (!generator.generate(pairs, model)) { failures++; continue; }
                inliers = n;
            } else {
                if (!matcher.process(pairs)) { failures++; continue; }
                model = matcher.getModelParameters();
                List<AssociatedPair> matchSet = matcher.getMatchSet();
                inliers = matchSet.size();
                if (mode == FitMode.REFINED) {
                    Object refined = switch (arm.kind()) {
                        case HOMOGRAPHY -> new Homography2D_F64();
                        case AFFINE -> new Affine2D_F64();
                        case SIMILARITY -> new Sim2_F64();
                    };
                    if (!generator.generate(matchSet, refined)) { failures++; continue; }
                    model = refined;
                }
            }

            double[] r = readout(arm, model, dec);
            double err = Math.toDegrees(wrap(r[0] - thetaRad));
            sumErr += err;
            sumSq += err * err;
            absErrs[ok] = Math.abs(err);
            sumScaleErr += r[1] - scale;
            sumInliers += inliers;
            ok++;
        }

        if (ok == 0) {
            return new Result(arm.name(), mode.name(), sigmaPx, n, outlierFraction, anisotropy,
                    perspective, trials, failures, Double.NaN, Double.NaN, Double.NaN, Double.NaN,
                    Double.NaN, Double.NaN, sumKappa / trials);
        }
        double mean = sumErr / ok;
        double var = ok > 1 ? (sumSq - ok * mean * mean) / (ok - 1) : 0.0;
        double[] sorted = java.util.Arrays.copyOf(absErrs, ok);
        java.util.Arrays.sort(sorted);
        return new Result(arm.name(), mode.name(), sigmaPx, n, outlierFraction, anisotropy,
                perspective, trials, failures, mean, Math.sqrt(Math.max(0, var)),
                Math.sqrt(sumSq / ok), sorted[(int) Math.min(ok - 1, Math.round(0.95 * (ok - 1)))],
                sumScaleErr / ok, sumInliers / ok, sumKappa / trials);
    }

    private static double wrap(double a) {
        while (a > Math.PI) a -= 2 * Math.PI;
        while (a < -Math.PI) a += 2 * Math.PI;
        return a;
    }

    public static List<Arm> defaultArms() {
        return List.of(
                new Arm("homography", ModelKind.HOMOGRAPHY, 4),
                new Arm("affine", ModelKind.AFFINE, 3),
                new Arm("similarity", ModelKind.SIMILARITY, 3),
                new Arm("similarity-2pt", ModelKind.SIMILARITY, 2));
    }
}
