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
import georegression.struct.point.Point2D_F64;
import org.boofcv.stitching.MotionModelSupport;
import org.boofcv.stitching.RigidMotionDecomposition;
import org.ddogleg.fitting.modelset.ModelFitter;
import org.ddogleg.fitting.modelset.ModelMatcherPost;
import org.ddogleg.fitting.modelset.ransac.Ransac;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * {@code EXP-VO-010} Phase 3: the minimal-sample winner against the refined model, on <b>the same
 * RANSAC run</b>, over a known-answer transform.
 *
 * <h2>Why the comparison is exactly paired</h2>
 *
 * <p>Each trial runs {@code Ransac.process} <b>once</b> and then reads both models out of that one
 * result: {@code getModelParameters()} is the minimal-sample winner and
 * {@code fitModel(getMatchSet(), …)} is what refinement would produce from it. So the two arms share
 * the correspondence set, the random draws, the winning hypothesis and the inlier set exactly — the
 * only difference is the final fit, which is precisely the ablation. This is the same arithmetic
 * {@code ImageMotionPointTrackerKey.process} performs, in the same order.
 *
 * <p>Error is measured against the <b>generating transform</b>, not against the other arm, in the
 * four quantities navigation consumes and one that summarises the whole model:
 *
 * <ul>
 *   <li><b>rotation</b> — polar rotation of the Jacobian at the image centre, i.e. exactly what
 *       {@code RigidNavigationState} integrates;</li>
 *   <li><b>translation</b> — displacement of the image centre, i.e. exactly what it integrates for
 *       position, in pixels;</li>
 *   <li><b>scale</b> — {@code √|det J|} at the centre, {@code DEC-VO-004}'s observable;</li>
 *   <li><b>reprojection</b> — rms over <em>all</em> correspondences, including the ones RANSAC
 *       rejected, so a model that fits its inliers by warping everything else is visible.</li>
 * </ul>
 *
 * <p>Emits one row per (model, condition). <b>Tier 1 (synthetic).</b> No imagery, no flight data.
 */
public final class RefinementMonteCarlo {

    /** BoofCV's hardcoded RANSAC seed, so this samples as the pipeline does. */
    private static final long RANSAC_SEED = 123123;

    public enum Model { HOMOGRAPHY, AFFINE }

    /** How the correspondences are laid out in the frame. */
    public enum Spread {
        /** Uniform over the whole frame — the well-conditioned case. */
        DISTRIBUTED,
        /** Confined to one quadrant — the ill-conditioned case a textureless region produces. */
        CLUSTERED
    }

    public record Condition(String name, Model model, int n, double sigmaPx, double outlierFraction,
                            Spread spread, double perspective) {
    }

    public record Result(String condition, String model, int n, double sigmaPx,
                         double outlierFraction, String spread, double perspective,
                         int trials, int failures,
                         double minimalInliers, double refinedInliers,
                         double minRotBiasDeg, double minRotSdDeg, double minRotRmsDeg,
                         double refRotBiasDeg, double refRotSdDeg, double refRotRmsDeg,
                         double minTransRmsPx, double refTransRmsPx,
                         double minScaleRmsRel, double refScaleRmsRel,
                         double minReprojRmsPx, double refReprojRmsPx) {
    }

    private final int width;
    private final int height;

    public RefinementMonteCarlo(int width, int height) {
        this.width = width;
        this.height = height;
    }

    // ------------------------------------------------------------------ truth and synthesis

    /** The generating transform: a small rotation, scale and translation, optionally projective. */
    private Homography2D_F64 truth(double thetaRad, double scale, double tx, double ty,
                                   double perspective) {
        double cx = width / 2.0, cy = height / 2.0;
        double c = Math.cos(thetaRad) * scale, s = Math.sin(thetaRad) * scale;
        // Built about the image centre so that `tx, ty` really is the centre's displacement.
        Homography2D_F64 h = new Homography2D_F64(
                c, -s, cx + tx - c * cx + s * cy,
                s, c, cy + ty - s * cx - c * cy,
                perspective / width, -perspective / height, 1);
        return h;
    }

    private List<AssociatedPair> synthesise(Random rand, Condition cond, Homography2D_F64 truth) {
        List<AssociatedPair> out = new ArrayList<>(cond.n());
        for (int i = 0; i < cond.n(); i++) {
            double px, py;
            if (cond.spread() == Spread.CLUSTERED) {
                // One quadrant, which is what a frame with texture on only one side gives the fit.
                px = rand.nextDouble() * width * 0.4;
                py = rand.nextDouble() * height * 0.4;
            } else {
                px = rand.nextDouble() * width;
                py = rand.nextDouble() * height;
            }
            Point2D_F64 q = MotionModelSupport.HOMOGRAPHY.apply(truth, px, py, null);
            if (rand.nextDouble() < cond.outlierFraction()) {
                q.setTo(rand.nextDouble() * width, rand.nextDouble() * height);
            } else if (cond.sigmaPx() > 0) {
                q.x += rand.nextGaussian() * cond.sigmaPx();
                q.y += rand.nextGaussian() * cond.sigmaPx();
            }
            out.add(new AssociatedPair(px, py, q.x, q.y));
        }
        return out;
    }

    // ------------------------------------------------------------------ readout

    private record Readout(double rotationRad, double scale, double tx, double ty) {
    }

    private Readout read(Homography2D_F64 h) {
        double cx = width / 2.0, cy = height / 2.0;
        double[] j = MotionModelSupport.HOMOGRAPHY.jacobian(h, cx, cy, null);
        RigidMotionDecomposition d = new RigidMotionDecomposition().set(j[0], j[1], j[2], j[3]);
        Point2D_F64 c = MotionModelSupport.HOMOGRAPHY.apply(h, cx, cy, null);
        return new Readout(d.getRotationRad(), d.getUniformScale(), c.x - cx, c.y - cy);
    }

    /** RMS reprojection over <em>every</em> correspondence, rejected ones included. */
    private double reprojection(Homography2D_F64 h, List<AssociatedPair> pairs) {
        double sum = 0;
        for (AssociatedPair p : pairs) {
            Point2D_F64 q = MotionModelSupport.HOMOGRAPHY.apply(h, p.p1.x, p.p1.y, null);
            sum += (q.x - p.p2.x) * (q.x - p.p2.x) + (q.y - p.p2.y) * (q.y - p.p2.y);
        }
        return Math.sqrt(sum / pairs.size());
    }

    // ------------------------------------------------------------------ the measurement

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Result measure(Condition cond, int trials, long seed,
                          double thetaRad, double scale, double tx, double ty,
                          int ransacIterations, double thresholdSq) {

        Random rand = new Random(seed);
        Homography2D_F64 truthH = truth(thetaRad, scale, tx, ty, cond.perspective());
        Readout truthReadout = read(truthH);

        ModelMatcherPost matcher;
        ModelFitter refiner;
        if (cond.model() == Model.AFFINE) {
            matcher = new Ransac<>(RANSAC_SEED, ransacIterations, thresholdSq,
                    new ModelManagerAffine2D_F64(), AssociatedPair.class);
            matcher.setModel(GenerateAffine2D::new, DistanceAffine2DSq::new);
            refiner = new GenerateAffine2D();
        } else {
            matcher = new Ransac<>(RANSAC_SEED, ransacIterations, thresholdSq,
                    new ModelManagerHomography2D_F64(), AssociatedPair.class);
            matcher.setModel(() -> new GenerateHomographyLinear(true), DistanceHomographySq::new);
            refiner = new GenerateHomographyLinear(true);
        }
        MotionModelSupport support = cond.model() == Model.AFFINE
                ? MotionModelSupport.AFFINE : MotionModelSupport.HOMOGRAPHY;

        Accum min = new Accum();
        Accum ref = new Accum();
        double inlierSum = 0;
        int ok = 0, failures = 0;

        for (int t = 0; t < trials; t++) {
            List<AssociatedPair> pairs = synthesise(rand, cond, truthH);
            if (!matcher.process(pairs)) {
                failures++;
                continue;
            }
            List<AssociatedPair> matchSet = matcher.getMatchSet();
            inlierSum += matchSet.size();

            Object minimalModel = matcher.getModelParameters();
            Object refinedModel = cond.model() == Model.AFFINE
                    ? new Affine2D_F64() : new Homography2D_F64();
            if (!refiner.fitModel(matchSet, minimalModel, refinedModel)) {
                failures++;
                continue;
            }

            Homography2D_F64 mh = support.asHomography(
                    (georegression.struct.InvertibleTransform) minimalModel, null);
            Homography2D_F64 rh = support.asHomography(
                    (georegression.struct.InvertibleTransform) refinedModel, null);
            min.add(read(mh), truthReadout, reprojection(mh, pairs));
            ref.add(read(rh), truthReadout, reprojection(rh, pairs));
            ok++;
        }

        return new Result(cond.name(), cond.model().name(), cond.n(), cond.sigmaPx(),
                cond.outlierFraction(), cond.spread().name(), cond.perspective(),
                trials, failures,
                ok > 0 ? inlierSum / ok : Double.NaN, ok > 0 ? inlierSum / ok : Double.NaN,
                min.biasDeg(ok), min.sdDeg(ok), min.rmsDeg(ok),
                ref.biasDeg(ok), ref.sdDeg(ok), ref.rmsDeg(ok),
                min.transRms(ok), ref.transRms(ok),
                min.scaleRms(ok), ref.scaleRms(ok),
                min.reprojRms(ok), ref.reprojRms(ok));
    }

    /** Running sums for one arm. Rotation is wrapped before accumulation. */
    private static final class Accum {
        double rot, rotSq, transSq, scaleSq, reprojSq;

        void add(Readout got, Readout truth, double reproj) {
            double dr = wrap(got.rotationRad() - truth.rotationRad());
            rot += dr;
            rotSq += dr * dr;
            double dx = got.tx() - truth.tx(), dy = got.ty() - truth.ty();
            transSq += dx * dx + dy * dy;
            double ds = (got.scale() - truth.scale()) / truth.scale();
            scaleSq += ds * ds;
            reprojSq += reproj * reproj;
        }

        double biasDeg(int n) { return n > 0 ? Math.toDegrees(rot / n) : Double.NaN; }

        double rmsDeg(int n) { return n > 0 ? Math.toDegrees(Math.sqrt(rotSq / n)) : Double.NaN; }

        double sdDeg(int n) {
            if (n < 2) return Double.NaN;
            double mean = rot / n;
            return Math.toDegrees(Math.sqrt(Math.max(0, (rotSq - n * mean * mean) / (n - 1))));
        }

        double transRms(int n) { return n > 0 ? Math.sqrt(transSq / n) : Double.NaN; }

        double scaleRms(int n) { return n > 0 ? Math.sqrt(scaleSq / n) : Double.NaN; }

        double reprojRms(int n) { return n > 0 ? Math.sqrt(reprojSq / n) : Double.NaN; }

        private static double wrap(double a) {
            while (a > Math.PI) a -= 2 * Math.PI;
            while (a < -Math.PI) a += 2 * Math.PI;
            return a;
        }
    }
}
