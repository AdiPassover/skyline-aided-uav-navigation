package org.boofcv.stitching.diagnostics;

import boofcv.alg.geo.robust.DistanceAffine2DSq;
import boofcv.alg.geo.robust.DistanceHomographySq;
import boofcv.alg.geo.robust.GenerateAffine2D;
import boofcv.alg.geo.robust.GenerateHomographyLinear;
import boofcv.struct.geo.AssociatedPair;
import georegression.fitting.affine.ModelManagerAffine2D_F64;
import georegression.fitting.homography.ModelManagerHomography2D_F64;
import georegression.struct.InvertibleTransform;
import org.boofcv.stitching.GenerateSimilarity2D;
import org.boofcv.stitching.DistanceSimilarity2DSq;
import org.boofcv.stitching.ModelManagerSim2_F64;
import org.boofcv.stitching.MotionModelSupport;
import org.boofcv.stitching.RigidMotionDecomposition;
import org.ddogleg.fitting.modelset.ModelMatcherPost;
import org.ddogleg.fitting.modelset.ransac.Ransac;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * {@code EXP-VO-011} Phases 3-5: what per-frame log-scale bias does a given <em>physical</em>
 * mechanism produce, measured through the real estimators, against an exactly known answer?
 *
 * <h2>What it simulates, and why at the correspondence level</h2>
 *
 * <p>A 3-D ground scene (planar, or with parameterised relief), two exact camera poses, an optional
 * Brown-Conrady radial lens distortion applied to both projections, and optional feature-
 * localisation noise. The correspondences that result go into the same BoofCV RANSAC and the same
 * model generators the pipeline uses, and the fitted transform is read exactly as
 * {@code RigidNavigationState.observe} reads it:
 *
 * <pre>
 *   inc_log_scale = log sqrt(|det J|) of D^-1 at the IMAGE CENTRE
 * </pre>
 *
 * <p>so a positive value means the estimator believes the camera is receding from the surface
 * ({@code LIT-VO-003} eq. 2). This is deliberately a geometry test rather than a rendering test:
 * {@code EXP-VO-005}'s renderer already established what the full pipeline does on ideal planar
 * imagery, and rendering relief would add a tracker-and-appearance confound to a question that is
 * purely about what geometry does to a fit. The rendered distortion arm exists separately, as a
 * confirmation that the correspondence-level answer survives the real tracker.
 *
 * <h2>The known answer</h2>
 *
 * <p>With planar ground, no distortion, no noise and no tilt, every arm must return a bias below
 * {@code 1e-9}: pure translation over a plane induces an exact translation in the image, whose
 * Jacobian is the identity. That is a gate, asserted in {@code ScaleBiasMonteCarloTest}, not a
 * result.
 *
 * <p><b>Tier 1 (synthetic).</b> No imagery, no flight data, no ground-truth tuning.
 */
public final class ScaleBiasMonteCarlo {

    /** BoofCV's hardcoded RANSAC seed, so this samples exactly as the pipeline does. */
    private static final long RANSAC_SEED = 123123;

    public enum Model { HOMOGRAPHY, AFFINE, SIMILARITY }

    /** Terrain within the camera footprint. All are scaled to a requested p95-p5 depth spread. */
    public enum Relief {
        /** Perfectly planar ground -- the control. */
        PLANAR,
        /** A constant gradient across the footprint. */
        SLOPE,
        /** Band-limited random field: a sum of sinusoids with random phases, re-drawn per trial. */
        RANDOM_FIELD,
        /** Sparse elevated structures -- buildings and canopy: a fraction of points raised. */
        STRUCTURES
    }

    /**
     * Where the correspondences sit in the frame.
     *
     * <p>The two LATTICE modes exist because a mechanism that is <em>odd in image position</em>
     * produces an apparent scale proportional to {@code (feature centroid . travel)}, and under
     * random sampling that centroid wanders from trial to trial. The resulting Monte Carlo error
     * swamps the effect: on the first pass every distortion cell came back with
     * {@code |bias| / SE < 1.5}, which is not a measurement of a small effect but an absence of a
     * measurement. A deterministic lattice fixes the centroid exactly, so the geometric answer is
     * returned with zero sampling error and the question becomes the sharp one -- how asymmetric
     * would the feature set have to be?
     */
    public enum Spread {
        /** Uniform random over the whole frame. */
        DISTRIBUTED,
        /** Uniform random over one quadrant -- an asymmetric support. */
        CLUSTERED,
        /** Deterministic grid over the whole frame: centroid exactly at the image centre. */
        LATTICE,
        /** Deterministic grid over one quadrant: the largest asymmetry worth considering. */
        LATTICE_QUADRANT
    }

    /**
     * One measurement condition. Physical quantities are in metres and degrees; `k1`/`k2` are the
     * Brown-Conrady coefficients on radius normalised by the focal length.
     */
    public record Condition(
            String name, Model model, Relief relief,
            double reliefSpreadM,      // target p95-p5 in-footprint depth spread, metres
            double k1, double k2,      // radial distortion; 0, 0 = pinhole
            boolean correctDistortion, // rectify before fitting (the "correct handling" arm)
            double heightM,            // camera height above the mean ground plane
            double dHeightM,           // height change between the two views
            double travelM,            // ground displacement between the two views
            double travelAngleDeg,     // direction of travel in the IMAGE frame
            double dYawDeg,            // in-plane rotation between the two views
            double tiltDeg, double dTiltDeg,   // boresight tilt of view 1, and its change
            double sigmaPx,            // per-coordinate feature-localisation noise
            int baselineFrames,        // frames since the keyframe; 0 = a plain consecutive pair
            int n, Spread spread) {
    }

    public record Result(
            String condition, String model, String relief, double reliefSpreadM,
            double k1, double k2, boolean correctDistortion,
            double heightM, double dHeightM, double travelM, double travelAngleDeg,
            double dYawDeg, double tiltDeg, double dTiltDeg, double sigmaPx, int baselineFrames,
            int n, String spread,
            int trials, int failures,
            double truthLogScale,
            double biasLogScale, double sdLogScale, double seLogScale,
            double meanAnisotropy, double meanPerspective,
            double rotationMeanDeg, double rotationSdDeg,
            double meanInliers, double reprojRmsPx,
            double actualSpreadM) {
    }

    private final int width;
    private final int height;
    private final double focal;
    /** Principal point. Defaults to the image centre; the flight camera's is 26-52 px away. */
    private final double ppx;
    private final double ppy;

    public ScaleBiasMonteCarlo(int width, int height, double focal, double ppx, double ppy) {
        this.width = width;
        this.height = height;
        this.focal = focal;
        this.ppx = ppx;
        this.ppy = ppy;
    }

    public static ScaleBiasMonteCarlo flightGeometry() {
        // EXP-002 working geometry at downsampleFactor 2: 2448x2048 / 2, fx 1471.065 / 2.
        return new ScaleBiasMonteCarlo(1224, 1024, 735.53, 1224 / 2.0, 1024 / 2.0);
    }

    public static ScaleBiasMonteCarlo flightGeometryTrueprincipalPoint() {
        // The MARS-LVIG calibration's principal point, halved with the downsample: the pipeline
        // reads the Jacobian at the image centre, which is NOT here (EXP-VO-011 fact 3).
        return new ScaleBiasMonteCarlo(1224, 1024, 735.53, 1172.3577 / 2.0, 1046.3674 / 2.0);
    }

    // ------------------------------------------------------------------ scene and projection

    /** A ground point in the world frame of view 1: metres east/north on the plane, plus relief. */
    private record ScenePoint(double x, double y, double z) {
    }

    /** Relief field, re-drawn per trial so a result is about the class of terrain, not one hill. */
    private static final class ReliefField {
        private final Relief kind;
        private final double amplitude;
        private final double[] kx = new double[4], ky = new double[4], phase = new double[4];
        private final double slope;
        private final Random rand;
        private final double structureFraction = 0.15;

        ReliefField(Relief kind, double spreadM, double footprintM, Random rand) {
            this.kind = kind;
            this.rand = rand;
            // Scale each generator so that its p95-p5 equals the requested spread.
            // sum of 4 unit sinusoids -> sd = sqrt(4 * 1/2) = 1.414; p95-p5 = 3.29 sd for a
            // near-Gaussian sum, so amplitude = spread / (3.29 * 1.414).
            this.amplitude = spreadM / (3.2897 * 1.4142);
            // A slope of g across the footprint has a uniform depth distribution, p95-p5 = 0.9 g L.
            this.slope = spreadM / (0.9 * footprintM);
            for (int i = 0; i < 4; i++) {
                // Wavelengths from a quarter of the footprint to twice it: the scales a
                // planar-homography fit can and cannot absorb.
                double lambda = footprintM * (0.25 + 1.75 * rand.nextDouble());
                double dir = rand.nextDouble() * Math.PI;
                kx[i] = 2 * Math.PI * Math.cos(dir) / lambda;
                ky[i] = 2 * Math.PI * Math.sin(dir) / lambda;
                phase[i] = rand.nextDouble() * 2 * Math.PI;
            }
        }

        double at(double x, double y) {
            switch (kind) {
                case PLANAR:
                    return 0.0;
                case SLOPE:
                    return slope * x;
                case RANDOM_FIELD: {
                    double s = 0;
                    for (int i = 0; i < 4; i++) s += Math.sin(kx[i] * x + ky[i] * y + phase[i]);
                    return amplitude * s;
                }
                case STRUCTURES:
                    // Deterministic in (x, y) so the same point has the same height in both views:
                    // a hash-based sparse field of flat-topped blocks.
                    return blockHeight(x, y);
                default:
                    return 0.0;
            }
        }

        private double blockHeight(double x, double y) {
            // 12 m cells; a fixed fraction of cells carry a structure of height `spread`.
            double cell = 12.0;
            long ix = (long) Math.floor(x / cell), iy = (long) Math.floor(y / cell);
            long h = ix * 73856093L ^ iy * 19349663L ^ seedSalt;
            h ^= (h >>> 33);
            h *= 0xff51afd7ed558ccdL;
            h ^= (h >>> 33);
            double u = ((h >>> 11) & ((1L << 53) - 1)) / (double) (1L << 53);
            // Height chosen so that a `structureFraction` of the area at full height has the
            // requested p95-p5 spread: with 15 % of points raised, p95 is the raised value.
            return u < structureFraction ? amplitude * 3.2897 * 1.4142 / 1.0 : 0.0;
        }

        private long seedSalt = 0;

        ReliefField salt(long s) {
            this.seedSalt = s;
            return this;
        }
    }

    /** Project a scene point into a camera at height h with boresight tilt about the image x axis. */
    private double[] project(ScenePoint p, double camX, double camY, double camH,
                             double tiltRad, double yawRad) {
        // Camera looks down. Rotate the scene into camera coordinates: first the in-plane yaw,
        // then the tilt about the camera's x axis.
        double dx = p.x() - camX, dy = p.y() - camY;
        double c = Math.cos(yawRad), s = Math.sin(yawRad);
        double xc = c * dx + s * dy;
        double yc = -s * dx + c * dy;
        double zc = camH - p.z();               // depth along the nadir before tilt

        // Tilt about the camera x axis by `tiltRad`: (y, z) rotate.
        double ct = Math.cos(tiltRad), st = Math.sin(tiltRad);
        double y2 = ct * yc - st * zc;
        double z2 = st * yc + ct * zc;
        if (z2 <= 1e-6) return null;            // behind the camera

        return new double[]{ppx + focal * xc / z2, ppy + focal * y2 / z2};
    }

    /** Brown-Conrady radial distortion about the principal point. */
    private double[] distort(double[] p, double k1, double k2) {
        if (p == null || (k1 == 0.0 && k2 == 0.0)) return p;
        double x = (p[0] - ppx) / focal, y = (p[1] - ppy) / focal;
        double r2 = x * x + y * y;
        double f = 1.0 + k1 * r2 + k2 * r2 * r2;
        return new double[]{ppx + focal * x * f, ppy + focal * y * f};
    }

    /** Exact inverse of {@link #distort} by fixed-point iteration -- the "correct handling" arm. */
    private double[] undistort(double[] p, double k1, double k2) {
        if (p == null || (k1 == 0.0 && k2 == 0.0)) return p;
        double xd = (p[0] - ppx) / focal, yd = (p[1] - ppy) / focal;
        double x = xd, y = yd;
        for (int i = 0; i < 20; i++) {
            double r2 = x * x + y * y;
            double f = 1.0 + k1 * r2 + k2 * r2 * r2;
            x = xd / f;
            y = yd / f;
        }
        return new double[]{ppx + focal * x, ppy + focal * y};
    }

    // ------------------------------------------------------------------ one trial

    /**
     * One trial: the same scene points and the same keyframe-coordinate noise, observed from the
     * keyframe and from two consecutive later views.
     *
     * <h3>Why the keyframe baseline matters, and why a consecutive pair is not enough</h3>
     *
     * <p>BoofCV fits {@code keyToCurr} -- the transform from the KEYFRAME to the current frame --
     * and the shipped per-frame increment is the difference between two consecutive such fits:
     * {@code RigidNavigationState.observe} composes {@code F_{k-1} . F_k^-1}, in which the
     * {@code worldToKey} factor cancels and {@code keyToA . keyToB^-1} remains. Within an epoch the
     * camera walks away from its keyframe, so the correspondence field a mechanism acts on is
     * displaced along the travel direction by the whole accumulated baseline, not by one frame's
     * flow. Any effect that is <em>odd in image position</em> therefore grows with that baseline,
     * and modelling only a consecutive pair would understate it by roughly the epoch length.
     *
     * <p>{@code baselineFrames = b} places the two current views {@code b} and {@code b+1} frames
     * after the keyframe; {@code b = 0} is the plain consecutive pair.
     */
    private record TrialPair(List<AssociatedPair> a, List<AssociatedPair> b, double spreadM,
                             double truthLogScale) {
    }

    private TrialPair synthesise(Random rand, Condition cond) {
        double h = cond.heightM();
        double footprint = width * h / focal;
        ReliefField relief = new ReliefField(cond.relief(), cond.reliefSpreadM(), footprint, rand)
                .salt(rand.nextLong());

        double ang = Math.toRadians(cond.travelAngleDeg());
        double ux = Math.cos(ang), uy = Math.sin(ang);
        int b = Math.max(0, cond.baselineFrames());

        // Height, tilt and yaw advance at the same per-frame rate, so a `dHeightM` of 0.03 m really
        // is 0.03 m per frame at any baseline and the known answer stays exactly log(h_B / h_A).
        double hA = h + cond.dHeightM() * b, hB = h + cond.dHeightM() * (b + 1);
        double tiltK = Math.toRadians(cond.tiltDeg());
        double tiltA = Math.toRadians(cond.tiltDeg() + cond.dTiltDeg() * b);
        double tiltB = Math.toRadians(cond.tiltDeg() + cond.dTiltDeg() * (b + 1));
        double yawA = Math.toRadians(cond.dYawDeg() * b);
        double yawB = Math.toRadians(cond.dYawDeg() * (b + 1));
        double txA = cond.travelM() * ux * b, tyA = cond.travelM() * uy * b;
        double txB = cond.travelM() * ux * (b + 1), tyB = cond.travelM() * uy * (b + 1);

        List<AssociatedPair> pa = new ArrayList<>(cond.n());
        List<AssociatedPair> pb = new ArrayList<>(cond.n());
        double[] zs = new double[cond.n() * 2];
        int made = 0, attempts = 0;
        // Scene points are drawn uniformly on the GROUND, over a region comfortably larger than the
        // footprint, and kept if they project inside every view. Sampling on the ground rather than
        // back-projecting a uniform image sample matters once the terrain is not flat: features live
        // on the surface, and an elevated one is genuinely nearer the camera and therefore genuinely
        // over-represented in the image. Back-projection would have had to iterate to find the
        // surface, which does not converge across a discontinuity -- it silently under-sampled the
        // STRUCTURES field by a factor of five, which is how this was caught.
        double half = 0.75 * footprint;
        boolean lattice = cond.spread() == Spread.LATTICE
                || cond.spread() == Spread.LATTICE_QUADRANT;
        // A grid of nx by ny image positions, back-projected onto the mean plane. Exact for a
        // planar scene; for a relief arm the sampling is then slightly non-uniform in the image,
        // which is why the relief sweeps use the random modes.
        double uHi = cond.spread() == Spread.LATTICE_QUADRANT ? width * 0.4 : width;
        double vHi = cond.spread() == Spread.LATTICE_QUADRANT ? height * 0.4 : height;
        int nx = Math.max(2, (int) Math.round(Math.sqrt(cond.n() * uHi / vHi)));
        int ny = Math.max(2, (int) Math.round(cond.n() / (double) nx));
        int total = lattice ? nx * ny : cond.n();

        while (made < total && attempts < total * 60) {
            double gx, gy;
            if (lattice) {
                int idx = attempts;
                attempts++;
                if (idx >= nx * ny) break;
                double u = (idx % nx + 0.5) * uHi / nx;
                double v = (idx / nx + 0.5) * vHi / ny;
                gx = (u - ppx) * h / focal;
                gy = (v - ppy) * h / focal;
            } else {
                attempts++;
                gx = (2 * rand.nextDouble() - 1) * half;
                gy = (2 * rand.nextDouble() - 1) * half * height / (double) width;
            }
            ScenePoint p = new ScenePoint(gx, gy, relief.at(gx, gy));

            double[] qk = observe(p, 0, 0, h, tiltK, 0.0, cond);
            if (qk != null && cond.spread() == Spread.CLUSTERED
                    && (qk[0] > width * 0.4 || qk[1] > height * 0.4)) {
                continue;   // image-space clustering, applied to where the point actually lands
            }
            double[] qa = observe(p, txA, tyA, hA, tiltA, yawA, cond);
            double[] qb = observe(p, txB, tyB, hB, tiltB, yawB, cond);
            if (qk == null || qa == null || qb == null) continue;
            if (outside(qk) || outside(qa) || outside(qb)) continue;

            if (cond.sigmaPx() > 0) {
                // The KEYFRAME coordinate is shared by the two fits, exactly as a tracked point's
                // `p1` is: BoofCV freezes `p1` at the keyframe change and never re-detects it. Its
                // noise is therefore common to both fits, which is a real property of the pipeline
                // and not a modelling shortcut.
                qk[0] += rand.nextGaussian() * cond.sigmaPx();
                qk[1] += rand.nextGaussian() * cond.sigmaPx();
                qa[0] += rand.nextGaussian() * cond.sigmaPx();
                qa[1] += rand.nextGaussian() * cond.sigmaPx();
                qb[0] += rand.nextGaussian() * cond.sigmaPx();
                qb[1] += rand.nextGaussian() * cond.sigmaPx();
            }
            zs[made] = p.z();
            pa.add(new AssociatedPair(qk[0], qk[1], qa[0], qa[1]));
            pb.add(new AssociatedPair(qk[0], qk[1], qb[0], qb[1]));
            made++;
        }

        double spread = 0;
        if (made > 1) {
            double[] sorted = java.util.Arrays.copyOf(zs, made);
            java.util.Arrays.sort(sorted);
            spread = sorted[(int) (0.95 * (made - 1))] - sorted[(int) (0.05 * (made - 1))];
        }
        return new TrialPair(pa, pb, spread, Math.log(hB / hA));
    }

    private boolean outside(double[] q) {
        return q[0] < 0 || q[0] >= width || q[1] < 0 || q[1] >= height;
    }

    /** Project, then apply the camera lens model and the pipeline's handling of it. */
    private double[] observe(ScenePoint p, double camX, double camY, double camH,
                             double tiltRad, double yawRad, Condition cond) {
        double[] q = project(p, camX, camY, camH, tiltRad, yawRad);
        if (q == null) return null;
        q = distort(q, cond.k1(), cond.k2());
        if (cond.correctDistortion()) q = undistort(q, cond.k1(), cond.k2());
        return q;
    }

    // ------------------------------------------------------------------ the measurement

    @SuppressWarnings({"unchecked", "rawtypes"})
    public Result measure(Condition cond, int trials, long seed,
                          int ransacIterations, double thresholdSq) {
        Random rand = new Random(seed);

        ModelMatcherPost matcher;
        MotionModelSupport support;
        switch (cond.model()) {
            case AFFINE -> {
                matcher = new Ransac<>(RANSAC_SEED, ransacIterations, thresholdSq,
                        new ModelManagerAffine2D_F64(), AssociatedPair.class);
                matcher.setModel(GenerateAffine2D::new, DistanceAffine2DSq::new);
                support = MotionModelSupport.AFFINE;
            }
            case SIMILARITY -> {
                matcher = new Ransac<>(RANSAC_SEED, ransacIterations, thresholdSq,
                        new ModelManagerSim2_F64(), AssociatedPair.class);
                matcher.setModel(GenerateSimilarity2D::new, DistanceSimilarity2DSq::new);
                support = MotionModelSupport.SIMILARITY;
            }
            default -> {
                matcher = new Ransac<>(RANSAC_SEED, ransacIterations, thresholdSq,
                        new ModelManagerHomography2D_F64(), AssociatedPair.class);
                matcher.setModel(() -> new GenerateHomographyLinear(true), DistanceHomographySq::new);
                support = MotionModelSupport.HOMOGRAPHY;
            }
        }

        double sum = 0, sumSq = 0, aniso = 0, persp = 0, rot = 0, rotSq = 0, inl = 0, reproj = 0;
        double spreadSum = 0, truthSum = 0;
        int ok = 0, failures = 0;
        double cx = width / 2.0, cy = height / 2.0;
        RigidMotionDecomposition dec = new RigidMotionDecomposition();

        for (int t = 0; t < trials; t++) {
            TrialPair tp = synthesise(rand, cond);
            if (tp.a().size() < 30 || !matcher.process(tp.a())) {
                failures++;
                continue;
            }
            // A defensive copy: `getModelParameters()` returns live estimator state that the next
            // `process` call overwrites, and both fits are needed at once.
            InvertibleTransform live = (InvertibleTransform) matcher.getModelParameters();
            InvertibleTransform keyToA = live.createInstance();
            keyToA.setTo(live);
            double inliersA = matcher.getMatchSet().size();
            if (!matcher.process(tp.b())) {
                failures++;
                continue;
            }
            InvertibleTransform keyToB = (InvertibleTransform) matcher.getModelParameters();
            double reprojB = reprojection(support, keyToB, tp.b());

            // D^-1 : C_B -> C_A  =  keyToA . keyToB^-1, exactly as RigidNavigationState composes it
            // (georegression: a.concat(b, r) gives r(p) = b(a(p))).
            InvertibleTransform bInv = keyToB.invert(null);
            InvertibleTransform increment = bInv.createInstance();
            bInv.concat(keyToA, increment);

            double[] j = support.jacobian(increment, cx, cy, null);
            dec.set(j[0], j[1], j[2], j[3]);
            if (!dec.isProperRotation()) {
                failures++;
                continue;
            }
            double logScale = Math.log(dec.getUniformScale());
            sum += logScale;
            sumSq += logScale * logScale;
            aniso += dec.getAnisotropy();
            persp += support.perspectiveMagnitude(increment, width, height);
            rot += dec.getRotationRad();
            rotSq += dec.getRotationRad() * dec.getRotationRad();
            inl += 0.5 * (inliersA + matcher.getMatchSet().size());
            reproj += reprojB;
            spreadSum += tp.spreadM();
            truthSum += tp.truthLogScale();
            ok++;
        }

        double mean = ok > 0 ? sum / ok : Double.NaN;
        double sd = ok > 1 ? Math.sqrt(Math.max(0, (sumSq - ok * mean * mean) / (ok - 1))) : Double.NaN;
        double rotMean = ok > 0 ? rot / ok : Double.NaN;
        double rotSd = ok > 1 ? Math.sqrt(Math.max(0, (rotSq - ok * rotMean * rotMean) / (ok - 1)))
                              : Double.NaN;
        double truth = ok > 0 ? truthSum / ok : Double.NaN;

        return new Result(cond.name(), cond.model().name(), cond.relief().name(),
                cond.reliefSpreadM(), cond.k1(), cond.k2(), cond.correctDistortion(),
                cond.heightM(), cond.dHeightM(), cond.travelM(), cond.travelAngleDeg(),
                cond.dYawDeg(), cond.tiltDeg(), cond.dTiltDeg(), cond.sigmaPx(),
                cond.baselineFrames(), cond.n(), cond.spread().name(), trials, failures,
                truth, mean - truth, sd, ok > 0 ? sd / Math.sqrt(ok) : Double.NaN,
                ok > 0 ? aniso / ok : Double.NaN, ok > 0 ? persp / ok : Double.NaN,
                Math.toDegrees(rotMean), Math.toDegrees(rotSd),
                ok > 0 ? inl / ok : Double.NaN, ok > 0 ? reproj / ok : Double.NaN,
                ok > 0 ? spreadSum / ok : Double.NaN);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private double reprojection(MotionModelSupport support, InvertibleTransform model,
                                List<AssociatedPair> pairs) {
        double sum = 0;
        for (AssociatedPair p : pairs) {
            var q = support.apply(model, p.p1.x, p.p1.y, null);
            sum += (q.x - p.p2.x) * (q.x - p.p2.x) + (q.y - p.p2.y) * (q.y - p.p2.y);
        }
        return Math.sqrt(sum / pairs.size());
    }

    /** Corner radial displacement, in pixels, that a `k1` produces -- the legible parameterisation. */
    public double cornerDistortionPx(double k1, double k2) {
        double dx = Math.max(ppx, width - ppx), dy = Math.max(ppy, height - ppy);
        double r = Math.hypot(dx, dy) / focal;
        return Math.abs(k1 * r * r + k2 * r * r * r * r) * r * focal;
    }

    public int width() { return width; }

    public int height() { return height; }
}
