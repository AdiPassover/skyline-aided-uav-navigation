package org.boofcv.stitching.diagnostics;

import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo.Condition;
import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo.Model;
import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo.Relief;
import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo.Result;
import org.boofcv.stitching.diagnostics.ScaleBiasMonteCarlo.Spread;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code EXP-VO-011} Phases 3-5, run as one pre-declared grid. Writes a single CSV; all
 * interpretation happens in {@code evaluation/tools/exp_vo_011/}.
 *
 * <p>The grid is fixed here rather than passed in, so that "which conditions were run" is a
 * version-controlled fact rather than a command line someone has to remember. Flight geometry
 * throughout: 1224 x 1024 at f = 735.53 px, 80 m, 0.3 m ground displacement per frame (2.76 px of
 * flow -- {@code COMP-001}'s measured 2.9), 1,800 correspondences, matching the real inlier counts.
 *
 * <pre>
 *   ./gradlew.bat run -PmainClass=org.boofcv.stitching.diagnostics.ScaleBiasMonteCarloApp \
 *       --args="--out evaluations/exp-vo-011/mechanisms.csv [--trials 400]"
 * </pre>
 */
public final class ScaleBiasMonteCarloApp {

    private static final double H = 80.0;          // camera height above the mean ground plane, m
    private static final double TRAVEL = 0.30;     // ground displacement per frame, m (3 m/s @ 10 Hz)
    private static final int N = 1800;             // correspondences, matching real inlier counts
    private static final int RANSAC_ITERATIONS = 220;
    private static final double THRESHOLD_SQ = 3.0;

    /** The two depth spreads the LiDAR actually measured, plus a control and two stress cases. */
    private static final double[] RELIEF_SPREADS = {0.0, 2.0, 5.0, 11.3, 22.0, 40.0};

    /**
     * Radial distortion sweep, quoted by `k1` but chosen by corner displacement: a wide
     * machine-vision lens at this 79.5-degree HFOV typically shows 1-5 % radial distortion at the
     * corner, and these brackets that band on both sides.
     */
    private static final double[] K1 = {0.0, -0.005, -0.01, -0.02, -0.05, -0.10, +0.02, +0.05};

    private static final Model[] MODELS = {Model.HOMOGRAPHY, Model.AFFINE, Model.SIMILARITY};

    /**
     * Frames since the keyframe. The shipped increment is the difference between two consecutive
     * keyframe-to-current fits, so an odd-in-position mechanism acts on a correspondence field
     * displaced by the WHOLE epoch, not by one frame's flow. Measured on the committed run records
     * (track-count rise >= 20 % plus recenters, `EXP-VO-001`'s proxy, which is a lower bound on the
     * keyframe-change count and therefore an upper bound on epoch length): median 9-15 frames for
     * the affine arms and 13-134 for the homography arms. `BASELINE_REALISTIC` sits at the affine
     * median; the sweep shows the dependence rather than resting on one choice.
     */
    private static final int[] BASELINES = {0, 1, 5, 10, 20, 40, 80};
    private static final int BASELINE_REALISTIC = 10;

    public static void main(String[] args) throws IOException {
        Path out = Path.of("evaluations/exp-vo-011/mechanisms.csv");
        int trials = 400;
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals("--out")) out = Path.of(args[i + 1]);
            if (args[i].equals("--trials")) trials = Integer.parseInt(args[i + 1]);
        }

        ScaleBiasMonteCarlo mc = ScaleBiasMonteCarlo.flightGeometry();
        List<Condition> grid = buildGrid();
        System.err.printf("EXP-VO-011: %d conditions x %d trials%n", grid.size(), trials);

        List<Result> results = new ArrayList<>();
        long seed = 20260827L;
        int done = 0;
        for (Condition c : grid) {
            ScaleBiasMonteCarlo engine = c.name().startsWith("ppoffset")
                    ? ScaleBiasMonteCarlo.flightGeometryTrueprincipalPoint() : mc;
            results.add(engine.measure(c, trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            if (++done % 10 == 0) System.err.printf("  %d/%d%n", done, grid.size());
        }

        Files.createDirectories(out.getParent());
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out))) {
            w.println("condition,model,relief,relief_spread_m,k1,k2,correct_distortion,height_m,"
                    + "d_height_m,travel_m,travel_angle_deg,d_yaw_deg,tilt_deg,d_tilt_deg,sigma_px,"
                    + "baseline_frames,n,spread,trials,failures,truth_log_scale,bias_log_scale,sd_log_scale,"
                    + "se_log_scale,mean_anisotropy,mean_perspective,rotation_mean_deg,"
                    + "rotation_sd_deg,mean_inliers,reproj_rms_px,actual_spread_m,corner_distortion_px");
            for (Result r : results) {
                w.printf("%s,%s,%s,%.4f,%.6f,%.6f,%b,%.3f,%.4f,%.4f,%.1f,%.4f,%.3f,%.4f,%.4f,"
                                + "%d,%d,%s,%d,%d,%.10e,%.10e,%.10e,%.10e,%.6f,%.6e,%.6f,%.6f,"
                                + "%.1f,%.6f,%.4f,%.3f%n",
                        r.condition(), r.model(), r.relief(), r.reliefSpreadM(), r.k1(), r.k2(),
                        r.correctDistortion(), r.heightM(), r.dHeightM(), r.travelM(),
                        r.travelAngleDeg(), r.dYawDeg(), r.tiltDeg(), r.dTiltDeg(), r.sigmaPx(),
                        r.baselineFrames(), r.n(), r.spread(), r.trials(), r.failures(),
                        r.truthLogScale(),
                        r.biasLogScale(), r.sdLogScale(), r.seLogScale(), r.meanAnisotropy(),
                        r.meanPerspective(), r.rotationMeanDeg(), r.rotationSdDeg(),
                        r.meanInliers(), r.reprojRmsPx(), r.actualSpreadM(),
                        mc.cornerDistortionPx(r.k1(), r.k2()));
            }
        }
        System.err.println("wrote " + out.toAbsolutePath());
    }

    private static List<Condition> buildGrid() {
        List<Condition> g = new ArrayList<>();

        for (Model m : MODELS) {
            // --- Gate: the known answer. Planar, pinhole, noiseless, nadir. Bias must be ~0. ---
            g.add(cond("gate-ideal", m, Relief.PLANAR, 0, 0, false, 0, TRAVEL, 0, 0, 0, 0, 0,
                    Spread.DISTRIBUTED));
            // --- Gate: a known height change must be recovered exactly. ---
            g.add(cond("gate-climb", m, Relief.PLANAR, 0, 0, false, 0.03, TRAVEL, 0, 0, 0, 0, 0,
                    Spread.DISTRIBUTED));

            // --- Phase 3: lens distortion, uncorrected then corrected, across four travel
            //     directions so that the odd-symmetry prediction is tested rather than assumed. ---
            for (double k1 : K1) {
                for (double ang : new double[]{0, 90, 180, 270}) {
                    g.add(cond("dist", m, Relief.PLANAR, 0, k1, false, 0, TRAVEL, ang, 0, 0, 0, 0,
                            Spread.DISTRIBUTED));
                }
                g.add(cond("dist-corrected", m, Relief.PLANAR, 0, k1, true, 0, TRAVEL, 0, 0, 0, 0, 0,
                        Spread.DISTRIBUTED));
                // Asymmetric feature support, where an odd mechanism is at its largest.
                g.add(cond("dist-clustered", m, Relief.PLANAR, 0, k1, false, 0, TRAVEL, 0, 0, 0, 0, 0,
                        Spread.CLUSTERED));
                // The same sweep on a DETERMINISTIC lattice, so the geometric answer comes back
                // with no sampling error at all: symmetric support (centroid exactly at the image
                // centre) and the most asymmetric support worth considering (one quadrant).
                for (double ang : new double[]{0, 90, 180, 270}) {
                    g.add(cond("dist-lattice", m, Relief.PLANAR, 0, k1, false, 0, TRAVEL, ang, 0, 0,
                            0, 0, Spread.LATTICE));
                    g.add(cond("dist-lattice-quadrant", m, Relief.PLANAR, 0, k1, false, 0, TRAVEL,
                            ang, 0, 0, 0, 0, Spread.LATTICE_QUADRANT));
                }
            }
            // Distortion under the other motion types, at the middle of the realistic band.
            g.add(cond("dist-yaw", m, Relief.PLANAR, 0, -0.02, false, 0, 0, 0, 0.2, 0, 0, 0,
                    Spread.DISTRIBUTED));
            g.add(cond("dist-transyaw", m, Relief.PLANAR, 0, -0.02, false, 0, TRAVEL, 0, 0.2, 0, 0, 0,
                    Spread.DISTRIBUTED));
            g.add(cond("dist-climb", m, Relief.PLANAR, 0, -0.02, false, 0.03, TRAVEL, 0, 0, 0, 0, 0,
                    Spread.DISTRIBUTED));

            // --- Phase 4: terrain relief, three geometries, spreads bracketing the LiDAR values. ---
            for (Relief r : new Relief[]{Relief.SLOPE, Relief.RANDOM_FIELD, Relief.STRUCTURES}) {
                for (double spread : RELIEF_SPREADS) {
                    for (double ang : new double[]{0, 90}) {
                        g.add(cond("relief", m, r, spread, 0, false, 0, TRAVEL, ang, 0, 0, 0, 0,
                                Spread.DISTRIBUTED));
                    }
                }
            }
            // Relief with a turn, and relief with a climb: does motion type change the answer?
            g.add(cond("relief-yaw", m, Relief.RANDOM_FIELD, 22.0, 0, false, 0, TRAVEL, 0, 0.2, 0, 0,
                    0, Spread.DISTRIBUTED));
            g.add(cond("relief-climb", m, Relief.RANDOM_FIELD, 22.0, 0, false, 0.03, TRAVEL, 0, 0, 0,
                    0, 0, Spread.DISTRIBUTED));

            // --- Phase 5: the pre-declared 2x2, at realistic levels only. No factorial sweep. ---
            for (double spread : new double[]{11.3, 22.0}) {
                g.add(cond("both", m, Relief.RANDOM_FIELD, spread, -0.02, false, 0, TRAVEL, 0, 0, 0,
                        0, 0, Spread.DISTRIBUTED));
            }

            // --- H6b: errors-in-variables attenuation. Planar, pinhole, noise only. ---
            for (double sigma : new double[]{0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0}) {
                g.add(cond("eiv", m, Relief.PLANAR, 0, 0, false, 0, TRAVEL, 0, 0, 0, 0, sigma,
                        Spread.DISTRIBUTED));
                g.add(cond("eiv-clustered", m, Relief.PLANAR, 0, 0, false, 0, TRAVEL, 0, 0, 0, 0,
                        sigma, Spread.CLUSTERED));
            }

            // --- H5b/H6a: tilt, held and changing, at the flights' measured magnitudes. ---
            for (double tilt : new double[]{0, 1.5, 6.5, 15.5}) {
                g.add(cond("tilt", m, Relief.PLANAR, 0, 0, false, 0, TRAVEL, 0, 0, tilt, 0, 0,
                        Spread.DISTRIBUTED));
                g.add(cond("tilt-changing", m, Relief.PLANAR, 0, 0, false, 0, TRAVEL, 0, 0, tilt, 0.1,
                        0, Spread.DISTRIBUTED));
            }

            // --- H6c: the principal point is not the image centre, and the Jacobian is read at the
            //     centre. Same conditions as the tilt arm, offset optics. ---
            for (double tilt : new double[]{0, 6.5, 15.5}) {
                g.add(cond("ppoffset-tilt", m, Relief.PLANAR, 0, 0, false, 0, TRAVEL, 0, 0, tilt, 0,
                        0, Spread.DISTRIBUTED));
            }
            g.add(cond("ppoffset-dist", m, Relief.PLANAR, 0, -0.02, false, 0, TRAVEL, 0, 0, 0, 0, 0,
                    Spread.DISTRIBUTED));

            // --- The baseline sweep: how each mechanism grows with distance from the keyframe.
            //     This is the axis a consecutive-pair model cannot see at all. ---
            for (int b : BASELINES) {
                g.add(cond("baseline-dist", m, Relief.PLANAR, 0, -0.02, false, 0, TRAVEL, 0, 0, 0, 0,
                        0, Spread.DISTRIBUTED, b));
                g.add(cond("baseline-relief", m, Relief.RANDOM_FIELD, 22.0, 0, false, 0, TRAVEL, 0, 0,
                        0, 0, 0, Spread.DISTRIBUTED, b));
                g.add(cond("baseline-eiv", m, Relief.PLANAR, 0, 0, false, 0, TRAVEL, 0, 0, 0, 0, 0.5,
                        Spread.DISTRIBUTED, b));
                g.add(cond("baseline-ideal", m, Relief.PLANAR, 0, 0, false, 0, TRAVEL, 0, 0, 0, 0, 0,
                        Spread.DISTRIBUTED, b));
                g.add(cond("baseline-dist-lattice-quadrant", m, Relief.PLANAR, 0, -0.02, false, 0,
                        TRAVEL, 0, 0, 0, 0, 0, Spread.LATTICE_QUADRANT, b));
            }

            // --- The realistic composite: everything at once, at each flight's measured values. ---
            g.add(cond("realistic-hkairport", m, Relief.RANDOM_FIELD, 22.0, -0.02, false, 0, TRAVEL,
                    0, 0.02, 6.5, 0.1, 0.3, Spread.DISTRIBUTED));
            g.add(cond("realistic-amtown", m, Relief.RANDOM_FIELD, 11.3, -0.02, false, 0, 0.40,
                    0, 0.01, 1.5, 0.05, 0.3, Spread.DISTRIBUTED));
        }
        return g;
    }

    private static Condition cond(String name, Model m, Relief r, double spread, double k1,
                                  boolean correct, double dH, double travel, double ang,
                                  double dYaw, double tilt, double dTilt, double sigma,
                                  Spread sp) {
        return cond(name, m, r, spread, k1, correct, dH, travel, ang, dYaw, tilt, dTilt, sigma, sp,
                BASELINE_REALISTIC);
    }

    private static Condition cond(String name, Model m, Relief r, double spread, double k1,
                                  boolean correct, double dH, double travel, double ang,
                                  double dYaw, double tilt, double dTilt, double sigma,
                                  Spread sp, int baseline) {
        return new Condition(name, m, r, spread, k1, 0.0, correct, H, dH, travel, ang, dYaw,
                tilt, dTilt, sigma, baseline, N, sp);
    }
}
