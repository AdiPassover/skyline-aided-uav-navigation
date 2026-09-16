package org.boofcv.stitching.diagnostics;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs {@link ModelRotationMonteCarlo}'s pre-registered sweep and writes the result as JSON.
 *
 * <p>{@code EXP-VO-009} Phase 5. Tier 1 — synthetic correspondences only, no imagery, no flight
 * data. The sweep is fixed here rather than passed on the command line so the reported numbers have
 * one version-controlled definition.
 *
 * <pre>
 *   ./gradlew.bat run -PmainClass=org.boofcv.stitching.diagnostics.ModelRotationMonteCarloApp \
 *       --args="--out evaluations/exp-vo-009/monte_carlo.json"
 * </pre>
 */
public final class ModelRotationMonteCarloApp {

    /**
     * The working resolution of every flight arm ({@code downsampleFactor = 2} on MARS-LVIG's
     * 2448 × 2048), so the synthetic geometry matches the real one.
     */
    private static final int WIDTH = 1224;
    private static final int HEIGHT = 1024;

    /**
     * A per-frame rotation of 0.05° and scale of 1.0003 — the order of magnitude the flights
     * actually show ({@code EXP-VO-004}: ≈ 3 × 10⁻⁴ log-scale per frame; {@code EXP-VO-008}:
     * per-frame rotation increments well under a degree).
     */
    private static final double THETA_DEG = 0.05;
    private static final double SCALE = 1.0003;

    private static final int TRIALS = 1000;
    private static final int RANSAC_ITERATIONS = 220;
    private static final double INLIER_THRESHOLD_SQ = 3.0;

    public static void main(String[] args) throws IOException {
        String out = arg(args, "--out");
        int trials = args.length > 0 && arg(args, "--trials") != null
                ? Integer.parseInt(arg(args, "--trials")) : TRIALS;

        ModelRotationMonteCarlo mc = new ModelRotationMonteCarlo(WIDTH, HEIGHT);
        List<Map<String, Object>> rows = new ArrayList<>();

        // Track counts spanning the two flights' measured medians: homography 591-2064,
        // affine 2094-2226 (weekly report section 5). 2000 is the primary operating point.
        int[] counts = {2000, 500, 120};
        // KLT correspondence noise. EXP-VO-001 measured RANSAC residual rms 0.72-1.03 px on real
        // imagery, which bounds sigma from above; 0.3 is the primary, 0.6 and 1.0 the sensitivity.
        double[] sigmas = {0.3, 0.6, 1.0};

        for (ModelRotationMonteCarlo.Arm arm : ModelRotationMonteCarlo.defaultArms()) {
            for (ModelRotationMonteCarlo.FitMode mode : ModelRotationMonteCarlo.FitMode.values()) {
                // ALL has no RANSAC, so the 2-point control is meaningless there (it differs only
                // in the sample size RANSAC draws).
                if (mode == ModelRotationMonteCarlo.FitMode.ALL
                        && arm.name().equals("similarity-2pt")) {
                    continue;
                }
                for (int n : counts) {
                    for (double sigma : sigmas) {
                        if (n != 2000 && sigma != 0.3) continue;      // one axis at a time
                        rows.add(row(mc.measure(arm, mode, trials, 90210L, n, sigma,
                                0.0, 1.0, 0.0, SCALE, Math.toRadians(THETA_DEG),
                                RANSAC_ITERATIONS, INLIER_THRESHOLD_SQ)));
                    }
                }
                // Outlier contamination, at the primary operating point.
                for (double f : new double[]{0.05, 0.20}) {
                    rows.add(row(mc.measure(arm, mode, trials, 90210L, 2000, 0.3, f, 1.0, 0.0,
                            SCALE, Math.toRadians(THETA_DEG), RANSAC_ITERATIONS,
                            INLIER_THRESHOLD_SQ)));
                }
                // Model violation: the similarity is intentionally imperfect here. Anisotropy
                // values bracket the measured per-frame p99 (1.013-1.025, EXP-VO-006); the
                // perspective magnitudes bracket its p99 (0.053-0.095 px).
                for (double aniso : new double[]{1.004, 1.02, 1.04}) {
                    rows.add(row(mc.measure(arm, mode, trials, 90210L, 2000, 0.3, 0.0, aniso, 0.0,
                            SCALE, Math.toRadians(THETA_DEG), RANSAC_ITERATIONS,
                            INLIER_THRESHOLD_SQ)));
                }
                for (double persp : new double[]{0.1, 1.0}) {
                    rows.add(row(mc.measure(arm, mode, trials, 90210L, 2000, 0.3, 0.0, 1.0, persp,
                            SCALE, Math.toRadians(THETA_DEG), RANSAC_ITERATIONS,
                            INLIER_THRESHOLD_SQ)));
                }
            }
        }

        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(rows);
        if (out != null) {
            Path p = Paths.get(out);
            if (p.getParent() != null) Files.createDirectories(p.getParent());
            Files.writeString(p, json);
            System.out.println("wrote " + rows.size() + " rows to " + p);
        } else {
            System.out.println(json);
        }
    }

    /**
     * <b>The sweep is a paired design and that is what makes it sensitive.</b> Every
     * {@code measure} call reseeds from the same constant, so for one configuration all four arms
     * see <em>byte-identical</em> correspondence sets trial by trial — the arms differ only in the
     * model fitted to them. A few thousand trials therefore resolve a difference that an unpaired
     * design would need far more of.
     */
    private static Map<String, Object> row(ModelRotationMonteCarlo.Result r) {
        System.err.printf("  %-16s %-8s n=%-5d sigma=%.1f out=%.2f aniso=%.3f persp=%.1f  "
                        + "sd=%.5f deg  mean=%+.5f  inliers=%.0f%n",
                r.arm(), r.fitMode(), r.nPoints(), r.sigmaPx(), r.outlierFraction(),
                r.anisotropy(), r.perspective(), r.sdRotationErrorDeg(),
                r.meanRotationErrorDeg(), r.meanInliers());
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("arm", r.arm());
        m.put("fit_mode", r.fitMode());
        m.put("n_points", r.nPoints());
        m.put("sigma_px", r.sigmaPx());
        m.put("outlier_fraction", r.outlierFraction());
        m.put("anisotropy", r.anisotropy());
        m.put("perspective_px", r.perspective());
        m.put("trials", r.trials());
        m.put("failures", r.failures());
        m.put("mean_rotation_error_deg", r.meanRotationErrorDeg());
        m.put("sd_rotation_error_deg", r.sdRotationErrorDeg());
        m.put("rms_rotation_error_deg", r.rmsRotationErrorDeg());
        m.put("p95_abs_rotation_error_deg", r.p95AbsRotationErrorDeg());
        m.put("mean_scale_error", r.meanScaleError());
        m.put("mean_inliers", r.meanInliers());
        m.put("mean_kappa", r.meanKappa());
        return m;
    }

    private static String arg(String[] args, String flag) {
        for (int i = 0; i < args.length - 1; i++) {
            if (flag.equals(args[i])) return args[i + 1];
        }
        return null;
    }
}
