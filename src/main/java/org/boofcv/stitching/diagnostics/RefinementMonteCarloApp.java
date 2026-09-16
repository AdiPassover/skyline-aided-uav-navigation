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
 * Runs {@link RefinementMonteCarlo}'s pre-registered conditions and writes the result as JSON.
 *
 * <p>{@code EXP-VO-010} Phase 3. Tier 1 — synthetic correspondences only. The condition list is
 * fixed here rather than passed on the command line so the reported numbers have one
 * version-controlled definition.
 *
 * <pre>
 *   ./gradlew.bat run -PmainClass=org.boofcv.stitching.diagnostics.RefinementMonteCarloApp \
 *       --args="--out evaluations/exp-vo-010/monte_carlo.json"
 * </pre>
 */
public final class RefinementMonteCarloApp {

    /** The working resolution of every flight arm (`downsampleFactor = 2` on 2448 × 2048). */
    private static final int WIDTH = 1224;
    private static final int HEIGHT = 1024;

    /** Per-frame motion of the order the flights actually show (`EXP-VO-004`, `EXP-VO-008`). */
    private static final double THETA_DEG = 0.05;
    private static final double SCALE = 1.0003;
    private static final double TX = 2.9;
    private static final double TY = -1.4;

    private static final int TRIALS = 400;
    private static final int RANSAC_ITERATIONS = 220;
    private static final double INLIER_THRESHOLD_SQ = 3.0;

    public static void main(String[] args) throws IOException {
        String out = arg(args, "--out");
        int trials = arg(args, "--trials") != null
                ? Integer.parseInt(arg(args, "--trials")) : TRIALS;

        RefinementMonteCarlo mc = new RefinementMonteCarlo(WIDTH, HEIGHT);
        List<RefinementMonteCarlo.Condition> conditions = new ArrayList<>();

        for (RefinementMonteCarlo.Model model : RefinementMonteCarlo.Model.values()) {
            var D = RefinementMonteCarlo.Spread.DISTRIBUTED;
            var C = RefinementMonteCarlo.Spread.CLUSTERED;
            // 1-2: exact, no noise -- refinement must not regress an already-perfect fit.
            conditions.add(new RefinementMonteCarlo.Condition("exact", model, 2000, 0.0, 0.0, D, 0));
            // 3-4: small and moderate Gaussian correspondence noise.
            conditions.add(new RefinementMonteCarlo.Condition("noise-0.3", model, 2000, 0.3, 0.0, D, 0));
            conditions.add(new RefinementMonteCarlo.Condition("noise-1.0", model, 2000, 1.0, 0.0, D, 0));
            // 5: outlier contamination, at the primary noise level.
            conditions.add(new RefinementMonteCarlo.Condition("outliers-5pct", model, 2000, 0.3, 0.05, D, 0));
            conditions.add(new RefinementMonteCarlo.Condition("outliers-20pct", model, 2000, 0.3, 0.20, D, 0));
            // 6-7: spatial distribution -- the kappa axis LIT-VO-005 s4 identified.
            conditions.add(new RefinementMonteCarlo.Condition("clustered", model, 2000, 0.3, 0.0, C, 0));
            conditions.add(new RefinementMonteCarlo.Condition("clustered-noise-1.0", model, 2000, 1.0, 0.0, C, 0));
            // 8: realistic feature counts. Homography's median on AMtown01 is 671, affine's 2060.
            conditions.add(new RefinementMonteCarlo.Condition("n-671", model, 671, 0.3, 0.0, D, 0));
            conditions.add(new RefinementMonteCarlo.Condition("n-120", model, 120, 0.3, 0.0, D, 0));
            conditions.add(new RefinementMonteCarlo.Condition("n-30", model, 30, 0.3, 0.0, D, 0));
            // A projective truth, so the affine arm is genuinely misspecified and the homography arm
            // is exactly specified -- the asymmetry Phase 10 asks about.
            conditions.add(new RefinementMonteCarlo.Condition("perspective-0.1", model, 2000, 0.3, 0.0, D, 0.1));
        }

        List<Map<String, Object>> rows = new ArrayList<>();
        for (RefinementMonteCarlo.Condition c : conditions) {
            rows.add(row(mc.measure(c, trials, 90210L, Math.toRadians(THETA_DEG), SCALE, TX, TY,
                    RANSAC_ITERATIONS, INLIER_THRESHOLD_SQ)));
        }

        String json = new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(rows);
        if (out != null) {
            Path p = Paths.get(out);
            if (p.getParent() != null) Files.createDirectories(p.getParent());
            Files.writeString(p, json);
            System.out.println("wrote " + rows.size() + " rows to " + p);
        } else {
            System.out.println(json);
        }
    }

    private static Map<String, Object> row(RefinementMonteCarlo.Result r) {
        System.err.printf("  %-20s %-11s n=%-5d sig=%.1f out=%.2f %-11s persp=%.1f | "
                        + "rot rms %8.5f -> %8.5f | trans rms %7.4f -> %7.4f | "
                        + "reproj %7.4f -> %7.4f | inliers %.0f%n",
                r.condition(), r.model(), r.n(), r.sigmaPx(), r.outlierFraction(), r.spread(),
                r.perspective(), r.minRotRmsDeg(), r.refRotRmsDeg(),
                r.minTransRmsPx(), r.refTransRmsPx(),
                r.minReprojRmsPx(), r.refReprojRmsPx(), r.minimalInliers());
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("condition", r.condition());
        m.put("model", r.model());
        m.put("n_points", r.n());
        m.put("sigma_px", r.sigmaPx());
        m.put("outlier_fraction", r.outlierFraction());
        m.put("spread", r.spread());
        m.put("perspective_px", r.perspective());
        m.put("trials", r.trials());
        m.put("failures", r.failures());
        m.put("mean_inliers", r.minimalInliers());
        m.put("minimal_rot_bias_deg", r.minRotBiasDeg());
        m.put("minimal_rot_sd_deg", r.minRotSdDeg());
        m.put("minimal_rot_rms_deg", r.minRotRmsDeg());
        m.put("refined_rot_bias_deg", r.refRotBiasDeg());
        m.put("refined_rot_sd_deg", r.refRotSdDeg());
        m.put("refined_rot_rms_deg", r.refRotRmsDeg());
        m.put("minimal_trans_rms_px", r.minTransRmsPx());
        m.put("refined_trans_rms_px", r.refTransRmsPx());
        m.put("minimal_scale_rms_rel", r.minScaleRmsRel());
        m.put("refined_scale_rms_rel", r.refScaleRmsRel());
        m.put("minimal_reproj_rms_px", r.minReprojRmsPx());
        m.put("refined_reproj_rms_px", r.refReprojRmsPx());
        return m;
    }

    private static String arg(String[] args, String flag) {
        for (int i = 0; i < args.length - 1; i++) {
            if (flag.equals(args[i])) return args[i + 1];
        }
        return null;
    }
}
