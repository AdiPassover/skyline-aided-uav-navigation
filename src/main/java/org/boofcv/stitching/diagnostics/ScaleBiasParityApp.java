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
 * {@code EXP-VO-011}: which candidate mechanisms are <b>odd</b> under time reversal, and which are
 * <b>even</b>?
 *
 * <h2>Why this is the decisive question</h2>
 *
 * <p>Phase 6b replays real flight frames backwards and measures the observed bias to be
 * <b>77-132 % odd</b> in the direction of travel, with an even component of at most
 * {@code 1.1e-4} that does not even agree in sign across arms. Time reversal negates two things at
 * once: the travel step {@code t} and the per-frame attitude rate {@code dTilt}. A mechanism can
 * therefore only be the dominant cause of the real drift if it is <b>odd</b> in those quantities --
 * whatever its magnitude. This app measures the parity of each candidate directly, by running it at
 * {@code +x} and {@code -x} and comparing, rather than arguing it from symmetry.
 *
 * <p>Deliberately a separate, small driver rather than more rows in
 * {@link ScaleBiasMonteCarloApp}'s grid: the parity question came out of that grid's results, and
 * re-running 660 conditions to add eight would make the main grid's provenance harder to follow, not
 * easier.
 *
 * <pre>
 *   ./gradlew.bat run -PmainClass=org.boofcv.stitching.diagnostics.ScaleBiasParityApp \
 *       --args="--out evaluations/exp-vo-011/parity.csv [--trials 400]"
 * </pre>
 */
public final class ScaleBiasParityApp {

    private static final double H = 80.0;
    private static final double TRAVEL = 0.30;
    private static final int N = 1800;
    private static final int BASELINE = 10;
    private static final int RANSAC_ITERATIONS = 220;
    private static final double THRESHOLD_SQ = 3.0;

    public static void main(String[] args) throws IOException {
        Path out = Path.of("evaluations/exp-vo-011/parity.csv");
        int trials = 400;
        // The two mechanisms whose per-frame bias is NOT resolvable above the Monte Carlo error at
        // the grid's trial count get their own, much larger, budget. Their one-sided bound is what
        // the scale-bias budget has to quote for them, and a bound set by sampling error rather than
        // by physics is not worth quoting loosely: 20x the trials tightens it by 4.5x.
        int precisionTrials = 8000;
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals("--out")) out = Path.of(args[i + 1]);
            if (args[i].equals("--trials")) trials = Integer.parseInt(args[i + 1]);
            if (args[i].equals("--precision-trials")) precisionTrials = Integer.parseInt(args[i + 1]);
        }

        ScaleBiasMonteCarlo mc = ScaleBiasMonteCarlo.flightGeometry();
        List<String[]> pairs = new ArrayList<>();
        List<Result> results = new ArrayList<>();
        long seed = 20260827_2L;

        for (Model m : Model.values()) {
            // 1. A coherent ground slope, travelled up and travelled down. LIT-VO-003 section 4b
            //    predicts a strictly odd effect: the sign follows (grad z . t).
            results.add(mc.measure(cond("slope+", m, Relief.SLOPE, 22.0, 0, 0, 0, TRAVEL, 0, 0, 0,
                    Spread.DISTRIBUTED), trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            results.add(mc.measure(cond("slope-", m, Relief.SLOPE, 22.0, 0, 0, 0, TRAVEL, 180, 0, 0,
                    Spread.DISTRIBUTED), trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            pairs.add(new String[]{"slope", m.name()});

            // 2. Lens distortion on an asymmetric support, travelled both ways. Predicted odd, since
            //    the apparent scale goes as (feature centroid . travel).
            results.add(mc.measure(cond("dist+", m, Relief.PLANAR, 0, -0.02, 0, 0, TRAVEL, 0, 0, 0,
                    Spread.LATTICE_QUADRANT), trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            results.add(mc.measure(cond("dist-", m, Relief.PLANAR, 0, -0.02, 0, 0, TRAVEL, 180, 0, 0,
                    Spread.LATTICE_QUADRANT), trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            pairs.add(new String[]{"dist", m.name()});

            // 3. A CHANGING boresight tilt, at the flights' measured median rate of about
            //    0.1 deg/frame, nosing over and pulling up. EXP-VO-005 R6 found an oscillating roll
            //    produces a large NON-ZERO mean drift in the affine model, which is only possible if
            //    this is even -- so this is the arm that tests that inference directly.
            results.add(mc.measure(cond("dtilt+", m, Relief.PLANAR, 0, 0, 0, 0, TRAVEL, 0, 6.5,
                    +0.1, Spread.DISTRIBUTED), trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            results.add(mc.measure(cond("dtilt-", m, Relief.PLANAR, 0, 0, 0, 0, TRAVEL, 0, 6.5,
                    -0.1, Spread.DISTRIBUTED), trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            pairs.add(new String[]{"dtilt", m.name()});

            // 4. Feature-localisation noise at a realistic sub-pixel level. Predicted even: an
            //    attenuation does not know which way the camera is going.
            results.add(mc.measure(cond("noise+", m, Relief.PLANAR, 0, 0, 0, 0, TRAVEL, 0, 0, 0,
                    Spread.DISTRIBUTED, 0.5), trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            results.add(mc.measure(cond("noise-", m, Relief.PLANAR, 0, 0, 0, 0, TRAVEL, 180, 0, 0,
                    Spread.DISTRIBUTED, 0.5), trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            pairs.add(new String[]{"noise", m.name()});

            // 5. Homogeneous relief at HKairport01's measured depth spread, both ways.
            results.add(mc.measure(cond("relief+", m, Relief.RANDOM_FIELD, 22.0, 0, 0, 0, TRAVEL, 0,
                    0, 0, Spread.DISTRIBUTED), trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            results.add(mc.measure(cond("relief-", m, Relief.RANDOM_FIELD, 22.0, 0, 0, 0, TRAVEL,
                    180, 0, 0, Spread.DISTRIBUTED), trials, seed++, RANSAC_ITERATIONS, THRESHOLD_SQ));
            pairs.add(new String[]{"relief", m.name()});

            // 6. The same two, at the precision budget: these are the arms the budget can only
            //    bound, so the bound is worth measuring properly.
            results.add(mc.measure(cond("reliefhi+", m, Relief.RANDOM_FIELD, 22.0, 0, 0, 0, TRAVEL,
                    0, 0, 0, Spread.DISTRIBUTED), precisionTrials, seed++, RANSAC_ITERATIONS,
                    THRESHOLD_SQ));
            results.add(mc.measure(cond("reliefhi-", m, Relief.RANDOM_FIELD, 22.0, 0, 0, 0, TRAVEL,
                    180, 0, 0, Spread.DISTRIBUTED), precisionTrials, seed++, RANSAC_ITERATIONS,
                    THRESHOLD_SQ));
            pairs.add(new String[]{"reliefhi", m.name()});
            results.add(mc.measure(cond("noisehi+", m, Relief.PLANAR, 0, 0, 0, 0, TRAVEL, 0, 0, 0,
                    Spread.DISTRIBUTED, 0.5), precisionTrials, seed++, RANSAC_ITERATIONS,
                    THRESHOLD_SQ));
            results.add(mc.measure(cond("noisehi-", m, Relief.PLANAR, 0, 0, 0, 0, TRAVEL, 180, 0, 0,
                    Spread.DISTRIBUTED, 0.5), precisionTrials, seed++, RANSAC_ITERATIONS,
                    THRESHOLD_SQ));
            pairs.add(new String[]{"noisehi", m.name()});
        }

        Files.createDirectories(out.getParent());
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out))) {
            w.println("condition,model,bias_log_scale,se_log_scale,sd_log_scale,trials,failures");
            for (Result r : results) {
                w.printf("%s,%s,%.10e,%.10e,%.10e,%d,%d%n", r.condition(), r.model(),
                        r.biasLogScale(), r.seLogScale(), r.sdLogScale(), r.trials(), r.failures());
            }
        }

        System.err.printf("%-10s %-12s %12s %12s %12s %12s %9s%n",
                "mechanism", "model", "forward", "reversed", "odd", "even", "verdict");
        for (String[] p : pairs) {
            Result f = find(results, p[0] + "+", p[1]);
            Result b = find(results, p[0] + "-", p[1]);
            double odd = (f.biasLogScale() - b.biasLogScale()) / 2;
            double even = (f.biasLogScale() + b.biasLogScale()) / 2;
            String verdict = Math.abs(odd) > 2 * Math.abs(even) ? "ODD"
                    : Math.abs(even) > 2 * Math.abs(odd) ? "EVEN" : "mixed/noise";
            System.err.printf("%-10s %-12s %12.3e %12.3e %12.3e %12.3e %9s%n",
                    p[0], p[1], f.biasLogScale(), b.biasLogScale(), odd, even, verdict);
        }
        System.err.println("wrote " + out.toAbsolutePath());
    }

    private static Result find(List<Result> rs, String cond, String model) {
        for (Result r : rs) {
            if (r.condition().equals(cond) && r.model().equals(model)) return r;
        }
        throw new IllegalStateException(cond + " " + model);
    }

    private static Condition cond(String name, Model m, Relief r, double spread, double k1,
                                  double k2, double dH, double travel, double ang, double tilt,
                                  double dTilt, Spread sp) {
        return cond(name, m, r, spread, k1, k2, dH, travel, ang, tilt, dTilt, sp, 0.0);
    }

    private static Condition cond(String name, Model m, Relief r, double spread, double k1,
                                  double k2, double dH, double travel, double ang, double tilt,
                                  double dTilt, Spread sp, double sigma) {
        return new Condition(name, m, r, spread, k1, k2, false, H, dH, travel, ang, 0.0,
                tilt, dTilt, sigma, BASELINE, N, sp);
    }
}
