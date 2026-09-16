package org.boofcv.stitching.metric;

import georegression.struct.homography.Homography2D_F64;
import org.boofcv.stitching.MotionModelSupport;
import org.boofcv.stitching.RigidNavigationState;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * <b>Python-to-Java parity for the metric readout</b> — the evidence that the runtime port
 * reproduces the already-validated offline implementation rather than merely resembling it.
 *
 * <p>Six cases, frozen by {@code evaluation/tools/vo_metric_runtime/make_parity_reference.py} from
 * {@code evaluation/tools/exp_vo_012/metric_readout.py} and {@code baro.py} — the code
 * {@code EXP-VO-012}, {@code EXP-VO-013} and {@code EXP-VO-014} actually ran. Each case supplies
 * one materialised height stream ({@code samples.csv}) and one expected trajectory
 * ({@code expected.csv}), and this test drives the <b>production</b> classes over exactly those
 * inputs:
 *
 * <pre>
 *   committed logical_transform.csv  -&gt;  RigidNavigationState   (the real increment computation)
 *   committed samples.csv            -&gt;  CausalHeightChannel    (the real causal ZOH)
 *                                    -&gt;  MetricNavigationState  (the real DEC-VO-007 D3 integrator)
 * </pre>
 *
 * <p><b>Nothing is re-derived in test code.</b> The increments come from the same committed
 * transforms the Python side reads, and they go through {@code RigidNavigationState} itself, so a
 * disagreement can only be a convention difference in the conversion — which is the thing worth
 * detecting. Four quantities are compared per frame: metric east, metric north, yaw, and the height
 * sample actually used, plus the height's validity and staleness state.
 *
 * <p>The cases cover an ideal per-frame channel, the fixed-height null, a 10x-slower channel, the
 * same channel with the freshness boundary declared differently, two dropouts that cross
 * {@code tau_stale}, and a real flight containing a stitching restart.
 */
class MetricReadoutPythonParityTest {

    private static final Path REPO = Paths.get("").toAbsolutePath();
    private static final Path REFERENCE =
            REPO.resolve("evaluation/tools/vo_metric_runtime/reference");

    /**
     * Absolute tolerance in metres on the integrated position, and in metres on the height.
     *
     * <p><b>Justified, not chosen for convenience.</b> Both sides consume the same committed
     * transforms and the same committed height samples, so the only difference available to them is
     * the order of IEEE-754 operations between NumPy and the JVM. Over the longest case — 2 902
     * frames and roughly 350 m of integrated metric path — accumulated double rounding is of order
     * {@code n * eps * |T|}, i.e. about {@code 3000 * 2.2e-16 * 350 = 2e-10} m. This threshold is
     * two orders above that and eight orders below any physically meaningful displacement, so it
     * cannot mask a convention error: getting the reference frame wrong costs a factor of
     * {@code h_k/h_{k-1}} per frame, and forgetting the downsample divisor costs a factor of two.
     */
    private static final double POS_TOL_M = 1e-8;

    /** Yaw is a plain sum of the same per-frame angles on both sides; degrees. */
    private static final double YAW_TOL_DEG = 1e-9;

    @TestFactory
    @DisplayName("G: the Java runtime metric track equals the Python MetricTrack, frame by frame")
    List<DynamicTest> parityAcrossFrozenCases() throws IOException {
        Path casesJson = REFERENCE.resolve("cases.json");
        assumeTrue(Files.exists(casesJson),
                "parity reference absent from this checkout: " + casesJson);

        List<Map<String, String>> cases = readCases(casesJson);
        assertTrue(cases.size() >= 5, "expected the frozen case set, got " + cases.size());

        List<DynamicTest> tests = new ArrayList<>();
        for (Map<String, String> c : cases) {
            tests.add(DynamicTest.dynamicTest(c.get("name"), () -> runCase(c)));
        }
        return tests;
    }

    private void runCase(Map<String, String> c) throws IOException {
        String name = c.get("name");
        Path runDir = REPO.resolve("runs").resolve(c.get("run_id"));
        Path caseDir = REFERENCE.resolve(name);
        assumeTrue(Files.exists(runDir.resolve("logical_transform.csv")),
                "committed run record absent: " + runDir);

        int width = Integer.parseInt(c.get("frame_width"));
        int height = Integer.parseInt(c.get("frame_height"));
        double fWorking = Double.parseDouble(c.get("f_working_px"));
        double h0 = Double.parseDouble(c.get("h0_agl_m"));
        double tauStale = Double.parseDouble(c.get("tau_stale_s"));
        double interval = Double.parseDouble(c.get("nominal_sample_interval_s"));

        List<String[]> transforms = readCsv(runDir.resolve("logical_transform.csv"));
        Map<String, Integer> tCols = headerIndex(transforms.remove(0));
        List<String[]> expected = readCsv(caseDir.resolve("expected.csv"));
        Map<String, Integer> eCols = headerIndex(expected.remove(0));
        List<String[]> samples = readCsv(caseDir.resolve("samples.csv"));
        samples.remove(0);

        assertEquals(expected.size(), transforms.size(),
                name + ": expected rows and transform rows must line up");

        // --- the production objects, exactly as the estimator builds them ------------------------
        RigidNavigationState<Homography2D_F64> rigid =
                new RigidNavigationState<>(MotionModelSupport.HOMOGRAPHY, new Homography2D_F64());
        MetricNavigationState metric = new MetricNavigationState(fWorking);
        CausalHeightChannel channel = new CausalHeightChannel(h0, tauStale, interval);

        Homography2D_F64 previous = new Homography2D_F64();
        Homography2D_F64 current = new Homography2D_F64();

        int sampleCursor = 0;
        double maxPos = 0.0, maxYaw = 0.0, maxHeight = 0.0, maxGsd = 0.0, maxRestartDq = 0.0;
        int restarts = 0;

        for (int k = 0; k < expected.size(); k++) {
            String[] row = expected.get(k);
            double t = Double.parseDouble(row[eCols.get("timestamp_s")]);
            String event = row[eCols.get("event")];

            // Causal replay: only samples already available at this frame time, in order, once each.
            while (sampleCursor < samples.size()
                    && Double.parseDouble(samples.get(sampleCursor)[0]) <= t) {
                channel.submit(Double.parseDouble(samples.get(sampleCursor)[0]),
                               Double.parseDouble(samples.get(sampleCursor)[1]));
                sampleCursor++;
            }
            HeightReading reading = channel.readAt(t);

            readHomography(transforms.get(k), tCols, current);
            if (k == 0) {
                rigid.observeOrigin();
                metric.observeOrigin(reading);
            } else if ("restart".equals(event)) {
                // The runtime does not observe the failing frame at all: the estimator restarts from
                // it, so the interval it spans is never estimated (COMP-001 section 8). Assert that
                // the recomposed increment there IS numerically zero, which is what makes skipping
                // it equivalent to the offline path's zero increment -- no motion is discarded here
                // and none is invented.
                rigid.observe(previous, current, width, height);
                maxRestartDq = Math.max(maxRestartDq,
                        Math.hypot(rigid.getIncrementDqX(), rigid.getIncrementDqY()));
                restarts++;
                metric.beginNewSegment(true);
                metric.observeOrigin(reading);
            } else {
                rigid.observe(previous, current, width, height);
                metric.observe(rigid.getIncrementDqX(), rigid.getIncrementDqY(),
                        rigid.getIncrementRotationRad(), reading);
            }
            previous.setTo(current);

            // --- position, yaw ------------------------------------------------------------------
            double expEast = Double.parseDouble(row[eCols.get("east_m")]);
            double expNorth = Double.parseDouble(row[eCols.get("north_m")]);
            double expYaw = Double.parseDouble(row[eCols.get("yaw_deg")]);
            maxPos = Math.max(maxPos, Math.abs(metric.eastM() - expEast));
            maxPos = Math.max(maxPos, Math.abs(metric.northM() - expNorth));
            maxYaw = Math.max(maxYaw, angleDiff(metric.metricPose().yaw, expYaw));

            // --- the height sample actually used, and its state ---------------------------------
            // Skipped on a restart row, and the reason is a real difference worth stating rather
            // than a tolerance to relax. There, the two implementations do different-but-equivalent
            // things: the offline path performs a NULL conversion (the recomposed increment is
            // exactly zero, asserted above) and so advances its "height used" to that frame's
            // sample, while the runtime performs NO conversion at all, because the estimator never
            // observes the frame it restarts from. Java's "the height the last increment was
            // converted with" therefore still names the previous segment's last increment, which is
            // the true statement; reporting a height as used when nothing was converted would not
            // be. Measured on hkairport01-a: the two differ by one ZOH step (0.1029 m) on that one
            // row and agree on every other row, and the integrated position across the restart moves
            // by 1.3e-15 m in the reference and by zero here.
            if (!"restart".equals(event)) {
                String expUsed = row[eCols.get("h_used_for_increment_m")];
                if (!expUsed.isEmpty() && metric.getLastUsedHeight() != null) {
                    maxHeight = Math.max(maxHeight, Math.abs(
                            metric.getLastUsedHeight().heightAglM() - Double.parseDouble(expUsed)));
                }
                String expGsd = row[eCols.get("gsd_used_m_per_px")];
                if (!expGsd.isEmpty() && !Double.isNaN(metric.getLastGsdMPerPx())) {
                    maxGsd = Math.max(maxGsd,
                            Math.abs(metric.getLastGsdMPerPx() - Double.parseDouble(expGsd)));
                }
            } else {
                // No zero-motion bridge is inserted by either side: the reference itself does not
                // move across the gap, so there is nothing for the runtime to reproduce.
                double prevEast = Double.parseDouble(expected.get(k - 1)[eCols.get("east_m")]);
                double prevNorth = Double.parseDouble(expected.get(k - 1)[eCols.get("north_m")]);
                assertEquals(prevEast, expEast, 1e-12, name + ": the reference moved across a gap");
                assertEquals(prevNorth, expNorth, 1e-12, name + ": the reference moved across a gap");
            }

            // --- validity and staleness ---------------------------------------------------------
            boolean expValid = "1".equals(row[eCols.get("h_valid")]);
            boolean expStale = "1".equals(row[eCols.get("h_stale")]);
            assertEquals(expValid, !reading.degraded(), () -> name + " frame " + row[0]
                    + ": Python valid=" + expValid + " but Java status=" + reading.status());
            assertEquals(expStale, reading.status() != HeightStatus.FRESH, () -> name + " frame "
                    + row[0] + ": Python stale=" + expStale + " but Java status=" + reading.status());
        }

        final double posErr = maxPos, yawErr = maxYaw, hErr = maxHeight, gsdErr = maxGsd;
        final double restartDq = maxRestartDq;
        assertTrue(posErr <= POS_TOL_M, () -> String.format(
                "%s: max metric position disagreement %.3e m exceeds %.1e m", name, posErr, POS_TOL_M));
        assertTrue(yawErr <= YAW_TOL_DEG, () -> String.format(
                "%s: max yaw disagreement %.3e deg exceeds %.1e deg", name, yawErr, YAW_TOL_DEG));
        assertTrue(hErr <= 1e-12, () -> name + ": height used disagrees by " + hErr + " m");
        assertTrue(gsdErr <= 1e-15, () -> name + ": gsd used disagrees by " + gsdErr + " m/px");
        if (restarts > 0) {
            assertTrue(restartDq < 1e-9, () -> name
                    + ": the increment at a restart frame is not zero (" + restartDq + " px), so "
                    + "skipping it would discard real motion");
        }

        System.out.printf("parity %-20s frames=%4d  maxPos=%.3e m  maxYaw=%.3e deg  "
                        + "maxH=%.3e m  maxGsd=%.3e m/px  restarts=%d (max |dq| %.3e px)%n",
                name, expected.size(), maxPos, maxYaw, maxHeight, maxGsd, restarts, maxRestartDq);
    }

    // ------------------------------------------------------------------ helpers

    private static double angleDiff(double a, double b) {
        double d = ((a - b) % 360.0 + 540.0) % 360.0 - 180.0;
        return Math.abs(d);
    }

    private static void readHomography(String[] row, Map<String, Integer> cols, Homography2D_F64 out) {
        double a33 = Double.parseDouble(row[cols.get("g22")]);
        out.a11 = Double.parseDouble(row[cols.get("g00")]) / a33;
        out.a12 = Double.parseDouble(row[cols.get("g01")]) / a33;
        out.a13 = Double.parseDouble(row[cols.get("g02")]) / a33;
        out.a21 = Double.parseDouble(row[cols.get("g10")]) / a33;
        out.a22 = Double.parseDouble(row[cols.get("g11")]) / a33;
        out.a23 = Double.parseDouble(row[cols.get("g12")]) / a33;
        out.a31 = Double.parseDouble(row[cols.get("g20")]) / a33;
        out.a32 = Double.parseDouble(row[cols.get("g21")]) / a33;
        out.a33 = 1.0;
    }

    private static Map<String, Integer> headerIndex(String[] header) {
        Map<String, Integer> idx = new LinkedHashMap<>();
        for (int i = 0; i < header.length; i++) {
            idx.put(header[i].trim(), i);
        }
        return idx;
    }

    private static List<String[]> readCsv(Path path) throws IOException {
        try (var lines = Files.lines(path, StandardCharsets.UTF_8)) {
            return lines.filter(l -> !l.isBlank())
                        .map(l -> l.split(",", -1))
                        .collect(Collectors.toCollection(ArrayList::new));
        }
    }

    /**
     * A deliberately small reader for the flat, machine-generated {@code cases.json}: string and
     * numeric scalars only, one object per case. Adding a JSON dependency to the test path to read a
     * file this project generates itself would be more machinery than the job needs.
     */
    private static List<Map<String, String>> readCases(Path path) throws IOException {
        String text = Files.readString(path, StandardCharsets.UTF_8);
        List<Map<String, String>> out = new ArrayList<>();
        Map<String, String> current = null;
        for (String raw : text.split("\n")) {
            String line = raw.trim();
            if (line.startsWith("{")) {
                current = new LinkedHashMap<>();
            } else if (line.startsWith("}") && current != null) {
                out.add(current);
                current = null;
            } else if (current != null && line.startsWith("\"")) {
                int colon = line.indexOf("\":");
                if (colon < 0) {
                    continue;
                }
                String key = line.substring(1, colon);
                String value = line.substring(colon + 2).trim();
                if (value.endsWith(",")) {
                    value = value.substring(0, value.length() - 1).trim();
                }
                if (value.startsWith("\"") && value.endsWith("\"") && value.length() >= 2) {
                    value = value.substring(1, value.length() - 1);
                }
                current.put(key, value);
            }
        }
        return out;
    }
}
