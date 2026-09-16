package org.boofcv.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.boofcv.relocalization.LocalPoseSample;
import org.boofcv.relocalization.MatchResult;
import org.boofcv.relocalization.RelocalizationConfig;
import org.boofcv.relocalization.RelocalizationPipeline;
import org.boofcv.relocalization.SkylineDescriptor;
import org.boofcv.relocalization.SkylineMatcher;
import org.boofcv.relocalization.SkylineObservation;
import org.boofcv.relocalization.SkylineProfileFile;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Claim-closure Phase 4 (2026-09): the cost of the relocalization layer, measured on a recorded track
 * with the frozen policy, plus the exact-retrieval scaling of the C0 matcher. Experiment-only
 * instrumentation: a single-file program launched against the installed jars, in the evaluation
 * package so it can reuse {@link RelocalizationReplayApp#readTrack} — it changes nothing under
 * {@code src/}.
 *
 * <pre>
 *   java -cp "build/install/skyline-aided-uav-navigation/lib/*" evaluation/tools/claim_closure/IntLayerBench.java \
 *        &lt;vo-run&gt; &lt;policy.json&gt; &lt;skyline_profiles.csv&gt; &lt;repetitions&gt; &lt;out.json&gt;
 * </pre>
 *
 * Per frame the whole {@code RelocalizationPipeline.step} is timed (aligner + instability input +
 * scheduler + retrieval + gate + memory insertion); frames are split into those on which a retrieval
 * ran and those on which none did. The last repetition is reported (earlier ones warm the JIT).
 * The matcher scaling benchmark times one query against N synthetic references with the frozen C0
 * matcher exactly as {@code ReferenceMemory.retrieve} calls it, for N in 10 … 20 000.
 */
public class IntLayerBench {

    public static void main(String[] args) throws Exception {
        Path voRun = Paths.get(args[0]);
        Path cfg = Paths.get(args[1]);
        Path profilesPath = Paths.get(args[2]);
        int reps = Integer.parseInt(args[3]);
        Path outJson = Paths.get(args[4]);

        RelocalizationConfig rc = RelocalizationConfig.load(cfg);
        List<RelocalizationReplayApp.RecordedFrame> frames = RelocalizationReplayApp.readTrack(voRun);
        SkylineProfileFile profiles = SkylineProfileFile.load(profilesPath, rc);

        long[] stepNs = new long[frames.size()];
        boolean[] retrieved = new boolean[frames.size()];
        boolean[] hadSkyline = new boolean[frames.size()];
        int memorySize = 0;
        long wallNs = 0;
        for (int rep = 0; rep < reps; rep++) {
            RelocalizationPipeline pipeline = new RelocalizationPipeline(rc);
            long t0 = System.nanoTime();
            for (int i = 0; i < frames.size(); i++) {
                LocalPoseSample s = frames.get(i).sample();
                SkylineObservation obs = profiles.lookup(s.frameIndex(), s.timestampS());
                long a = System.nanoTime();
                RelocalizationPipeline.FrameStep step = pipeline.step(s, obs, null);
                long b = System.nanoTime();
                stepNs[i] = b - a;
                retrieved[i] = step.retrieval() != null;
                hadSkyline[i] = obs != null;
            }
            wallNs = System.nanoTime() - t0;
            memorySize = pipeline.memory().size();
        }

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("vo_run", voRun.toString());
        out.put("policy", cfg.toString());
        out.put("frames", frames.size());
        out.put("repetitions", reps);
        out.put("memory_size_at_end", memorySize);
        out.put("wall_ns_last_rep", wallNs);
        out.put("per_frame_all", stats(stepNs, null));
        out.put("per_frame_with_retrieval", stats(stepNs, retrieved));
        boolean[] noRetrieval = new boolean[frames.size()];
        boolean[] skylineNoRetrieval = new boolean[frames.size()];
        for (int i = 0; i < frames.size(); i++) {
            noRetrieval[i] = !retrieved[i];
            skylineNoRetrieval[i] = hadSkyline[i] && !retrieved[i];
        }
        out.put("per_frame_without_retrieval", stats(stepNs, noRetrieval));
        out.put("per_frame_skyline_present_no_retrieval", stats(stepNs, skylineNoRetrieval));
        long peakHeap = 0;
        for (MemoryPoolMXBean p : ManagementFactory.getMemoryPoolMXBeans()) {
            if (p.getType() == java.lang.management.MemoryType.HEAP && p.getPeakUsage() != null) {
                peakHeap += p.getPeakUsage().getUsed();
            }
        }
        out.put("jvm_peak_heap_used_bytes_sum_of_pools", peakHeap);
        out.put("matcher_scaling", matcherScaling(rc));
        new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT).writeValue(outJson.toFile(), out);
        System.out.println("bench written to " + outJson);
    }

    static Map<String, Object> stats(long[] ns, boolean[] mask) {
        List<Long> v = new ArrayList<>();
        for (int i = 0; i < ns.length; i++) {
            if (mask == null || mask[i]) {
                v.add(ns[i]);
            }
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("n", v.size());
        if (v.isEmpty()) {
            return m;
        }
        long[] a = v.stream().mapToLong(Long::longValue).sorted().toArray();
        m.put("median_us", a[a.length / 2] / 1e3);
        m.put("p95_us", a[(int) Math.floor(0.95 * (a.length - 1))] / 1e3);
        m.put("p99_us", a[(int) Math.floor(0.99 * (a.length - 1))] / 1e3);
        m.put("max_us", a[a.length - 1] / 1e3);
        m.put("mean_us", Arrays.stream(a).average().orElse(0) / 1e3);
        return m;
    }

    /** One query against N references with the frozen C0 matcher (single view), as the memory scores it. */
    static List<Map<String, Object>> matcherScaling(RelocalizationConfig rc) {
        SkylineMatcher matcher = SkylineMatcher.fromConfig(rc);
        Random rnd = new Random(20260910);
        int n = rc.descriptorLength;
        SkylineDescriptor query = synthetic(rnd, n, rc.degenerateStdFloor);
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int N : new int[] {10, 100, 1000, 10000, 20000}) {
            List<SkylineDescriptor> refs = new ArrayList<>(N);
            for (int i = 0; i < N; i++) {
                refs.add(synthetic(rnd, n, rc.degenerateStdFloor));
            }
            double sink = 0;
            int trials = N <= 1000 ? 200 : 20;
            long best = Long.MAX_VALUE;
            for (int t = 0; t < trials + 5; t++) {
                long a = System.nanoTime();
                for (SkylineDescriptor r : refs) {
                    MatchResult mr = matcher.match(query, r);
                    sink += mr.score();
                }
                long d = System.nanoTime() - a;
                if (t >= 5) {
                    best = Math.min(best, d);
                }
            }
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("N", N);
            row.put("matcher", matcher.variant());
            row.put("best_of_trials_us_per_query_single_view", best / 1e3);
            row.put("ns_per_reference", best / (double) N);
            row.put("sink", sink);
            rows.add(row);
        }
        return rows;
    }

    static SkylineDescriptor synthetic(Random rnd, int n, double floor) {
        double[] p = new double[n];
        double a = 0.02 + 0.05 * rnd.nextDouble(), b = 0.01 + 0.03 * rnd.nextDouble();
        double f1 = 1 + 4 * rnd.nextDouble(), f2 = 5 + 10 * rnd.nextDouble(), ph = rnd.nextDouble() * Math.PI;
        for (int i = 0; i < n; i++) {
            double x = i / (double) (n - 1);
            p[i] = 0.5 + a * Math.sin(2 * Math.PI * f1 * x + ph) + b * Math.sin(2 * Math.PI * f2 * x) + 0.002 * rnd.nextGaussian();
        }
        return SkylineDescriptor.of(p, n, floor);
    }
}
