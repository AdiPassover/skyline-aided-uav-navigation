package org.boofcv.evaluation;

import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import org.boofcv.relocalization.AlignmentSidecarWriter;
import org.boofcv.relocalization.RelocalizationConfig;
import org.boofcv.relocalization.RelocalizationPipeline;
import org.boofcv.relocalization.SkylineProfileFile;
import org.boofcv.stitching.MotionModelStitchingEstimator;
import org.boofcv.stitching.ScriptedStitchingHarness;
import org.boofcv.stitching.ScriptedStitchingHarness.ScriptedStitching;
import org.boofcv.stitching.metric.HeadingSemantics;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * <b>The offline replay is the live loop's own input, replayed.</b>
 *
 * <p>The same scripted VO is run twice — once with the alignment sidecar alone (the VO_ONLY arm),
 * once with the relocalization pipeline inside the capture (the live INT arm) — and the replay is
 * driven from the VO_ONLY run's recorded track. If the replay were an approximation, anything in
 * the recorded contract that it reconstructed differently (a status, a gap flag, a NaN heading, the
 * hard-loss label) would show up as a differing sidecar row; the assertion is byte equality of both
 * sidecars, on a run that actually retrieves, re-anchors, loses the track and recovers.
 *
 * <p>Evidence tier: T1 (synthetic).
 */
public class RelocalizationReplayAppTest {

    private static final int FRAME = ScriptedStitchingHarness.FRAME;
    private static final int FRAMES = 60;
    private static final int LOSS_FRAME = 40;

    /** True position along the out-and-back track, metres east: 10 px per frame at 0.1 m/px. */
    private static int xOf(int frame) {
        if (frame <= 25) return frame;
        if (frame <= 50) return 50 - frame;
        return frame - 50;
    }

    /** Out 25 m, back 25 m, out again — so the return leg revisits the outward captures exactly. */
    private static FrameSource frames(ScriptedStitching stitch) {
        return new FrameSource() {
            @Override public int frameCount() { return FRAMES; }
            @Override public TimestampedFrame frame(int i) {
                if (i > 0) {
                    double step = (xOf(i) - xOf(i - 1)) * 10.0;
                    stitch.motion.advance(ScriptedStitchingHarness.translation(step));
                }
                return new TimestampedFrame(i, i * 0.1, new GrayF32(FRAME, FRAME));
            }
        };
    }

    private static final int[] CYCLES = {3, 5, 7, 11};                  // orthogonal on 256 samples
    private static final double[] NORTH_RATES = {0.5, 0.31, 0.73, 1.1};  // rad per metre of place
    private static final double[] WEST_RATES = {0.43, 0.67, 0.29, 0.97};

    /**
     * A place-dependent profile whose autocorrelation over place is
     * {@code mean(cos(rate_i · Δx))}: 1 at the same place, ≈ 0.76 one metre away (inside the 1.5 m
     * ambiguity ball), and no more than ≈ 0.36 anywhere from 2 to 22 m — so only the same place
     * clears 0.9 and every competitor leaves a margin far above 0.2.
     */
    private static String profile(int x, double[] rates) {
        StringBuilder sb = new StringBuilder();
        for (int k = 0; k < 256; k++) {
            double v = 0.0;
            for (int i = 0; i < CYCLES.length; i++) {
                v += Math.sin(2 * Math.PI * CYCLES[i] * k / 256.0 + rates[i] * x);
            }
            if (k > 0) sb.append(';');
            sb.append(String.format(Locale.ROOT, "%.9f", v));
        }
        return sb.toString();
    }

    private static Path writeProfiles(Path dir) throws IOException {
        StringBuilder sb = new StringBuilder("frame_index,timestamp_s,sync_dt_s,observation_id,valid,invalid_reason,profile,"
                + "west_observation_id,west_available,west_valid,west_invalid_reason,west_profile\n");
        for (int f = 2; f < FRAMES; f += 2) {           // a capture every second frame, both views
            int x = xOf(f);
            sb.append(String.format(Locale.ROOT, "%d,%.1f,0.0,sky_%03d,true,,%s,sky_%03d_west,true,true,,%s%n",
                    f, f * 0.1, f, profile(x, NORTH_RATES), f, profile(x, WEST_RATES)));
        }
        Path csv = dir.resolve(SkylineProfileFile.CSV_NAME);
        Files.writeString(csv, sb.toString(), StandardCharsets.UTF_8);
        return csv;
    }

    private static Path writeConfig(Path dir, String extra) throws IOException {
        Path cfg = dir.resolve("reloc.json");
        Files.writeString(cfg, "{\n"
                + "  \"matcher\": \"c0_frozen_ncc\", \"max_lag_samples\": 0,\n"
                + "  \"fusion_rule\": \"weakest_view\", \"temporal_confirmation\": \"fallback\",\n"
                + "  \"region_rule\": \"position_radius\", \"ambiguity_region_radius_m\": 1.5,\n"
                + "  \"dual_agreement_radius_m\": 1.5, \"temporal_region_radius_m\": 1.5,\n"
                + "  \"top_k\": 5, \"match_threshold\": 0.9, \"margin_threshold\": 0.2,\n"
                + "  \"recent_exclusion_mode\": \"time\", \"recent_exclusion_seconds\": 1.0,\n"
                + "  \"search_exposure_bound\": 100000, \"search_time_max_s\": 0.5, \"min_retry_gap_frames\": 0,\n"
                + "  \"novelty_min_distance\": 0.0, \"max_spacing_frames\": 1, \"skyline_sync_tolerance_s\": 0.0"
                + extra + "\n}\n", StandardCharsets.UTF_8);
        return cfg;
    }

    /** One live capture: VO_ONLY (alignment sidecar) or INT (the pipeline inside the run). */
    private static Path live(Path runDir, boolean withPipeline, Path cfg, Path profiles) throws IOException {
        Files.createDirectories(runDir);
        StringBuilder height = new StringBuilder("timestamp_s,baro_relative_m\n");
        StringBuilder heading = new StringBuilder("timestamp_s,heading_deg\n");
        for (int i = 0; i < FRAMES; i++) {
            height.append(String.format(Locale.ROOT, "%.3f,0.0%n", i * 0.1));
            heading.append(String.format(Locale.ROOT, "%.3f,0.0%n", i * 0.1));
        }
        Path heightCsv = runDir.resolve("height.csv");
        Path headingCsv = runDir.resolve("heading.csv");
        Files.writeString(heightCsv, height.toString(), StandardCharsets.UTF_8);
        Files.writeString(headingCsv, heading.toString(), StandardCharsets.UTF_8);

        ScriptedStitching stitch = new ScriptedStitching(0);
        MotionModelStitchingEstimator<GrayF32, Homography2D_F64> estimator =
                ScriptedStitchingHarness.headingEstimator(stitch, HeadingSemantics.SIM_NADIR_CAMERA_HEADING, null);
        VoRunner runner = new VoRunner(estimator, frames(stitch));
        runner.setMetricTrackCsv(runDir.resolve("metric_track.csv"),
                HeightSampleStream.fromCsv(heightCsv, "timestamp_s", "baro_relative_m", 0.0, 1, false));
        runner.setHeadingStream(HeadingSampleStream.fromCsv(headingCsv, "timestamp_s", "heading_deg", 0.0, 1));
        if (withPipeline) {
            RelocalizationConfig rc = RelocalizationConfig.load(cfg);
            runner.setRelocalization(new RelocalizationPipeline(rc), SkylineProfileFile.load(profiles, rc), runDir);
        } else {
            runner.setAlignmentSidecar(runDir);
        }
        TreeSet<Integer> injected = new TreeSet<>();
        injected.add(LOSS_FRAME);
        runner.setSyntheticHardLossFrames(injected);

        RunManifest manifest = new RunManifest();
        manifest.runId = withPipeline ? "live-int" : "vo-only";
        manifest.datasetId = "d";
        manifest.datasetRevision = "r";
        manifest.estimatorId = "stitching-vo";
        manifest.estimatorVersion = "test";
        manifest.evaluatorCaptureVersion = "test";
        manifest.estimatorConfig = Map.of();
        manifest.environment = new Environment("h", "os", "cpu", "19", 1024, false);
        manifest.runTimestamp = "2026-09-08T00:00:00Z";
        try (RunRecordWriter writer = new RunRecordWriter(runDir, manifest, FRAMES)) {
            runner.run(writer);
        }
        return runDir;
    }

    private static long count(Path events, String kind, String detailContains) throws IOException {
        List<String> lines = Files.readAllLines(events);
        String[] header = lines.get(0).split(",", -1);
        int kindCol = -1, srcCol = -1;
        for (int i = 0; i < header.length; i++) {
            if (header[i].equals("kind")) kindCol = i;
            if (header[i].equals("loss_source")) srcCol = i;
        }
        long n = 0;
        for (String l : lines.subList(1, lines.size())) {
            String[] f = l.split(",", -1);
            if (f[kindCol].equals(kind) && (detailContains == null || f[srcCol].equals(detailContains))) n++;
        }
        return n;
    }

    /** kind:reason counts, for a failure message that says what the scenario actually did. */
    private static String histogram(Path events) throws IOException {
        List<String> lines = Files.readAllLines(events);
        String[] header = lines.get(0).split(",", -1);
        int kindCol = -1, reasonCol = -1, detailCol = -1;
        for (int i = 0; i < header.length; i++) {
            if (header[i].equals("kind")) kindCol = i;
            if (header[i].equals("reason")) reasonCol = i;
            if (header[i].equals("detail")) detailCol = i;
        }
        java.util.TreeMap<String, Integer> h = new java.util.TreeMap<>();
        StringBuilder sample = new StringBuilder();
        int shown = 0;
        for (String l : lines.subList(1, lines.size())) {
            String[] f = l.split(",", -1);
            String why = f[kindCol].equals("attempt") ? f[detailCol].split(";", -1)[0] : "";
            h.merge(f[kindCol] + ":" + f[reasonCol] + (why.isEmpty() ? "" : ":" + why), 1, Integer::sum);
            if (f[kindCol].equals("retrieval") && f[reasonCol].isEmpty() && shown++ < 3) {
                sample.append("\n  retrieval@").append(f[0]);
                for (String c : new String[]{"top_k_ids", "top_k_scores", "top_region_score", "competing_region_score",
                        "region_margin", "north_score", "west_score", "excluded_recent_ids", "database_size"}) {
                    for (int i = 0; i < header.length; i++) {
                        if (header[i].equals(c)) sample.append(' ').append(c).append('=').append(f[i]);
                    }
                }
            }
        }
        return h + sample.toString();
    }

    @Test
    @DisplayName("replaying the recorded VO_ONLY track reproduces the live INT sidecars byte for byte")
    void replayEqualsLive(@TempDir Path tmp) throws IOException {
        Path profiles = writeProfiles(tmp);
        Path cfg = writeConfig(tmp, "");
        Path voOnly = live(tmp.resolve("vo-only"), false, cfg, profiles);
        Path liveInt = live(tmp.resolve("live-int"), true, cfg, profiles);

        Path liveEvents = liveInt.resolve(AlignmentSidecarWriter.EVENTS_FILE);
        assertTrue(count(liveEvents, "reanchor", null) >= 2,
                "the scenario must actually re-anchor (drift correction on the return leg and recovery after "
                        + "the loss); events were: " + histogram(liveEvents));
        assertEquals(1, count(liveEvents, "hard_loss", AlignmentSidecarWriter.LOSS_SYNTHETIC),
                "the injected loss is in the live record, labelled synthetic");

        Path replay = tmp.resolve("replay");
        Map<String, Object> summary = RelocalizationReplayApp.replay(voOnly, cfg, profiles, replay);
        assertEquals(FRAMES, summary.get("frames"));

        assertEquals(Files.readString(liveEvents),
                Files.readString(replay.resolve(AlignmentSidecarWriter.EVENTS_FILE)),
                "alignment_events.csv: the replay must be the live loop, not an approximation of it");
        assertEquals(Files.readString(liveInt.resolve(AlignmentSidecarWriter.FRAMES_FILE)),
                Files.readString(replay.resolve(AlignmentSidecarWriter.FRAMES_FILE)),
                "alignment_frames.csv: identical persistent track, validity, epochs and lineage");
        String manifest = Files.readString(replay.resolve(RelocalizationReplayApp.MANIFEST_NAME));
        assertTrue(manifest.contains("REPLAY over a recorded VO track"), "a replay says it is one");
        assertTrue(manifest.contains("PRIMARY (DEC-INT-007"), "the matcher role is recorded");
    }

    @Test
    @DisplayName("a config that needs the instability signal is refused: the recorded track cannot supply it")
    void instabilityConfigIsRefused(@TempDir Path tmp) throws IOException {
        Path profiles = writeProfiles(tmp);
        Path cfg = writeConfig(tmp, ",\n  \"instability_threshold_deg\": 5.0");
        Path voOnly = live(tmp.resolve("vo-only"), false, cfg, profiles);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> RelocalizationReplayApp.replay(voOnly, cfg, profiles, tmp.resolve("replay")));
        assertTrue(e.getMessage().contains("delta_rot_refit"));
    }

    @Test
    @DisplayName("two files that are not from one run are refused, not merged")
    void mismatchedTrackIsRefused(@TempDir Path tmp) throws IOException {
        Path profiles = writeProfiles(tmp);
        Path cfg = writeConfig(tmp, "");
        Path voOnly = live(tmp.resolve("vo-only"), false, cfg, profiles);
        // Corrupt one local position in alignment_frames.csv so it no longer equals metric_track's.
        Path af = voOnly.resolve(AlignmentSidecarWriter.FRAMES_FILE);
        List<String> lines = Files.readAllLines(af);
        String[] f = lines.get(5).split(",", -1);
        f[4] = "123.456";
        lines.set(5, String.join(",", f));
        Files.write(af, lines, StandardCharsets.UTF_8);
        IOException e = assertThrows(IOException.class,
                () -> RelocalizationReplayApp.replay(voOnly, cfg, profiles, tmp.resolve("replay")));
        assertTrue(e.getMessage().contains("not from one run"));
    }
}
