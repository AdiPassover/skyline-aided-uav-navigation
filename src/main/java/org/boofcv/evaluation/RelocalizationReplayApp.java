package org.boofcv.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.boofcv.relocalization.AlignmentSidecarWriter;
import org.boofcv.relocalization.HardLossEvent;
import org.boofcv.relocalization.LocalPoseSample;
import org.boofcv.relocalization.PlanarPosition;
import org.boofcv.relocalization.RelocalizationConfig;
import org.boofcv.relocalization.RelocalizationPipeline;
import org.boofcv.relocalization.SkylineObservation;
import org.boofcv.relocalization.SkylineProfileFile;
import org.boofcv.stitching.metric.HeadingStatus;
import org.boofcv.stitching.metric.HeightStatus;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Re-drives the relocalization layer over a <b>recorded</b> VO track — the same
 * {@link RelocalizationPipeline}, the same {@link AlignmentSidecarWriter}, the same skyline profile
 * file — without running the stitching VO again.
 *
 * <p>The layer consumes exactly one contract, {@link LocalPoseSample} ({@code DEC-INT-001}
 * amendment), and a VO run made with {@code metric_readout} + {@code heading_readout} +
 * {@code alignment_sidecar} already writes every field of that record per frame:
 * {@code metric_track.csv} carries the segment index, the two gap flags, the segment-relative
 * East/North, {@code metric_frame_usable}, the height status, the authoritative heading with its
 * status and the diagnostic visual yaw; {@code alignment_frames.csv} carries the {@code processFrame}
 * boolean. Rebuilding the samples from those columns and stepping the pipeline over them is
 * therefore not an approximation of the live loop but the live loop's own input replayed —
 * {@code RelocalizationReplayAppTest} asserts that the sidecars it writes are byte-identical to the
 * ones {@code VoRunner} writes when the pipeline runs inside the capture.
 *
 * <p><b>What it is for.</b> Policy calibration on DEVELOPMENT recordings ({@code EXP-INT-001}
 * §calibration): the VO is run once per recording, and every candidate
 * {@link RelocalizationConfig} is replayed over the recorded track in seconds instead of minutes.
 * The pre-registered milestone arms themselves are <em>live</em> {@code VoRunnerApp} runs; a replay
 * is labelled {@code mode = REPLAY} in its manifest and is never presented as one.
 *
 * <p><b>What it cannot do.</b> Anything that would feed back into the VO — there is nothing to
 * feed back into. The {@code delta_rot_refit} instability signal is not in the recorded track, so a
 * config that sets {@code instability_threshold_deg} is refused. Hard losses are replayed as
 * recorded and keep the recorded {@code loss_source} label (natural / synthetic), so a forced loss
 * stays distinguishable from an observed one here as well ({@code DEC-INT-006}).
 *
 * <pre>
 *   java -cp "build/install/skyline-aided-uav-navigation/lib/*" org.boofcv.evaluation.RelocalizationReplayApp \
 *        --vo-run runs/&lt;vo-only run&gt; --relocalization-config &lt;RelocalizationConfig JSON&gt; \
 *        --skyline-profiles datasets/&lt;id&gt;/skyline_profiles.csv --out &lt;dir&gt;
 * </pre>
 */
public final class RelocalizationReplayApp {

    public static final String MANIFEST_NAME = "relocalization_manifest.json";
    static final String MODE = "REPLAY over a recorded VO track (metric_track.csv + alignment_frames.csv); "
            + "the stitching VO did not run — not a live capture";

    /** One recorded frame of the VO → INT handoff. */
    record RecordedFrame(LocalPoseSample sample, String lossSourceIfHardLoss) {
    }

    private RelocalizationReplayApp() {
    }

    public static void main(String[] args) throws IOException {
        Map<String, String> a = parseArgs(args);
        Path voRun = Paths.get(require(a, "--vo-run"));
        Path configPath = Paths.get(require(a, "--relocalization-config"));
        Path profilesPath = Paths.get(require(a, "--skyline-profiles"));
        Path out = Paths.get(require(a, "--out"));
        Map<String, Object> summary = replay(voRun, configPath, profilesPath, out);
        System.out.println("Replay written to " + out + ": " + summary.get("frames") + " frames, "
                + summary.get("retrievals") + " retrievals, " + summary.get("reanchors") + " re-anchors");
    }

    /**
     * Runs the replay and writes {@code alignment_frames.csv}, {@code alignment_events.csv} and the
     * manifest into {@code out}.
     *
     * @return a small summary (frames, retrievals, re-anchors)
     */
    public static Map<String, Object> replay(Path voRun, Path configPath, Path profilesPath, Path out)
            throws IOException {
        RelocalizationConfig rc = RelocalizationConfig.load(configPath);
        if (rc.instabilityThresholdDeg != null) {
            throw new IllegalArgumentException("instability_threshold_deg is set, but delta_rot_refit is not "
                    + "part of the recorded track; a replay cannot supply it. Refused rather than replayed "
                    + "without the signal.");
        }
        List<RecordedFrame> frames = readTrack(voRun);
        SkylineProfileFile profiles = SkylineProfileFile.load(profilesPath, rc);
        if (profiles.matchedCount(frames.size()) == 0) {
            throw new IllegalArgumentException("skyline_profiles " + profilesPath + ": none of its "
                    + profiles.size() + " frames fall inside the recorded track's " + frames.size()
                    + " frames — a misaligned feed, refused");
        }

        Files.createDirectories(out);
        RelocalizationPipeline pipeline = new RelocalizationPipeline(rc);
        int retrievals = 0, reanchors = 0, hardLossesWritten = 0;
        try (AlignmentSidecarWriter sidecar = new AlignmentSidecarWriter(out)) {
            sidecar.setReferenceMemory(pipeline.memory());
            for (RecordedFrame f : frames) {
                LocalPoseSample s = f.sample();
                SkylineObservation obs = profiles.lookup(s.frameIndex(), s.timestampS());
                RelocalizationPipeline.FrameStep step = pipeline.step(s, obs, null);
                List<HardLossEvent> losses = pipeline.aligner().hardLossEvents();
                while (hardLossesWritten < losses.size()) {
                    HardLossEvent e = losses.get(hardLossesWritten++);
                    sidecar.writeHardLoss(e, f.lossSourceIfHardLoss() != null
                            ? f.lossSourceIfHardLoss() : AlignmentSidecarWriter.LOSS_NATURAL);
                }
                sidecar.writeStep(step);
                if (step.retrieval() != null) {
                    retrievals++;
                }
                if (step.reanchor() != null) {
                    reanchors++;
                }
            }
        }

        Map<String, Object> m = new LinkedHashMap<>();
        m.put("mode", MODE);
        m.put("source_vo_run", voRun.toString());
        m.put("source_vo_run_manifest", readJsonIfPresent(voRun.resolve("manifest.json")));
        Map<String, Object> digests = new LinkedHashMap<>();
        digests.put("metric_track.csv", sha256(voRun.resolve("metric_track.csv")));
        digests.put("alignment_frames.csv", sha256(voRun.resolve("alignment_frames.csv")));
        digests.put("skyline_profiles.csv", sha256(profilesPath));
        digests.put("relocalization_config", sha256(configPath));
        m.put("input_sha256", digests);
        m.put("relocalization_config_path", configPath.toString());
        m.put("relocalization_config", new ObjectMapper().convertValue(rc, Map.class));
        m.put("skyline_profiles_path", profilesPath.toString());
        m.put("skyline_pairing", Map.of(
                "identity", profiles.pairingIdentity(), "mechanism", profiles.pairingMechanism(),
                "schema_version", profiles.schemaVersion(), "rows", profiles.size(),
                "north_valid", profiles.validCount(), "west_available", profiles.westAvailableCount(),
                "west_valid", profiles.westValidCount(), "sync_rejected", profiles.syncRejectedCount(),
                "max_abs_sync_dt_s", profiles.maxAbsSyncDtS()));
        m.put("instability_source", "not available in a replay (refused when configured)");
        m.put("pose_rule", "discrete_reference (p_reloc = p_ref, position only; DEC-INT-002)");
        m.put("matcher", VoRunnerApp.matcherBlock(rc));
        m.put("dual_view", VoRunnerApp.dualViewBlock(rc));
        m.put("snap_safety_units", org.boofcv.relocalization.SnapSafety.UNITS);
        Map<String, Object> counts = new LinkedHashMap<>();
        counts.put("frames", frames.size());
        counts.put("retrievals", retrievals);
        counts.put("reanchors", reanchors);
        counts.put("hard_losses_replayed", hardLossesWritten);
        m.put("counts", counts);
        m.put("evidence_tier_note", "a replay of recorded mechanism logs; no navigation-performance claim");
        new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT)
                .writeValue(out.resolve(MANIFEST_NAME).toFile(), m);
        Map<String, Object> summary = new LinkedHashMap<>(counts);
        return summary;
    }

    /**
     * Rebuilds the per-frame {@link LocalPoseSample}s from a VO run's {@code metric_track.csv} and
     * {@code alignment_frames.csv}, and the recorded hard-loss labels from its
     * {@code alignment_events.csv}.
     */
    static List<RecordedFrame> readTrack(Path voRun) throws IOException {
        List<Map<String, String>> metric = readCsv(voRun.resolve("metric_track.csv"));
        List<Map<String, String>> align = readCsv(voRun.resolve("alignment_frames.csv"));
        if (metric.size() != align.size()) {
            throw new IOException(voRun + ": metric_track.csv has " + metric.size()
                    + " rows but alignment_frames.csv has " + align.size());
        }
        Map<Integer, String> lossSource = new HashMap<>();
        Path events = voRun.resolve("alignment_events.csv");
        if (Files.exists(events)) {
            for (Map<String, String> e : readCsv(events)) {
                if ("hard_loss".equals(e.get("kind"))) {
                    String src = e.get("loss_source");
                    lossSource.put(Integer.parseInt(e.get("frame_index")),
                            src == null || src.isEmpty() ? AlignmentSidecarWriter.LOSS_NATURAL : src);
                }
            }
        }
        List<RecordedFrame> out = new ArrayList<>(metric.size());
        for (int i = 0; i < metric.size(); i++) {
            Map<String, String> m = metric.get(i);
            Map<String, String> a = align.get(i);
            int frame = Integer.parseInt(m.get("frame_index"));
            if (frame != Integer.parseInt(a.get("frame_index"))) {
                throw new IOException(voRun + ": row " + i + " is frame " + frame + " in metric_track.csv but "
                        + a.get("frame_index") + " in alignment_frames.csv");
            }
            double ts = Double.parseDouble(m.get("timestamp_s"));
            double segE = Double.parseDouble(m.get("segment_east_m"));
            double segN = Double.parseDouble(m.get("segment_north_m"));
            double localE = Double.parseDouble(a.get("local_east_m"));
            double localN = Double.parseDouble(a.get("local_north_m"));
            if (localE != segE || localN != segN) {
                throw new IOException(voRun + ": frame " + frame + ": alignment_frames local position ("
                        + localE + ", " + localN + ") is not metric_track's segment position (" + segE + ", "
                        + segN + ") — the two files are not from one run");
            }
            // metric_track.csv writes its flags as 1/0 and alignment_frames.csv as true/false.
            HeadingStatus headingStatus = HeadingStatus.valueOf(m.get("yaw_nav_status").trim());
            double heading = parseDoubleOrNaN(m.get("yaw_nav_deg"));
            // h_status is the reading THIS frame was converted with (blank on the origin frame and on
            // a frame that converted nothing); the live sample reports the reference reading's status.
            // The field feeds only the translation-dropout event's detail, never a state transition,
            // so a dropout row of a replay can read UNAVAILABLE where the live row said HELD/STALE.
            String h = m.get("h_status").trim();
            HeightStatus heightStatus = h.isEmpty() ? HeightStatus.UNAVAILABLE : HeightStatus.valueOf(h);
            LocalPoseSample s = new LocalPoseSample(frame, ts, flag(a.get("vo_success")),
                    Integer.parseInt(m.get("segment_index").trim()),
                    flag(m.get("unknown_translation_gap")),
                    flag(m.get("heading_known_across_gap")),
                    new PlanarPosition(segE, segN),
                    flag(m.get("metric_frame_usable")),
                    heightStatus,
                    heading, headingStatus, parseDoubleOrNaN(m.get("raw_yaw_deg")));
            out.add(new RecordedFrame(s, lossSource.get(frame)));
        }
        return out;
    }

    /** {@code 1} / {@code true} → true; {@code 0} / {@code false} / blank → false. */
    private static boolean flag(String s) {
        String v = s == null ? "" : s.trim();
        return "1".equals(v) || "true".equalsIgnoreCase(v);
    }

    private static double parseDoubleOrNaN(String s) {
        if (s == null || s.isBlank() || "NaN".equalsIgnoreCase(s.trim())) {
            return Double.NaN;
        }
        return Double.parseDouble(s.trim());
    }

    static List<Map<String, String>> readCsv(Path path) throws IOException {
        List<Map<String, String>> rows = new ArrayList<>();
        try (BufferedReader in = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String header = in.readLine();
            if (header == null) {
                throw new IOException(path + ": empty");
            }
            String[] names = header.split(",", -1);
            String line;
            while ((line = in.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                String[] f = line.split(",", -1);
                if (f.length != names.length) {
                    throw new IOException(path + ": expected " + names.length + " fields, got " + f.length);
                }
                Map<String, String> row = new HashMap<>(names.length * 2);
                for (int i = 0; i < names.length; i++) {
                    row.put(names[i], f[i]);
                }
                rows.add(row);
            }
        }
        return rows;
    }

    @SuppressWarnings("unchecked")
    private static Object readJsonIfPresent(Path p) throws IOException {
        return Files.exists(p) ? new ObjectMapper().readValue(p.toFile(), Map.class) : "absent";
    }

    static String sha256(Path p) throws IOException {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] d = md.digest(Files.readAllBytes(p));
            StringBuilder sb = new StringBuilder();
            for (byte b : d) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i + 1 < args.length; i += 2) {
            m.put(args[i], args[i + 1]);
        }
        return m;
    }

    private static String require(Map<String, String> a, String key) {
        String v = a.get(key);
        if (v == null) {
            throw new IllegalArgumentException("missing " + key + "; usage: --vo-run <dir> "
                    + "--relocalization-config <json> --skyline-profiles <csv> --out <dir>");
        }
        return v;
    }
}
