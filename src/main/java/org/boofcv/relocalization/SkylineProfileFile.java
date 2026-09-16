package org.boofcv.relocalization;

import com.fasterxml.jackson.databind.ObjectMapper;

import javax.annotation.Nullable;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The file-based SKY → INT boundary ({@code DEC-INT-002}, dual view per {@code DEC-INT-003}): a
 * canonical per-frame profile export produced by the validated Python pipeline
 * ({@code evaluation/tools/int/export_skyline_profiles.py}) and read here into
 * {@link SkylineObservation}s. No image is opened and no extraction runs in Java.
 *
 * <h2>Formats — {@code skyline_profiles.csv}</h2>
 *
 * <p>Schema 1 (North only, every export before 2026-09-07):
 * <pre>
 *   frame_index,timestamp_s,observation_id,valid,invalid_reason,profile
 * </pre>
 *
 * <p>Schema 2 (North + optional synchronised West, and the time-pairing residual):
 * <pre>
 *   frame_index,timestamp_s,sync_dt_s,observation_id,valid,invalid_reason,profile,
 *   west_observation_id,west_available,west_valid,west_invalid_reason,west_profile
 * </pre>
 *
 * <p>{@code frame_index} is the <b>VO dataset's</b> frame index the observation is paired to (by
 * the exporter's explicit map, index offset, or nearest capture time); {@code sync_dt_s} is the
 * skyline capture time minus the VO frame time when paired by time, empty otherwise;
 * {@code timestamp_s} is the SKY side's clock and is kept for cross-reference only — the VO
 * frame's own timestamp is what the pipeline uses; {@code profile} is {@code n} semicolon-separated
 * samples of the resampled, mean-removed height profile, empty when {@code valid} is false.
 * {@code west_available} is false when the capture carried no West view (then every other West
 * field is empty) — a missing West is represented, never fabricated. A companion
 * {@code skyline_profiles.json} carries the producer's description and is echoed into the run's
 * relocalization manifest when present.
 *
 * <p>Loading is strict: a wrong profile length, a duplicate frame, a non-finite sample or a
 * malformed row is an error, never a repair. A row whose {@code |sync_dt_s|} exceeds the configured
 * {@code skyline_sync_tolerance_s} becomes an <em>invalid</em> observation naming the residual —
 * explicit, not dropped. A frame the file does not mention simply has no observation
 * ({@link #lookup} returns {@code null}); a frame the file mentions that the VO run does not contain
 * is reported by {@link #framesOutside} so the caller can refuse a misaligned feed.
 *
 * <h2>Capture identity — exact or approximate, never mixed</h2>
 *
 * <p>The manifest's {@code frame_pairing.identity} says how {@code frame_index} was obtained:
 * {@link #IDENTITY_EXACT} — the simulator recorded which nadir/VO frame consumed each skyline
 * capture ({@code vo_frame_id}; recordings since 2026-09-07), verified against the VO clock, so one
 * skyline observation is one VO frame is one query-time local pose, and North and West are one
 * simulator row; {@link #IDENTITY_APPROXIMATE} — the legacy nearest-capture-time pairing with its
 * per-row {@code sync_dt_s} residual; {@link #IDENTITY_DECLARED} — an operator's frame map or
 * offset; {@link #IDENTITY_UNKNOWN} — no manifest (a pre-2.1 export). Under an exact identity a
 * row carrying a non-zero residual is refused: the file would be mixing the two contracts.
 */
public final class SkylineProfileFile {

    public static final String CSV_NAME = "skyline_profiles.csv";
    public static final String MANIFEST_NAME = "skyline_profiles.json";
    public static final String IDENTITY_EXACT = "exact";
    public static final String IDENTITY_APPROXIMATE = "approximate";
    public static final String IDENTITY_DECLARED = "declared";
    public static final String IDENTITY_UNKNOWN = "unknown";
    /** The largest residual an exact identity may carry — clock arithmetic, not a pairing slack. */
    static final double EXACT_TOLERANCE_S = 1e-6;
    static final String HEADER_V1 = "frame_index,timestamp_s,observation_id,valid,invalid_reason,profile";
    static final String HEADER_V2 = "frame_index,timestamp_s,sync_dt_s,observation_id,valid,invalid_reason,profile,"
            + "west_observation_id,west_available,west_valid,west_invalid_reason,west_profile";

    private record Row(int frameIndex, double timestampS, @Nullable Double syncDtS, SkylineView north,
                       @Nullable SkylineView west) {
    }

    private final Path path;
    private final int schemaVersion;
    private final Map<Integer, Row> rows;
    @Nullable private final Map<String, Object> manifest;
    private final int validCount;
    private final int westAvailableCount;
    private final int westValidCount;
    private final int syncRejectedCount;
    private final double maxAbsSyncDtS;
    private final String pairingIdentity;
    private final String pairingMechanism;

    private SkylineProfileFile(Path path, int schemaVersion, Map<Integer, Row> rows,
                               @Nullable Map<String, Object> manifest, int validCount,
                               int westAvailableCount, int westValidCount, int syncRejectedCount,
                               double maxAbsSyncDtS, String pairingIdentity, String pairingMechanism) {
        this.path = path;
        this.schemaVersion = schemaVersion;
        this.rows = rows;
        this.manifest = manifest;
        this.validCount = validCount;
        this.westAvailableCount = westAvailableCount;
        this.westValidCount = westValidCount;
        this.syncRejectedCount = syncRejectedCount;
        this.maxAbsSyncDtS = maxAbsSyncDtS;
        this.pairingIdentity = pairingIdentity;
        this.pairingMechanism = pairingMechanism;
    }

    @SuppressWarnings("unchecked")
    public static SkylineProfileFile load(Path csv, RelocalizationConfig config) throws IOException {
        if (csv == null || config == null) {
            throw new IllegalArgumentException("csv path and config are required");
        }
        config.validate();
        Map<Integer, Row> rows = new LinkedHashMap<>();
        int valid = 0, westAvail = 0, westValid = 0, syncRejected = 0;
        double maxAbsDt = 0.0;
        try (BufferedReader in = Files.newBufferedReader(csv, StandardCharsets.UTF_8)) {
            String header = in.readLine();
            int schema;
            if (header != null && HEADER_V1.equals(header.trim())) {
                schema = 1;
            } else if (header != null && HEADER_V2.equals(header.trim())) {
                schema = 2;
            } else {
                throw new IOException(csv + ": expected header '" + HEADER_V1 + "' or '" + HEADER_V2
                        + "', got '" + header + "'");
            }
            int expectedFields = schema == 1 ? 6 : 12;
            String line;
            int lineNo = 1;
            while ((line = in.readLine()) != null) {
                lineNo++;
                if (line.isBlank()) {
                    continue;
                }
                String[] f = line.split(",", -1);
                if (f.length != expectedFields) {
                    throw new IOException(csv + ":" + lineNo + ": expected " + expectedFields
                            + " fields, got " + f.length);
                }
                int frame;
                double ts;
                try {
                    frame = Integer.parseInt(f[0].trim());
                    ts = Double.parseDouble(f[1].trim());
                } catch (NumberFormatException e) {
                    throw new IOException(csv + ":" + lineNo + ": bad frame_index/timestamp_s", e);
                }
                if (rows.containsKey(frame)) {
                    throw new IOException(csv + ":" + lineNo + ": duplicate frame_index " + frame);
                }
                int o = schema == 1 ? 2 : 3;            // index of observation_id
                Double syncDt = null;
                if (schema == 2 && !f[2].trim().isEmpty()) {
                    try {
                        syncDt = Double.parseDouble(f[2].trim());
                    } catch (NumberFormatException e) {
                        throw new IOException(csv + ":" + lineNo + ": bad sync_dt_s", e);
                    }
                    if (!Double.isFinite(syncDt)) {
                        throw new IOException(csv + ":" + lineNo + ": non-finite sync_dt_s");
                    }
                    maxAbsDt = Math.max(maxAbsDt, Math.abs(syncDt));
                }
                String syncReason = null;
                if (syncDt != null && Math.abs(syncDt) > config.skylineSyncToleranceS) {
                    syncReason = String.format("sync residual %.4f s exceeds skyline_sync_tolerance_s %.4f s",
                            syncDt, config.skylineSyncToleranceS);
                    syncRejected++;
                }

                SkylineView north = view(SkylineView.NORTH, f[o], f[o + 1], f[o + 2], f[o + 3],
                        config, csv, lineNo, syncReason);
                SkylineView west = null;
                if (schema == 2) {
                    boolean available = "true".equalsIgnoreCase(f[8].trim());
                    if (available) {
                        westAvail++;
                        west = view(SkylineView.WEST, f[7], f[9], f[10], f[11], config, csv, lineNo,
                                syncReason);
                        if (west.valid()) {
                            westValid++;
                        }
                    } else if (!f[9].trim().isEmpty() || !f[11].trim().isEmpty()) {
                        throw new IOException(csv + ":" + lineNo
                                + ": west_available=false but West fields are populated");
                    }
                }
                if (north.valid()) {
                    valid++;
                }
                rows.put(frame, new Row(frame, ts, syncDt, north, west));
            }
            Map<String, Object> manifest = null;
            Path manifestPath = csv.resolveSibling(MANIFEST_NAME);
            if (Files.exists(manifestPath)) {
                manifest = new ObjectMapper().readValue(manifestPath.toFile(), Map.class);
            }
            String identity = IDENTITY_UNKNOWN;
            String mechanism = "unknown";
            if (manifest != null && manifest.get("frame_pairing") instanceof Map<?, ?> fp) {
                Object id = fp.get("identity");
                Object mech = fp.get("mechanism");
                if (mech != null) {
                    mechanism = mech.toString();
                }
                if (id != null) {
                    identity = id.toString();
                    if (!IDENTITY_EXACT.equals(identity) && !IDENTITY_APPROXIMATE.equals(identity)
                            && !IDENTITY_DECLARED.equals(identity)) {
                        throw new IOException(manifestPath + ": frame_pairing.identity '" + identity
                                + "' is not exact / approximate / declared");
                    }
                } else if ("vo_frames_nearest_time".equals(mechanism)) {
                    identity = IDENTITY_APPROXIMATE;          // a 2.0 export: time-paired, no identity field
                } else if ("frame_map".equals(mechanism) || "frame_offset".equals(mechanism)) {
                    identity = IDENTITY_DECLARED;
                }
            }
            if (IDENTITY_EXACT.equals(identity)) {
                if (syncRejected > 0 || maxAbsDt > EXACT_TOLERANCE_S) {
                    throw new IOException(csv + ": the manifest declares an exact capture identity but a row "
                            + "carries a time residual of up to " + maxAbsDt + " s — exact and approximate "
                            + "pairings are never mixed in one file");
                }
                for (Row r : rows.values()) {
                    if (r.syncDtS == null) {
                        throw new IOException(csv + ": frame " + r.frameIndex + " has no sync_dt_s under an "
                                + "exact capture identity (the verified residual must be written, as 0)");
                    }
                }
            }
            return new SkylineProfileFile(csv, schema, rows, manifest, valid, westAvail, westValid,
                    syncRejected, maxAbsDt, identity, mechanism);
        }
    }

    private static SkylineView view(String name, String idField, String validField, String reasonField,
                                    String profileField, RelocalizationConfig config, Path csv,
                                    int lineNo, @Nullable String syncReason) throws IOException {
        String id = idField.trim().isEmpty() ? null : idField.trim();
        boolean isValid = "true".equalsIgnoreCase(validField.trim());
        if (!isValid) {
            String reason = reasonField.trim().isEmpty() ? "invalid (no reason given)" : reasonField.trim();
            return SkylineView.invalid(name, reason, id);
        }
        double[] profile = parseProfile(profileField, csv, lineNo, name);
        SkylineDescriptor d = SkylineDescriptor.of(profile, config.descriptorLength, config.degenerateStdFloor);
        if (d == null) {
            return SkylineView.invalid(name, "degenerate profile (std below " + config.degenerateStdFloor + ")", id);
        }
        if (syncReason != null) {
            return SkylineView.invalid(name, syncReason, id);
        }
        return SkylineView.valid(name, d, id);
    }

    private static double[] parseProfile(String field, Path csv, int lineNo, String view) throws IOException {
        String s = field.trim();
        if (s.isEmpty()) {
            throw new IOException(csv + ":" + lineNo + ": valid " + view + " row with an empty profile");
        }
        String[] parts = s.split(";", -1);
        double[] out = new double[parts.length];
        try {
            for (int i = 0; i < parts.length; i++) {
                out[i] = Double.parseDouble(parts[i].trim());
            }
        } catch (NumberFormatException e) {
            throw new IOException(csv + ":" + lineNo + ": bad " + view + " profile sample", e);
        }
        return out;
    }

    /**
     * The observation for a VO frame, carrying the VO frame's timestamp, or {@code null} when the
     * file has no row for that frame.
     */
    @Nullable
    public SkylineObservation lookup(int frameIndex, double voTimestampS) {
        Row r = rows.get(frameIndex);
        if (r == null) {
            return null;
        }
        return new SkylineObservation(frameIndex, voTimestampS, r.north, r.west, r.syncDtS);
    }

    /** Rows whose frame index is outside {@code [0, frameCount)} — a misaligned feed. */
    public List<Integer> framesOutside(int frameCount) {
        List<Integer> out = new ArrayList<>();
        for (int f : rows.keySet()) {
            if (f < 0 || f >= frameCount) {
                out.add(f);
            }
        }
        Collections.sort(out);
        return out;
    }

    /** Rows whose frame index is inside {@code [0, frameCount)}. */
    public int matchedCount(int frameCount) {
        return rows.size() - framesOutside(frameCount).size();
    }

    public int size() {
        return rows.size();
    }

    /** Rows whose North view is valid. */
    public int validCount() {
        return validCount;
    }

    public int westAvailableCount() {
        return westAvailableCount;
    }

    public int westValidCount() {
        return westValidCount;
    }

    /** Rows turned invalid because their time-pairing residual exceeded the tolerance. */
    public int syncRejectedCount() {
        return syncRejectedCount;
    }

    /** Largest |sync_dt_s| in the file (0 when no row was paired by time). */
    public double maxAbsSyncDtS() {
        return maxAbsSyncDtS;
    }

    public int schemaVersion() {
        return schemaVersion;
    }

    /**
     * How {@code frame_index} was obtained: {@link #IDENTITY_EXACT}, {@link #IDENTITY_APPROXIMATE},
     * {@link #IDENTITY_DECLARED} or {@link #IDENTITY_UNKNOWN} (no manifest).
     */
    public String pairingIdentity() {
        return pairingIdentity;
    }

    /** The exporter's pairing mechanism name, or {@code "unknown"}. */
    public String pairingMechanism() {
        return pairingMechanism;
    }

    /** True when every row is the VO frame the simulator itself synchronised the capture with. */
    public boolean exactCaptureIdentity() {
        return IDENTITY_EXACT.equals(pairingIdentity);
    }

    public Path path() {
        return path;
    }

    @Nullable
    public Map<String, Object> manifest() {
        return manifest;
    }
}
