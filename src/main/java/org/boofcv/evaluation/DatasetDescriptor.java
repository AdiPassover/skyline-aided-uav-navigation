package org.boofcv.evaluation;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads {@code dataset.json} and {@code frames.csv} per contracts/dataset.md. Only the fields the
 * Java capture side needs to locate and order frames are parsed here; ground-truth handling and
 * coordinate conversion are Python-side concerns (plan.md responsibility split).
 */
public final class DatasetDescriptor {

    private static final int SUPPORTED_SCHEMA_MAJOR = 1;

    public final String schemaVersion;
    public final String datasetId;
    public final String datasetRevision;
    public final Path root;
    public final List<FrameRef> frames;
    /** Declared native image width from {@code dataset.json}, or {@code null} if not declared. */
    public final Integer imageWidth;
    /** Declared native image height from {@code dataset.json}, or {@code null} if not declared. */
    public final Integer imageHeight;

    public static final class FrameRef {
        public final int frameIndex;
        public final double timestampS;
        public final String imagePath;

        public FrameRef(int frameIndex, double timestampS, String imagePath) {
            this.frameIndex = frameIndex;
            this.timestampS = timestampS;
            this.imagePath = imagePath;
        }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    private static final class DatasetJson {
        public String schema_version;
        public String dataset_id;
        public String dataset_revision;
        public Integer image_width;
        public Integer image_height;
        // The dataset contract's own example places image_width/image_height inside `metadata`
        // (contracts/dataset.md), and every real MARS-LVIG descriptor does so; the top-level
        // fields are kept as the preferred spelling, with `metadata.*` as the fallback.
        public java.util.Map<String, Object> metadata;
    }

    private static Integer declaredDimension(DatasetJson json, Integer topLevel, String key) {
        if (topLevel != null) {
            return topLevel;
        }
        if (json.metadata != null && json.metadata.get(key) instanceof Number n) {
            return n.intValue();
        }
        return null;
    }

    private DatasetDescriptor(String schemaVersion, String datasetId, String datasetRevision,
                               Path root, List<FrameRef> frames,
                               Integer imageWidth, Integer imageHeight) {
        this.schemaVersion = schemaVersion;
        this.datasetId = datasetId;
        this.datasetRevision = datasetRevision;
        this.root = root;
        this.frames = frames;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
    }

    public static DatasetDescriptor load(Path datasetRoot) throws IOException {
        Path jsonPath = datasetRoot.resolve("dataset.json");
        Path framesPath = datasetRoot.resolve("frames.csv");
        if (!Files.exists(jsonPath)) {
            throw new IOException("Missing dataset.json in " + datasetRoot);
        }
        if (!Files.exists(framesPath)) {
            throw new IOException("Missing frames.csv in " + datasetRoot);
        }

        DatasetJson json = new ObjectMapper().readValue(jsonPath.toFile(), DatasetJson.class);
        int major = majorVersion(json.schema_version);
        if (major != SUPPORTED_SCHEMA_MAJOR) {
            throw new IOException(
                    "Unsupported dataset schema major version: " + json.schema_version
                            + " (this reader supports major version " + SUPPORTED_SCHEMA_MAJOR + ")");
        }

        List<FrameRef> frames = readFrames(framesPath);

        return new DatasetDescriptor(json.schema_version, json.dataset_id, json.dataset_revision,
                datasetRoot, frames,
                declaredDimension(json, json.image_width, "image_width"),
                declaredDimension(json, json.image_height, "image_height"));
    }

    private static int majorVersion(String schemaVersion) throws IOException {
        if (schemaVersion == null) {
            throw new IOException("dataset.json is missing schema_version");
        }
        try {
            return Integer.parseInt(schemaVersion.split("\\.", 2)[0]);
        } catch (NumberFormatException e) {
            throw new IOException("Malformed schema_version: " + schemaVersion, e);
        }
    }

    private static List<FrameRef> readFrames(Path framesPath) throws IOException {
        List<String> lines = Files.readAllLines(framesPath, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            throw new IOException("frames.csv has no header row: " + framesPath);
        }
        List<FrameRef> frames = new ArrayList<>();
        int lastIndex = -1;
        double lastTimestamp = Double.NEGATIVE_INFINITY;
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            if (line.isEmpty()) {
                continue;
            }
            String[] cols = line.split(",", 3);
            if (cols.length != 3) {
                throw new IOException("Malformed frames.csv row " + i + ": " + line);
            }
            int frameIndex = Integer.parseInt(cols[0]);
            double timestampS = Double.parseDouble(cols[1]);
            String imagePath = cols[2];

            if (frameIndex != lastIndex + 1) {
                throw new IOException(
                        "frames.csv frame_index must be 0-based and contiguous: expected " + (lastIndex + 1)
                                + " but got " + frameIndex);
            }
            if (timestampS <= lastTimestamp) {
                throw new IOException("frames.csv timestamp_s must be strictly increasing at row " + i);
            }
            lastIndex = frameIndex;
            lastTimestamp = timestampS;

            frames.add(new FrameRef(frameIndex, timestampS, imagePath));
        }
        if (frames.isEmpty()) {
            throw new IOException("frames.csv has zero frames: " + framesPath);
        }
        return frames;
    }
}
