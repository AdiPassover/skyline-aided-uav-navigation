package org.boofcv.evaluation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Declared-image-dimension parsing (EXP-CONF-001 P3 integration fix, 2026-08-30). The dataset
 * contract's own example places {@code image_width}/{@code image_height} inside {@code metadata}
 * (contracts/dataset.md), and every real MARS-LVIG descriptor follows it; the P2 implementation
 * only read the top-level spelling, so the A3 binding check would have refused every real capture
 * for a "missing" resolution. Top-level remains preferred; {@code metadata.*} is the fallback.
 */
public class DatasetDescriptorTest {

    private static DatasetDescriptor load(Path root, String json) throws IOException {
        Files.writeString(root.resolve("dataset.json"), json, StandardCharsets.UTF_8);
        Files.writeString(root.resolve("frames.csv"),
                "frame_index,timestamp_s,image_path\n0,0.000,images/frame_000000.png\n",
                StandardCharsets.UTF_8);
        return DatasetDescriptor.load(root);
    }

    @Test
    public void topLevelDimensionsAreRead(@TempDir Path tmp) throws IOException {
        DatasetDescriptor d = load(tmp, "{\"schema_version\":\"1.0.0\",\"dataset_id\":\"t\","
                + "\"dataset_revision\":\"v1\",\"image_width\":2448,\"image_height\":2048}");
        assertEquals(2448, d.imageWidth);
        assertEquals(2048, d.imageHeight);
    }

    @Test
    public void metadataDimensionsAreTheFallback(@TempDir Path tmp) throws IOException {
        DatasetDescriptor d = load(tmp, "{\"schema_version\":\"1.0.0\",\"dataset_id\":\"t\","
                + "\"dataset_revision\":\"v1\",\"metadata\":{\"image_width\":2448,"
                + "\"image_height\":2048,\"frame_rate_hz\":10.0}}");
        assertEquals(2448, d.imageWidth);
        assertEquals(2048, d.imageHeight);
    }

    @Test
    public void topLevelWinsOverMetadata(@TempDir Path tmp) throws IOException {
        DatasetDescriptor d = load(tmp, "{\"schema_version\":\"1.0.0\",\"dataset_id\":\"t\","
                + "\"dataset_revision\":\"v1\",\"image_width\":100,\"image_height\":50,"
                + "\"metadata\":{\"image_width\":2448,\"image_height\":2048}}");
        assertEquals(100, d.imageWidth);
        assertEquals(50, d.imageHeight);
    }

    @Test
    public void absentDimensionsStayNull(@TempDir Path tmp) throws IOException {
        DatasetDescriptor d = load(tmp, "{\"schema_version\":\"1.0.0\",\"dataset_id\":\"t\","
                + "\"dataset_revision\":\"v1\",\"metadata\":{\"frame_rate_hz\":10.0}}");
        assertNull(d.imageWidth);
        assertNull(d.imageHeight);
    }
}
