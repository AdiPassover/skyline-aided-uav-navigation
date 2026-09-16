package org.boofcv.evaluation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Covers the {@code downsampleFactor} added 2026-08-14 while preparing EXP-002 Execution A:
 * discovered that no downsample step existed anywhere in the capture pipeline, even though
 * EXP-002's pre-registered Configuration specifies working at 1224 x 1024 (an exact 2x2 box
 * average of the native 2448 x 2048) -- fulfilling an already-specified design point, not
 * changing one, and done before any VO run on the real dataset.
 */
public class DirectoryFrameSourceTest {

    private static void writeGrayPng(Path path, int width, int height, int[] pixels) throws IOException {
        // setSample() writes the raw raster value directly, bypassing setRGB()'s implicit
        // color-model conversion (which distorts TYPE_BYTE_GRAY values via gamma/luminance
        // weighting and silently broke this test's expected block averages initially).
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img.getRaster().setSample(x, y, 0, pixels[y * width + x]);
            }
        }
        ImageIO.write(img, "png", path.toFile());
    }

    private static DatasetDescriptor writeMinimalDataset(Path root, int width, int height, int[] pixels) throws IOException {
        Files.createDirectories(root.resolve("images"));
        writeGrayPng(root.resolve("images/frame_000000.png"), width, height, pixels);

        String datasetJson = "{\"schema_version\":\"1.0.0\",\"dataset_id\":\"downsample-test\",\"dataset_revision\":\"v1\"}";
        Files.writeString(root.resolve("dataset.json"), datasetJson, StandardCharsets.UTF_8);
        Files.writeString(root.resolve("frames.csv"),
                "frame_index,timestamp_s,image_path\n0,0.000,images/frame_000000.png\n", StandardCharsets.UTF_8);
        return DatasetDescriptor.load(root);
    }

    @Test
    public void factorOneIsANoOp(@TempDir Path tmp) throws IOException {
        int[] px = {10, 20, 30, 40};
        DatasetDescriptor dataset = writeMinimalDataset(tmp, 2, 2, px);
        DirectoryFrameSource src = new DirectoryFrameSource(dataset, 1);

        FrameSource.TimestampedFrame frame = src.frame(0);
        assertEquals(2, frame.image().width);
        assertEquals(2, frame.image().height);
    }

    @Test
    public void factorTwoProducesExactBlockAverages(@TempDir Path tmp) throws IOException {
        // 4x4 image, four distinct 2x2 quadrants -> exact known averages after downsample.
        int[] px = {
                0, 0, 100, 100,
                0, 0, 100, 100,
                50, 50, 200, 200,
                50, 50, 200, 200,
        };
        DatasetDescriptor dataset = writeMinimalDataset(tmp, 4, 4, px);
        DirectoryFrameSource src = new DirectoryFrameSource(dataset, 2);

        FrameSource.TimestampedFrame frame = src.frame(0);
        assertEquals(2, frame.image().width);
        assertEquals(2, frame.image().height);
        assertEquals(0f, frame.image().get(0, 0), 1e-3f);
        assertEquals(100f, frame.image().get(1, 0), 1e-3f);
        assertEquals(50f, frame.image().get(0, 1), 1e-3f);
        assertEquals(200f, frame.image().get(1, 1), 1e-3f);
    }

    @Test
    public void nonDivisibleFactorRaisesRatherThanInterpolating(@TempDir Path tmp) throws IOException {
        int[] px = new int[3 * 3];
        DatasetDescriptor dataset = writeMinimalDataset(tmp, 3, 3, px);
        DirectoryFrameSource src = new DirectoryFrameSource(dataset, 2);

        assertThrows(IllegalStateException.class, () -> src.frame(0));
    }

    @Test
    public void constructorRejectsFactorBelowOne(@TempDir Path tmp) throws IOException {
        int[] px = new int[4];
        DatasetDescriptor dataset = writeMinimalDataset(tmp, 2, 2, px);
        assertThrows(IllegalArgumentException.class, () -> new DirectoryFrameSource(dataset, 0));
    }
}
