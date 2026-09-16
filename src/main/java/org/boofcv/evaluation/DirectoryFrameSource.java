package org.boofcv.evaluation;

import boofcv.alg.filter.misc.AverageDownSampleOps;
import boofcv.io.image.UtilImageIO;
import boofcv.struct.image.GrayF32;

import java.io.IOException;
import java.io.UncheckedIOException;

/** Supplies frames from a dataset directory in {@code frame_index} order. */
public final class DirectoryFrameSource implements FrameSource {

    private final DatasetDescriptor dataset;
    private final int downsampleFactor;

    public DirectoryFrameSource(DatasetDescriptor dataset) {
        this(dataset, 1);
    }

    /**
     * @param downsampleFactor exact NxN box-average downsample applied to every frame before
     *                         the estimator sees it (1 = no-op). Chosen over any interpolating
     *                         resize because an exact block average introduces no resampling
     *                         artifact distinct from a genuine estimator effect -- the same
     *                         reasoning EXP-002's Configuration section records for why
     *                         2448x2048 was downsampled to 1224x1024 rather than to 640x512.
     */
    public DirectoryFrameSource(DatasetDescriptor dataset, int downsampleFactor) {
        if (downsampleFactor < 1) {
            throw new IllegalArgumentException("downsampleFactor must be >= 1, got " + downsampleFactor);
        }
        this.dataset = dataset;
        this.downsampleFactor = downsampleFactor;
    }

    @Override
    public int frameCount() {
        return dataset.frames.size();
    }

    @Override
    public TimestampedFrame frame(int index) {
        DatasetDescriptor.FrameRef ref = dataset.frames.get(index);
        String path = dataset.root.resolve(ref.imagePath).toString();
        GrayF32 image = UtilImageIO.loadImage(path, GrayF32.class);
        if (image == null) {
            throw new UncheckedIOException(new IOException("Failed to load image: " + path));
        }
        if (downsampleFactor > 1) {
            if (image.width % downsampleFactor != 0 || image.height % downsampleFactor != 0) {
                throw new IllegalStateException(
                        "downsampleFactor " + downsampleFactor + " does not exactly divide frame "
                                + index + "'s size " + image.width + "x" + image.height
                                + " -- refusing to resample with interpolation instead of an exact box average");
            }
            GrayF32 down = new GrayF32(image.width / downsampleFactor, image.height / downsampleFactor);
            AverageDownSampleOps.down(image, downsampleFactor, down);
            image = down;
        }
        return new TimestampedFrame(ref.frameIndex, ref.timestampS, image);
    }
}
