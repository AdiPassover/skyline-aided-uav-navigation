package org.boofcv.evaluation;

import boofcv.struct.image.GrayF32;

/**
 * Abstraction over frame supply for {@link VoRunner}, so the estimator driver does not depend on
 * how frames are stored (plan.md Project Structure).
 */
public interface FrameSource {

    /** One timestamped frame, in the dataset's frame_index order. */
    record TimestampedFrame(int frameIndex, double timestampS, GrayF32 image) {
    }

    int frameCount();

    /** @param index position in [0, frameCount()), not necessarily equal to frame_index */
    TimestampedFrame frame(int index);
}
