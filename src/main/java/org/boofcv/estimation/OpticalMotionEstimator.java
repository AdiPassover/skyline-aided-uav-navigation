package org.boofcv.estimation;

import boofcv.struct.image.ImageBase;

/**
 * Interface for motion estimation algorithms that process sequence of images (optical navigation).
 *
 * @param <T> The type of the input image frame.
 */
public interface OpticalMotionEstimator<T extends ImageBase<T>> extends MotionEstimator {

    /**
     * Initializes the motion estimator. Must be called before processing frames.
     */
    void init();

    /**
     * Processes a new video frame to update the motion/pose estimate.
     *
     * @param frame The input image frame.
     * @return true if stitching/estimation succeeded, false otherwise.
     */
    boolean processFrame(T frame);

    /**
     * Processes a new video frame that carries a capture time.
     *
     * <p>Added for the metric local readout ({@code DEC-VO-007} D5), whose height channel is a
     * causal zero-order hold and therefore needs to know <i>which instant</i> a frame belongs to.
     * The default ignores the timestamp, so every existing implementation and every existing call
     * site is unaffected; an implementation that consumes time overrides it, and should refuse the
     * timestamp-less {@link #processFrame(ImageBase)} rather than guess.
     *
     * @param timestampS capture time in seconds, on the same clock the height source uses
     * @return true if stitching/estimation succeeded, false otherwise
     */
    default boolean processFrame(T frame, double timestampS) {
        return processFrame(frame);
    }

    /**
     * Resets the motion estimator to its initial state.
     */
    void reset();

}
