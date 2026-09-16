package org.boofcv.stitching.diagnostics;

import boofcv.abst.sfm.AccessPointTracks;
import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.struct.image.ImageBase;
import georegression.struct.InvertibleTransform;
import georegression.struct.point.Point2D_F64;

import java.util.ArrayList;
import java.util.List;

/**
 * Experiment-side timing decorator around an {@link ImageMotion2D}, added for {@code EXP-VO-002}.
 *
 * <p>Splits the stitching VO's per-frame cost into its two halves so an affine-vs-homography
 * comparison can say <em>where</em> a difference comes from:
 *
 * <ul>
 *   <li>the run record's existing {@code process_time_ns} wraps
 *       {@code StitchingEstimator.processFrame} — KLT tracking + RANSAC model estimation + mosaic
 *       recentre/render + pose update, excluding image load and JPEG decode;</li>
 *   <li>{@link #frameNanos()} here wraps only {@code ImageMotion2D.process} — KLT tracking +
 *       RANSAC model estimation, excluding all mosaic rendering.</li>
 * </ul>
 *
 * <p>The difference of the two is the mosaic/render/pose share. Only the motion half differs
 * between the two models, so this is the split that isolates the effect under study.
 *
 * <p><b>It does not change estimator behaviour.</b> Every method delegates; {@code process} adds
 * exactly two {@link System#nanoTime()} calls around the delegate and returns its result unaltered.
 * The overhead is identical in both arms of the comparison and therefore cancels in the
 * relative-cost figure, which is the figure {@code EXP-VO-002} actually reports.
 *
 * <p>{@link AccessPointTracks} is delegated as well, deliberately: {@code VoDiagnostics} discovers
 * per-frame track and inlier counts by casting the motion object to that interface, and a decorator
 * that did not implement it would silently downgrade those counts to "unavailable" — turning a
 * measurement the experiment needs into an absence, in exactly the way {@code DEC-VO-001} warns
 * against. The delegate is required to implement it; construction fails loudly otherwise rather
 * than degrading quietly.
 *
 * @param <I>  image type
 * @param <IT> motion model type
 */
public final class MotionTimingProbe<I extends ImageBase<I>, IT extends InvertibleTransform<IT>>
        implements ImageMotion2D<I, IT>, AccessPointTracks {

    private final ImageMotion2D<I, IT> delegate;
    private final AccessPointTracks tracks;
    private final List<Long> nanos = new ArrayList<>();

    public MotionTimingProbe(ImageMotion2D<I, IT> delegate) {
        this.delegate = delegate;
        if (!(delegate instanceof AccessPointTracks t)) {
            throw new IllegalArgumentException(
                    "Delegate must implement AccessPointTracks so track/inlier counts survive the "
                            + "decorator; got " + delegate.getClass().getName());
        }
        this.tracks = t;
    }

    /** Per-frame motion-estimation time in nanoseconds, in processing order. */
    public List<Long> frameNanos() {
        return List.copyOf(nanos);
    }

    public void clearTimings() {
        nanos.clear();
    }

    // --- ImageMotion2D: process is timed, everything else is pass-through ---

    @Override
    public boolean process(I input) {
        long start = System.nanoTime();
        boolean result = delegate.process(input);
        nanos.add(System.nanoTime() - start);
        return result;
    }

    @Override public void reset() { delegate.reset(); }
    @Override public void setToFirst() { delegate.setToFirst(); }
    @Override public long getFrameID() { return delegate.getFrameID(); }
    @Override public IT getFirstToCurrent() { return delegate.getFirstToCurrent(); }
    @Override public Class<IT> getTransformType() { return delegate.getTransformType(); }

    // --- AccessPointTracks: pure delegation ---

    @Override public int getTotalTracks() { return tracks.getTotalTracks(); }
    @Override public long getTrackId(int index) { return tracks.getTrackId(index); }
    @Override public void getTrackPixel(int index, Point2D_F64 pixel) { tracks.getTrackPixel(index, pixel); }
    @Override public boolean isTrackInlier(int index) { return tracks.isTrackInlier(index); }
    @Override public boolean isTrackNew(int index) { return tracks.isTrackNew(index); }

    @Override
    @SuppressWarnings("deprecation")
    public List<Point2D_F64> getAllTracks(List<Point2D_F64> storage) {
        return tracks.getAllTracks(storage);
    }
}
