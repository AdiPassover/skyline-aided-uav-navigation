package org.boofcv.evaluation;

import boofcv.abst.sfm.AccessPointTracks;
import boofcv.abst.sfm.d2.ImageMotion2D;

/**
 * Extracts per-frame track and inlier counts from the estimator's {@code getMotion()} object by
 * casting to BoofCV's {@link AccessPointTracks}, per research R5. No change to the estimator: the
 * cast is confined here, and degrades to empty values rather than throwing when it fails, since
 * absence of this diagnostic must never look like a measurement of zero.
 */
public final class VoDiagnostics {

    public record Counts(Integer trackCount, Integer inlierCount) {
        public static final Counts UNAVAILABLE = new Counts(null, null);
    }

    /**
     * {@link Counts} plus the frozen inlier spatial-coverage statistic ({@code EXP-CONF-001} P1):
     * the frame is divided into a fixed {@code 8 x 8} grid, a cell is occupied when at least one
     * inlier track's pixel lies in it, and coverage is {@code occupiedCells / 64.0}. Absent
     * ({@code null}) when fewer than {@link #COVERAGE_MIN_INLIERS} inliers exist — two points
     * "covering" a frame is not a measurement of spread.
     *
     * <p>Computed here rather than in {@code org.boofcv.confidence} because this package is where
     * the {@link AccessPointTracks} cast already lives, and the confidence package's own charter
     * forbids BoofCV imports (its {@code package-info}). {@code SignalExtractor} receives the
     * computed value. This is a recorded deviation from tasks.md T021's letter, honouring T001.
     */
    public record Diagnostics(Integer trackCount, Integer inlierCount, Double inlierCoverage) {
        public static final Diagnostics UNAVAILABLE = new Diagnostics(null, null, null);

        public Counts counts() {
            return new Counts(trackCount, inlierCount);
        }
    }

    /** Grid dimension of the frozen coverage definition: {@code 8 x 8} cells. */
    public static final int COVERAGE_GRID = 8;

    /** Below this many inliers the coverage statistic is absent, never zero. */
    public static final int COVERAGE_MIN_INLIERS = 3;

    private VoDiagnostics() {
    }

    public static Counts extract(ImageMotion2D<?, ?> motion) {
        if (!(motion instanceof AccessPointTracks tracks)) {
            return Counts.UNAVAILABLE;
        }
        int total = tracks.getTotalTracks();
        int inliers = 0;
        for (int i = 0; i < total; i++) {
            if (tracks.isTrackInlier(i)) {
                inliers++;
            }
        }
        return new Counts(total, inliers);
    }

    /**
     * As {@link #extract}, additionally computing inlier spatial coverage over the processed
     * frame's pixel grid ({@code width x height} — post-downsample, the grid every other per-frame
     * pixel quantity uses). Degrades to absent, never throws, matching {@link #extract}'s rule
     * that absence of a diagnostic must never look like a measurement of zero.
     */
    public static Diagnostics extractWithCoverage(ImageMotion2D<?, ?> motion, int width, int height) {
        if (!(motion instanceof AccessPointTracks tracks) || width <= 0 || height <= 0) {
            return Diagnostics.UNAVAILABLE;
        }
        java.util.List<georegression.struct.point.Point2D_F64> pixels = tracks.getAllTracks(null);
        int total = tracks.getTotalTracks();
        if (pixels.size() != total) {
            // The two views of the track set disagree; refuse to guess which is right.
            return new Diagnostics(total, countInliers(tracks, total), null);
        }
        boolean[] occupied = new boolean[COVERAGE_GRID * COVERAGE_GRID];
        int inliers = 0;
        for (int i = 0; i < total; i++) {
            if (!tracks.isTrackInlier(i)) {
                continue;
            }
            inliers++;
            georegression.struct.point.Point2D_F64 p = pixels.get(i);
            int cx = Math.min(COVERAGE_GRID - 1, Math.max(0, (int) (p.x * COVERAGE_GRID / width)));
            int cy = Math.min(COVERAGE_GRID - 1, Math.max(0, (int) (p.y * COVERAGE_GRID / height)));
            occupied[cy * COVERAGE_GRID + cx] = true;
        }
        Double coverage = null;
        if (inliers >= COVERAGE_MIN_INLIERS) {
            int cells = 0;
            for (boolean b : occupied) {
                if (b) {
                    cells++;
                }
            }
            coverage = cells / (double) (COVERAGE_GRID * COVERAGE_GRID);
        }
        return new Diagnostics(total, inliers, coverage);
    }

    private static Integer countInliers(AccessPointTracks tracks, int total) {
        int inliers = 0;
        for (int i = 0; i < total; i++) {
            if (tracks.isTrackInlier(i)) {
                inliers++;
            }
        }
        return inliers;
    }
}
