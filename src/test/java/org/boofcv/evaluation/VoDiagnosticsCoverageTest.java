package org.boofcv.evaluation;

import boofcv.abst.sfm.AccessPointTracks;
import boofcv.abst.sfm.d2.ImageMotion2D;
import boofcv.struct.image.GrayF32;
import georegression.struct.homography.Homography2D_F64;
import georegression.struct.point.Point2D_F64;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * The frozen inlier spatial-coverage definition ({@code EXP-CONF-001} P1): an 8 × 8 grid over the
 * processed frame, a cell occupied by ≥ 1 inlier pixel, coverage = occupied / 64.0; absent below
 * 3 inliers, never zero.
 *
 * <p>Evidence tier: <b>T1 (analytical)</b>.
 */
public class VoDiagnosticsCoverageTest {

    /** A motion stub carrying a scripted track set. Implements both interfaces, as the real wrapper does. */
    private static final class StubMotion implements ImageMotion2D<GrayF32, Homography2D_F64>, AccessPointTracks {
        final List<Point2D_F64> pixels = new ArrayList<>();
        final List<Boolean> inlier = new ArrayList<>();

        void track(double x, double y, boolean isInlier) {
            pixels.add(new Point2D_F64(x, y));
            inlier.add(isInlier);
        }

        @Override public int getTotalTracks() { return pixels.size(); }
        @Override public long getTrackId(int index) { return index; }
        @Override public void getTrackPixel(int index, Point2D_F64 pixel) { pixel.setTo(pixels.get(index)); }
        @Override public List<Point2D_F64> getAllTracks(List<Point2D_F64> storage) {
            List<Point2D_F64> out = storage != null ? storage : new ArrayList<>();
            out.clear();
            out.addAll(pixels);
            return out;
        }
        @Override public boolean isTrackInlier(int index) { return inlier.get(index); }
        @Override public boolean isTrackNew(int index) { return false; }

        @Override public boolean process(GrayF32 input) { return true; }
        @Override public void reset() { }
        @Override public void setToFirst() { }
        @Override public long getFrameID() { return 0; }
        @Override public Homography2D_F64 getFirstToCurrent() { return new Homography2D_F64(); }
        @Override public Class<Homography2D_F64> getTransformType() { return Homography2D_F64.class; }
    }

    @Test
    public void coverageCountsOccupiedGridCellsOverInliersOnly() {
        StubMotion m = new StubMotion();
        // 640x512 frame; cell size 80x64. Three inliers in three distinct cells, one outlier in a
        // fourth cell that must not count.
        m.track(10, 10, true);     // cell (0,0)
        m.track(90, 10, true);     // cell (1,0)
        m.track(10, 70, true);     // cell (0,1)
        m.track(300, 300, false);  // outlier — ignored

        VoDiagnostics.Diagnostics d = VoDiagnostics.extractWithCoverage(m, 640, 512);
        assertEquals(4, d.trackCount());
        assertEquals(3, d.inlierCount());
        assertEquals(3 / 64.0, d.inlierCoverage(), 0.0);
    }

    @Test
    public void twoInliersInOneCellOccupyOneCell() {
        StubMotion m = new StubMotion();
        m.track(1, 1, true);
        m.track(2, 2, true);       // same cell (0,0)
        m.track(639, 511, true);   // cell (7,7) — the far corner clamps into the last cell

        VoDiagnostics.Diagnostics d = VoDiagnostics.extractWithCoverage(m, 640, 512);
        assertEquals(2 / 64.0, d.inlierCoverage(), 0.0);
    }

    @Test
    public void coverageIsAbsentBelowThreeInliersNeverZero() {
        StubMotion m = new StubMotion();
        m.track(10, 10, true);
        m.track(600, 500, true);
        m.track(300, 300, false);

        VoDiagnostics.Diagnostics d = VoDiagnostics.extractWithCoverage(m, 640, 512);
        assertEquals(2, d.inlierCount());
        assertNull(d.inlierCoverage(), "two points spanning a frame is not a measurement of spread");
    }

    @Test
    public void aNonTrackingMotionDegradesToAbsent() {
        ImageMotion2D<GrayF32, Homography2D_F64> plain = new ImageMotion2D<>() {
            @Override public boolean process(GrayF32 input) { return true; }
            @Override public void reset() { }
            @Override public void setToFirst() { }
            @Override public long getFrameID() { return 0; }
            @Override public Homography2D_F64 getFirstToCurrent() { return new Homography2D_F64(); }
            @Override public Class<Homography2D_F64> getTransformType() { return Homography2D_F64.class; }
        };
        VoDiagnostics.Diagnostics d = VoDiagnostics.extractWithCoverage(plain, 640, 512);
        assertNull(d.trackCount());
        assertNull(d.inlierCount());
        assertNull(d.inlierCoverage());
    }

    @Test
    public void fullSpreadReachesOne() {
        StubMotion m = new StubMotion();
        // One inlier per cell: 64 inliers at each cell's centre.
        for (int cy = 0; cy < 8; cy++) {
            for (int cx = 0; cx < 8; cx++) {
                m.track(cx * 80 + 40, cy * 64 + 32, true);
            }
        }
        VoDiagnostics.Diagnostics d = VoDiagnostics.extractWithCoverage(m, 640, 512);
        assertEquals(1.0, d.inlierCoverage(), 0.0);
    }
}
