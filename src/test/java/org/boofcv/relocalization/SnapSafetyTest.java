package org.boofcv.relocalization;

import org.boofcv.stitching.metric.HeadingStatus;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The snap-safety quantities of design §14, in one metric convention (test I). T1. */
public class SnapSafetyTest {

    private static PlanarPosition p(double e, double n) {
        return new PlanarPosition(e, n);
    }

    @Test
    public void aSnapThatMovesTowardsTruthIsNotHarmful() {
        PlanarPosition truth = p(100, 200);
        PlanarPosition before = p(130, 240);   // 50 m off
        PlanarPosition snap = p(103, 204);     // 5 m off
        SnapSafety.Delta d = SnapSafety.evaluate(truth, before, snap);
        assertEquals(50.0, d.positionErrorBeforeM(), 1e-12);
        assertEquals(5.0, d.positionErrorAfterM(), 1e-12);
        assertEquals(-45.0, d.positionDeltaM(), 1e-12);
        assertFalse(d.harmful());
    }

    /** A correct place identity can still be a harmful correction: the design's central caveat. */
    @Test
    public void aSnapOntoAQuantisedReferenceCanBeHarmfulEvenWhenThePlaceIsRight() {
        PlanarPosition truth = p(0, 0);
        PlanarPosition before = p(3, 4);       // VO was 5 m off
        PlanarPosition snap = p(-20, 0);       // nearest stored reference: 20 m away
        SnapSafety.Delta d = SnapSafety.evaluate(truth, before, snap);
        assertEquals(15.0, d.positionDeltaM(), 1e-12);
        assertTrue(d.harmful());
    }

    /**
     * Ground truth (the dataset's own ENU frame, not re-origined) is brought into the persistent
     * frame by subtracting its value at the run's first frame — no rotation (common North datum),
     * no scale (both metres). Same units on both sides, by construction.
     */
    @Test
    public void groundTruthIsBroughtIntoThePersistentFrameByASubtraction() {
        PlanarPosition gt0 = p(-217.144, 125.710);         // a UE world position at frame 0
        PlanarPosition gtK = p(-117.144, 175.710);         // 100 m east, 50 m north of it
        PlanarPosition persistent = SnapSafety.groundTruthToPersistent(gt0, gtK);
        assertEquals(100.0, persistent.eastM(), 1e-9);
        assertEquals(50.0, persistent.northM(), 1e-9);
        // A persistent position from the layer at that frame compares directly, in metres.
        assertEquals(0.0, SnapSafety.positionErrorM(persistent, p(100.0, 50.0)), 1e-9);
        assertEquals(5.0, SnapSafety.positionErrorM(persistent, p(103.0, 54.0)), 1e-12);
        assertTrue(SnapSafety.UNITS.contains("metres"));
    }

    @Test
    public void aReanchorAfterALossHasNoBeforeErrorAndSaysSo() {
        AnchorLineage.Snapshot l = new AnchorLineage(0, 0.0).snapshot();
        AlignmentTransform t = AlignmentTransform.of(5, 5);
        ReanchorEvent afterLoss = new ReanchorEvent(10, 1.0, 0, p(5, 5), p(5, 5), p(0, 0), 10.0, null, p(5, 5),
                null, t, null, l, l, 1, 1, 5, 0.5, null, "discrete_reference");
        assertEquals(Optional.empty(), SnapSafety.evaluate(afterLoss, p(6, 5)));

        ReanchorEvent withBefore = new ReanchorEvent(10, 1.0, 0, p(5, 5), p(5, 5), p(0, 0), 10.0, p(9, 5), p(5, 5),
                AlignmentTransform.of(9, 5), t, t.deltaFrom(AlignmentTransform.of(9, 5)), l, l, 0, 1, 5, 0.5,
                null, "discrete_reference");
        SnapSafety.Delta d = SnapSafety.evaluate(withBefore, p(6, 5)).orElseThrow();
        assertEquals(3.0, d.positionErrorBeforeM(), 1e-12);
        assertEquals(1.0, d.positionErrorAfterM(), 1e-12);
        assertFalse(d.harmful());
        assertEquals(HeadingStatus.FRESH.usable(), true);
    }
}
