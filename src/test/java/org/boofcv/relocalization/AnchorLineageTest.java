package org.boofcv.relocalization;

import org.boofcv.stitching.metric.HeadingStatus;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The non-decaying ledger of design §4 / §9.1. T1. */
public class AnchorLineageTest {

    private static TrustedReference reference(int id, long baselineExposure, boolean baselineInstability) {
        return new TrustedReference(id, 500, 50.0, TestDescriptors.sine(0.1), null, new PlanarPosition(1, 2), 3.0,
                HeadingStatus.FRESH, 0, 0, null, null, baselineExposure, baselineInstability,
                TrustedReference.POSE_SCHEMA);
    }

    @Test
    public void rootConvention() {
        AnchorLineage l = new AnchorLineage(0, 0.0);
        assertEquals(0, l.exposureAnchor());
        assertEquals(0, l.exposureSince());
        assertEquals(0, l.effectiveExposure());
        assertFalse(l.instabilityAnchor());
        assertFalse(l.instabilitySince());
        assertFalse(l.effectiveInstability());
        assertNull(l.anchorReferenceId());
    }

    @Test
    public void exposureNeverDecays() {
        AnchorLineage l = new AnchorLineage(0, 0.0);
        for (int i = 0; i < 1000; i++) {
            l.incrementExposure();
            assertEquals(i + 1, l.effectiveExposure(), "monotone at increment " + i);
        }
        assertEquals(1000, l.exposureSince());
        assertEquals(1000.0, l.timeSinceAnchorS(1000.0), 0.0);
    }

    @Test
    public void instabilityIsStickyThroughLaterStableIncrements() {
        AnchorLineage l = new AnchorLineage(0, 0.0);
        l.incrementExposure();
        l.noteInstability();
        for (int i = 0; i < 500; i++) {
            l.incrementExposure();
            assertTrue(l.instabilitySince(), "still sticky at " + i);
            assertTrue(l.effectiveInstability());
        }
    }

    @Test
    public void inheritanceUsesTheReferencesStoredBaselineNotZero() {
        AnchorLineage l = new AnchorLineage(0, 0.0);
        for (int i = 0; i < 120; i++) {
            l.incrementExposure();
        }
        l.noteInstability();

        l.inherit(reference(4, 70, false), 200, 20.0);
        assertEquals(70, l.exposureAnchor(), "E_anchor ← E_i_at_storage");
        assertEquals(0, l.exposureSince(), "E_since ← 0");
        assertEquals(70, l.effectiveExposure(), "not zero: the reference's own history is kept");
        assertFalse(l.instabilityAnchor());
        assertFalse(l.instabilitySince(), "I_since cleared only by an accepted anchor");
        assertEquals(4, l.anchorReferenceId());
        assertEquals(200, l.anchorFrameIndex());

        for (int i = 0; i < 20; i++) {
            l.incrementExposure();
        }
        assertEquals(90, l.effectiveExposure(), "the design's worked example: 70 + 20");
    }

    @Test
    public void anAnchorThatCarriedInstabilityPropagatesIt() {
        // The MVP insertion policy never creates such a reference; the ledger must still honour
        // it if one is ever supplied, rather than silently laundering the lineage.
        AnchorLineage l = new AnchorLineage(0, 0.0);
        l.inherit(reference(9, 10, true), 10, 1.0);
        assertTrue(l.instabilityAnchor());
        assertFalse(l.instabilitySince());
        assertTrue(l.effectiveInstability());
    }

    @Test
    public void snapshotIsAnImmutableCopy() {
        AnchorLineage l = new AnchorLineage(0, 0.0);
        l.incrementExposure();
        AnchorLineage.Snapshot s = l.snapshot();
        l.incrementExposure();
        l.noteInstability();
        assertEquals(1, s.effectiveExposure());
        assertFalse(s.effectiveInstability());
        assertEquals(2, l.effectiveExposure());
    }
}
