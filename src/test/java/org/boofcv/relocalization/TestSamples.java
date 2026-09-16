package org.boofcv.relocalization;

import org.boofcv.stitching.metric.HeadingStatus;
import org.boofcv.stitching.metric.HeightStatus;

/**
 * Synthetic {@link LocalPoseSample}s for the T1 tests: what the metric VO hands over, frame by
 * frame, under the contract of {@code DEC-VO-009}/{@code -010}. Positions are metres East/North
 * relative to the VO segment's origin; headings are degrees clockwise from North.
 */
final class TestSamples {

    private TestSamples() {
    }

    /** A successful frame with a usable metric increment and a fresh authoritative heading. */
    static LocalPoseSample ok(int frame, int segment, double eastM, double northM, double headingDeg) {
        return new LocalPoseSample(frame, frame * 0.1, true, segment, segment > 0, true,
                new PlanarPosition(eastM, northM), true, HeightStatus.FRESH, headingDeg,
                HeadingStatus.FRESH, headingDeg - 3.0);
    }

    /** A successful frame with a STALE (held, still used, degraded) heading. */
    static LocalPoseSample stale(int frame, int segment, double eastM, double northM, double headingDeg) {
        return new LocalPoseSample(frame, frame * 0.1, true, segment, segment > 0, true,
                new PlanarPosition(eastM, northM), true, HeightStatus.FRESH, headingDeg,
                HeadingStatus.STALE, headingDeg - 3.0);
    }

    /**
     * The restart frame of a hard loss: the VO opened segment {@code newSegment}, its segment
     * position is the origin, and the heading channel (independent of the visual failure) reports
     * {@code headingDeg} (or NaN when it had no sample yet).
     */
    static LocalPoseSample loss(int frame, int newSegment, double headingDeg) {
        boolean headingKnown = !Double.isNaN(headingDeg);
        return new LocalPoseSample(frame, frame * 0.1, false, newSegment, true, headingKnown,
                PlanarPosition.ORIGIN, true, HeightStatus.FRESH, headingDeg,
                headingKnown ? HeadingStatus.FRESH : HeadingStatus.UNAVAILABLE, 12.0);
    }

    /**
     * A successful frame on which the VO produced NO metric increment because no authoritative
     * heading existed ({@code HeadingStatus.UNAVAILABLE}): the segment position holds its last
     * value while the aircraft may have moved. The visual yaw is still a number — and must not be
     * promoted.
     */
    static LocalPoseSample noHeading(int frame, int segment, double heldEastM, double heldNorthM) {
        return new LocalPoseSample(frame, frame * 0.1, true, segment, segment > 0, false,
                new PlanarPosition(heldEastM, heldNorthM), false, HeightStatus.FRESH, Double.NaN,
                HeadingStatus.UNAVAILABLE, 41.0);
    }

    /** A successful frame with a valid heading but an UNAVAILABLE height: no metric increment. */
    static LocalPoseSample noHeight(int frame, int segment, double heldEastM, double heldNorthM,
                                    double headingDeg) {
        return new LocalPoseSample(frame, frame * 0.1, true, segment, segment > 0, true,
                new PlanarPosition(heldEastM, heldNorthM), false, HeightStatus.UNAVAILABLE,
                headingDeg, HeadingStatus.FRESH, headingDeg - 3.0);
    }
}
