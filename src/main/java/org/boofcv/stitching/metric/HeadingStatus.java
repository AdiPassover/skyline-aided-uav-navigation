package org.boofcv.stitching.metric;

/**
 * Freshness of the authoritative navigation heading a frame was rotated with — the heading channel's
 * exact counterpart to {@link HeightStatus}, and deliberately the same four states so a consumer
 * reads both the same way.
 *
 * <p>There is one asymmetry with height, and it is a physical one rather than a design choice. A
 * takeoff-relative height channel reads <b>zero at its own datum by construction</b>, so before its
 * first sample the runtime still knows {@code h_AGL = h0}. Heading has no such datum: an absolute
 * azimuth cannot be derived from the absence of a measurement. So before the first accepted heading
 * sample this channel reports {@link #UNAVAILABLE}, not {@code HELD} — see
 * {@link CausalHeadingChannel}.
 */
public enum HeadingStatus {

    /**
     * A sample arrived within {@code 1.5x} the declared nominal interval. On the only real-flight
     * channel measured ({@code /dji_osdk_ros/attitude} at 101.3 Hz against 10 Hz imagery) this is
     * every frame of every window.
     */
    FRESH,

    /**
     * Older than {@code 1.5x} the nominal interval but within {@code tau_stale}: held, used, and
     * <b>not</b> degraded. Exactly {@link HeightStatus#HELD}'s meaning.
     */
    HELD,

    /**
     * Older than {@code tau_stale}. The value is <b>still held and still used</b> — the alternative
     * is falling back to the visually integrated yaw, which is the quantity being replaced — but the
     * frame is flagged degraded. As with height, {@code tau_stale} therefore changes what is
     * <i>reported</i>, never what is <i>computed</i>.
     */
    STALE,

    /**
     * <b>No authoritative heading exists at all</b>: no sample has ever been accepted, or the value
     * is not finite. No navigation heading is produced for this frame, and none is invented — in
     * particular the visually integrated yaw is <b>not</b> silently promoted to fill the gap.
     */
    UNAVAILABLE;

    /** True when a navigation heading may be produced from this sample (everything but {@link #UNAVAILABLE}). */
    public boolean usable() {
        return this != UNAVAILABLE;
    }

    /** True when the resulting navigation pose must be reported degraded — {@link #STALE} or {@link #UNAVAILABLE}. */
    public boolean degraded() {
        return this == STALE || this == UNAVAILABLE;
    }
}
