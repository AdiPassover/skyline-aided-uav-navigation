package org.boofcv.relocalization;

import javax.annotation.Nullable;

/**
 * The non-decaying anchor-lineage ledger (design §4, §9.1):
 *
 * <pre>
 *   E_eff = E_anchor + E_since
 *   I_eff = I_anchor OR I_since
 * </pre>
 *
 * <p>{@code E_since} counts usable VO increments since the active anchor, with elapsed time kept
 * beside it as the maximum-gap companion; {@code E_anchor} is the exposure the anchor itself
 * inherited when it was stored. {@code I_since} is sticky once an instability event is noted;
 * {@code I_anchor} is whatever the accepted reference carried.
 *
 * <p><b>Nothing here decays, and nothing here is a probability, a covariance, a confidence or a
 * pose weight.</b> The only operation that lowers {@code E_eff} or clears {@code I_since} is
 * {@link #inherit} — an accepted trusted anchor, which replaces the baseline with the reference's
 * stored baseline rather than with zero. A hard loss does not touch the ledger: exposure keeps
 * counting on the new segment and instability stays sticky (design §4 table).
 *
 * <p>Exposure is a heuristic provenance count for scheduling and reference curation. It must not
 * be reported as metres of error.
 */
public final class AnchorLineage {

    private long exposureAnchor;
    private long exposureSince;
    private boolean instabilityAnchor;
    private boolean instabilitySince;
    @Nullable private Integer anchorReferenceId;
    private int anchorFrameIndex;
    private double anchorTimestampS;

    /** Root convention: {@code E_anchor = 0}, {@code I_anchor = false}, no anchor reference. */
    public AnchorLineage(int rootFrameIndex, double rootTimestampS) {
        this.exposureAnchor = 0;
        this.exposureSince = 0;
        this.instabilityAnchor = false;
        this.instabilitySince = false;
        this.anchorReferenceId = null;
        this.anchorFrameIndex = rootFrameIndex;
        this.anchorTimestampS = rootTimestampS;
    }

    /** One usable VO increment composed. */
    public void incrementExposure() {
        exposureSince++;
    }

    /** A validated instability event occurred: sticky until the next accepted anchor. */
    public void noteInstability() {
        instabilitySince = true;
    }

    /**
     * An accepted trusted reference becomes the active anchor. The baseline is the reference's
     * stored provenance, never a fictitious zero (design §9.1): if the reference was stored at
     * {@code E = 70}, the effective exposure immediately after re-anchoring is 70.
     */
    public void inherit(TrustedReference reference, int frameIndex, double timestampS) {
        if (reference == null) {
            throw new IllegalArgumentException("reference is required");
        }
        exposureAnchor = reference.baselineExposure();
        exposureSince = 0;
        instabilityAnchor = reference.baselineInstability();
        instabilitySince = false;
        anchorReferenceId = reference.id();
        anchorFrameIndex = frameIndex;
        anchorTimestampS = timestampS;
    }

    public long exposureAnchor() {
        return exposureAnchor;
    }

    public long exposureSince() {
        return exposureSince;
    }

    /** {@code E_anchor + E_since}. */
    public long effectiveExposure() {
        return exposureAnchor + exposureSince;
    }

    public boolean instabilityAnchor() {
        return instabilityAnchor;
    }

    public boolean instabilitySince() {
        return instabilitySince;
    }

    /** {@code I_anchor OR I_since}. */
    public boolean effectiveInstability() {
        return instabilityAnchor || instabilitySince;
    }

    /** The active anchor's reference id, or {@code null} for the root anchor. */
    @Nullable
    public Integer anchorReferenceId() {
        return anchorReferenceId;
    }

    public int anchorFrameIndex() {
        return anchorFrameIndex;
    }

    public double anchorTimestampS() {
        return anchorTimestampS;
    }

    /** Elapsed time since the anchor was set — the maximum-gap companion to {@code E_since}. */
    public double timeSinceAnchorS(double nowS) {
        return nowS - anchorTimestampS;
    }

    /** An immutable copy of the ledger for logging. */
    public Snapshot snapshot() {
        return new Snapshot(exposureAnchor, exposureSince, effectiveExposure(),
                instabilityAnchor, instabilitySince, effectiveInstability(), anchorReferenceId,
                anchorFrameIndex, anchorTimestampS);
    }

    /** Immutable ledger values at one instant. */
    public record Snapshot(long exposureAnchor, long exposureSince, long effectiveExposure,
                           boolean instabilityAnchor, boolean instabilitySince,
                           boolean effectiveInstability, @Nullable Integer anchorReferenceId,
                           int anchorFrameIndex, double anchorTimestampS) {
    }
}
