package org.boofcv.relocalization;

/**
 * The "same place?" predicates the retrieval, the agreement test and the temporal tracker use
 * (design §7; {@code EXP-SKY-010}/{@code -011} region rule).
 *
 * <ul>
 *   <li>{@code id_gap} — chronological: two references are one region when their memory ids differ
 *       by at most a gap (neighbouring insertions along the flight). Needs no position and is the
 *       rule most INT tests were written against.</li>
 *   <li>{@code position_radius} — geometric: two references are one region when their stored
 *       persistent positions lie within a declared radius. The radii are declared, terrain-dependent
 *       config values with no default: none is a recognition radius and nothing here bakes one
 *       in.</li>
 * </ul>
 *
 * <h2>Three questions, three radii (2026-09-08)</h2>
 *
 * <p>These are not one question asked three times, and until 2026-09-08 a single
 * {@code region_radius_m} answered all three — a value taken from SKY's <em>recognition-evaluation</em>
 * τ-ball and thereby given a runtime meaning it never had:
 *
 * <ol type="A">
 *   <li><b>Ambiguity grouping</b> — "are these references close enough that they must not count as
 *       competing place hypotheses?" ({@link #inAmbiguityRegionOf}, used by the retrieval's region
 *       0 and its competing score, and so by the margin.)</li>
 *   <li><b>Dual-view agreement</b> — "did the two orthogonal cameras identify the same physical
 *       place?" ({@link #sameDualPlace}.)</li>
 *   <li><b>Temporal continuity</b> — "is this query's candidate the same hypothesis as the previous
 *       query's?" ({@link #sameTemporalRegion}.)</li>
 * </ol>
 *
 * <p>Widening A shrinks the set of competitors and can remove the margin test entirely; widening B
 * makes two views agree that never saw the same place; widening C lets one confirmation chain walk
 * across the map. A radius that is right for one can be badly wrong for another, so each is
 * configured on its own. Under {@code id_gap} the same split holds with the two id gaps:
 * {@code region_gap_references} answers A and B (within one retrieval), {@code region_track_max_gap}
 * answers C (across queries).
 */
final class RegionRule {

    private final boolean positional;
    private final int retrievalIdGap;
    private final int trackIdGap;
    private final double ambiguityRadiusM;
    private final double dualRadiusM;
    private final double temporalRadiusM;

    private RegionRule(boolean positional, int retrievalIdGap, int trackIdGap,
                       double ambiguityRadiusM, double dualRadiusM, double temporalRadiusM) {
        this.positional = positional;
        this.retrievalIdGap = retrievalIdGap;
        this.trackIdGap = trackIdGap;
        this.ambiguityRadiusM = ambiguityRadiusM;
        this.dualRadiusM = dualRadiusM;
        this.temporalRadiusM = temporalRadiusM;
    }

    static RegionRule fromConfig(RelocalizationConfig c) {
        c.validate();
        if (RelocalizationConfig.REGION_POSITION.equals(c.regionRule)) {
            return new RegionRule(true, c.regionGapReferences, c.regionTrackMaxGap,
                    c.effectiveAmbiguityRadiusM(), c.effectiveDualAgreementRadiusM(),
                    c.effectiveTemporalRadiusM());
        }
        return new RegionRule(false, c.regionGapReferences, c.regionTrackMaxGap,
                Double.NaN, Double.NaN, Double.NaN);
    }

    static RegionRule idGap(int retrievalIdGap, int trackIdGap) {
        return new RegionRule(false, retrievalIdGap, trackIdGap, Double.NaN, Double.NaN, Double.NaN);
    }

    boolean positional() {
        return positional;
    }

    /**
     * <b>A.</b> Within one retrieval: does {@code other} belong to the ambiguity region around
     * {@code top}, and therefore <em>not</em> count as a competing hypothesis?
     */
    boolean inAmbiguityRegionOf(TrustedReference top, TrustedReference other) {
        if (positional) {
            return top.positionGlobal().distanceTo(other.positionGlobal()) <= ambiguityRadiusM;
        }
        return Math.abs(top.id() - other.id()) <= retrievalIdGap;
    }

    /** <b>B.</b> Did the two views' winners identify the same physical place? */
    boolean sameDualPlace(TrustedReference northTop, TrustedReference westTop) {
        if (positional) {
            return northTop.positionGlobal().distanceTo(westTop.positionGlobal()) <= dualRadiusM;
        }
        return Math.abs(northTop.id() - westTop.id()) <= retrievalIdGap;
    }

    /** <b>C.</b> Across queries: do two retrieved regions name the same temporal hypothesis? */
    boolean sameTemporalRegion(RetrievalResult.Region a, RetrievalResult.Region b) {
        if (positional) {
            return a.bestPosition().distanceTo(b.bestPosition()) <= temporalRadiusM;
        }
        return sameRegionByIds(a.memberIds(), b.memberIds(), trackIdGap);
    }

    /** Symmetric id-proximity: any member of one within {@code gap} of any member of the other. */
    static boolean sameRegionByIds(java.util.List<Integer> a, java.util.List<Integer> b, int gap) {
        for (int x : a) {
            for (int y : b) {
                if (Math.abs(x - y) <= gap) {
                    return true;
                }
            }
        }
        return false;
    }

    String describe() {
        return positional
                ? RelocalizationConfig.REGION_POSITION + "(ambiguity " + ambiguityRadiusM
                        + " m, dual agreement " + dualRadiusM + " m, temporal " + temporalRadiusM + " m)"
                : RelocalizationConfig.REGION_ID_GAP + "(retrieval " + retrievalIdGap + ", track " + trackIdGap + ")";
    }
}
