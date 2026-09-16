package org.boofcv.relocalization;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

import javax.annotation.Nullable;

import java.io.IOException;
import java.nio.file.Path;

/**
 * The reference-memory, retrieval, scheduling and acceptance policy, as a config file rather than
 * scattered constants (design §13).
 *
 * <p><b>Every default below is an unvalidated starting value.</b> The design requires these to be
 * selected sequentially on development trajectories and frozen before any held-out evaluation;
 * none has been. They exist so the mechanism is testable and so a committed config can pin whatever
 * an experiment used. Unknown keys are refused, not ignored — the SKY lane's {@code build_matcher}
 * once ran a diagnostic at the wrong setting because a misspelt key was silently dropped.
 *
 * <p><b>Nothing here is a universal production constant.</b> The dual-view fusion rule, the
 * temporal confirmation requirement, the region rule and every threshold are selectable so that a
 * policy is a config file, not a rewrite of retrieval, candidate representation or logging. The
 * <em>starting experimental policy</em> selected on {@code EXP-SKY-011} + {@code EXP-SKY-012}
 * ({@code DEC-INT-003}, amended 2026-09-07) is the committed
 * {@code evaluation/eval_configs/int/reloc-starting-dual-weakest.json}: {@code weakest_view},
 * region-level margins on the SKY τ-ball, 0.95 / 0.20, temporal confirmation {@code fallback};
 * {@code reloc-alternative-dual-strict.json} is the equally-safe stricter comparison. The
 * <em>code</em> defaults below still reproduce the North-only, temporally-confirmed behaviour every
 * earlier INT test was written against — a baseline, not the policy.
 *
 * <p>Two things this file deliberately does <em>not</em> parameterise: the alignment group/units
 * (translation in metres ENU, {@code DEC-INT-001} amendment) and any conversion from a matcher's
 * lag to a position (none exists; {@code DEC-INT-002}).
 */
public class RelocalizationConfig {

    public static final String MODE_COUNT = "count";
    public static final String MODE_TIME = "time";
    public static final String MATCHER_C0 = "c0_frozen_ncc";
    public static final String MATCHER_C1 = "c1_bounded_lag_ncc";

    /** North evidence only; West, when present, is logged and never consulted. The legacy rule. */
    public static final String FUSION_NORTH_ONLY = "north_only";
    /** {@code EXP-SKY-011} "strict_agreement": both views pass their gate and name one region. */
    public static final String FUSION_STRICT = "strict_agreement";
    /** {@code EXP-SKY-011} "weakest_view": one ranking by {@code min(s_N, s_W)}, then the gate. */
    public static final String FUSION_WEAKEST = "weakest_view";
    /** {@code EXP-SKY-011} "mean_score": one ranking by {@code (s_N + s_W)/2}. Reported there, not adopted. */
    public static final String FUSION_MEAN = "mean_score";

    public static final String REGION_ID_GAP = "id_gap";
    public static final String REGION_POSITION = "position_radius";

    public static final String TEMPORAL_REQUIRED = "required";
    public static final String TEMPORAL_FALLBACK = "fallback";
    public static final String TEMPORAL_DISABLED = "disabled";

    // ------------------------------------------------------------------ descriptor

    /** Fixed descriptor length; 256 in every SKY record ({@code hsreloc.retrieval.profile}). */
    @JsonProperty("descriptor_length")
    public int descriptorLength = 256;

    /** Profiles whose sample SD (population, ddof 0) is below this are degenerate and refused. */
    @JsonProperty("degenerate_std_floor")
    public double degenerateStdFloor = 1e-3;

    // ------------------------------------------------------------------ matcher (DEC-INT-002)

    /**
     * {@link #MATCHER_C0} — pointwise NCC at lag 0, no alignment search — or {@link #MATCHER_C1}, the
     * same primitive after a bounded horizontal shift. Both sit on one score scale; the same matcher
     * scores both views.
     *
     * <p><b>Roles since {@code DEC-INT-007} (2026-09-08, adopting {@code DEC-SKY-008} on
     * {@code EXP-SKY-013}):</b> C0 is the <b>primary</b> matcher for the {@code DISCRETE_REFERENCE}
     * pose action — bounded-lag freedom raises the score of <em>distant</em> references (figure-eight
     * West, 150–300 m: C0 0.809 vs C1-32 0.951) while near matches stay at 0.997 either way, so the
     * selected reference lands nearer under C0. {@code C1-4} is the pre-registered
     * <b>fallback / comparison</b> arm should C0's mountains coverage prove inadequate; {@code C1-32}
     * ({@code DEC-SKY-007}'s recognition recommendation, the 2026-09-07 starting policy) is a
     * <b>legacy / comparison</b> setting, no longer a starting default. Scores are <b>not portable
     * between arms</b> ({@code EXP-SKY-013} R6): every arm carries its own DEV-calibrated gate.
     *
     * <p>The <em>code</em> default stays {@link #MATCHER_C1} at 32 so that every earlier config and
     * test means what it meant; a policy names its matcher explicitly and the run manifest records
     * the variant, the effective lag bound and the role ({@link #matcherRole()}).
     */
    @JsonProperty("matcher")
    public String matcher = MATCHER_C1;

    /**
     * C1's lag bound in profile samples. 32 is the {@code C1-32} the SKY lane evaluated; 4 is the
     * {@code C1-4} fallback rung of {@code DEC-INT-007}. The bound is declared from heading
     * uncertainty and camera FOV ({@code lag_samples_for_degrees}) and must never be chosen from
     * retrieval correctness. <b>Ignored under C0</b>, which searches no alignment
     * ({@link #effectiveMaxLagSamples()} is 0 there).
     */
    @JsonProperty("max_lag_samples")
    public int maxLagSamples = 32;

    /** C1: alignments comparing fewer than this fraction of the samples are not scored. */
    @JsonProperty("min_overlap_frac")
    public double minOverlapFrac = 0.6;

    // ------------------------------------------------------------------ insertion (design §6.2)

    /**
     * Novelty: minimum {@code 1 − NCC} of the North view to the last trusted reference for a new
     * insert. Below this the view is not novel and only the max-spacing fallback can insert.
     */
    @JsonProperty("novelty_min_distance")
    public double noveltyMinDistance = 0.05;

    /** Max-spacing fallback: insert after this many frames since the last trusted reference. */
    @JsonProperty("max_spacing_frames")
    public int maxSpacingFrames = 100;

    /**
     * {@code N_search}: the normal exposure deadline for a scheduled search (design §5), reused
     * as the conservative trusted-insertion bound of §6.2 — references created after
     * {@code E_eff} exceeds it are not trusted anchors.
     */
    @JsonProperty("search_exposure_bound")
    public long searchExposureBound = 300;

    // ------------------------------------------------------------------ recent exclusion (§6.3)

    /**
     * Exactly one recent-reference exclusion mechanism is active: {@link #MODE_COUNT} excludes
     * the latest {@code recent_exclusion_count} references; {@link #MODE_TIME} excludes references
     * newer than {@code recent_exclusion_seconds} before the query. The two are never combined
     * silently (author correction at the P0 checkpoint).
     */
    @JsonProperty("recent_exclusion_mode")
    public String recentExclusionMode = MODE_COUNT;

    @JsonProperty("recent_exclusion_count")
    public int recentExclusionCount = 5;

    @JsonProperty("recent_exclusion_seconds")
    public Double recentExclusionSeconds = null;

    // ------------------------------------------------------------------ retrieval / regions (§7)

    /**
     * {@link #REGION_ID_GAP}: two references are one region when their ids differ by at most
     * {@code region_gap_references} (within a retrieval) or {@code region_track_max_gap} (across
     * queries) — chronological neighbours. {@link #REGION_POSITION}: one region when their stored
     * persistent positions lie within a declared radius. No radius has a default and none is a
     * recognition radius.
     */
    @JsonProperty("region_rule")
    public String regionRule = REGION_ID_GAP;

    /** Candidates whose reference ids differ by at most this from the top-1 join its region ({@code id_gap}). */
    @JsonProperty("region_gap_references")
    public int regionGapReferences = 1;

    /**
     * <b>Legacy.</b> One radius for all three positional questions. Accepted so that pre-2026-09-08
     * configs and experiment manifests keep meaning exactly what they meant, and refused in
     * combination with any of the three specific radii below — an old value is mapped explicitly
     * and labelled ({@link #regionRadiiAreLegacyMapped()}), never silently reinterpreted as a
     * considered choice of three.
     */
    @JsonProperty("region_radius_m")
    public Double regionRadiusM = null;

    /**
     * <b>A. Ambiguity grouping.</b> "Are these two references close enough that they must not count
     * as competing place hypotheses?" Governs the top region within one retrieval and therefore the
     * competing score and the margin. Too large and the whole memory becomes one region, the margin
     * becomes {@code null} and condition (2) stops being evaluated at all.
     */
    @JsonProperty("ambiguity_region_radius_m")
    public Double ambiguityRegionRadiusM = null;

    /**
     * <b>B. Dual-view agreement.</b> "Did the two orthogonal cameras identify the same physical
     * place?" Governs only the North/West agreement test. It is a statement about how far apart two
     * independent recognitions may land and still be one place — not about ambiguity, and not about
     * time.
     */
    @JsonProperty("dual_agreement_radius_m")
    public Double dualAgreementRadiusM = null;

    /**
     * <b>C. Temporal continuity.</b> "Is this query's candidate the same hypothesis as the previous
     * query's?" Governs {@link RegionTracker} only. Note that the tracked representative moves with
     * each confirmation, so this bounds the step between consecutive queries, not the extent of the
     * chain — see {@code RegionTracker}'s class javadoc.
     */
    @JsonProperty("temporal_region_radius_m")
    public Double temporalRegionRadiusM = null;

    /** Exact retrieval returns the best {@code K} references per view (before region grouping). */
    @JsonProperty("top_k")
    public int topK = 5;

    // ------------------------------------------------------------------ dual view (DEC-INT-003)

    /**
     * How North and West evidence combine — one of the three {@code EXP-SKY-011} rules, or
     * {@link #FUSION_NORTH_ONLY}. <b>Not a frozen choice</b>: the final policy awaits the city
     * HardSwipe recording. Under any rule a query without a valid West view is scored on North
     * alone and marked incomplete; missing West is never agreement.
     */
    @JsonProperty("fusion_rule")
    public String fusionRule = FUSION_NORTH_ONLY;

    /** {@code theta_s} for the West view's own gate under {@code strict_agreement}; {@code null} = same as North's. */
    @JsonProperty("west_match_threshold")
    public Double westMatchThreshold = null;

    /** {@code theta_m} for the West view's own gate under {@code strict_agreement}; {@code null} = same as North's. */
    @JsonProperty("west_margin_threshold")
    public Double westMarginThreshold = null;

    /**
     * When temporal (region-persistence) confirmation is required for acceptance:
     * {@link #TEMPORAL_REQUIRED} always; {@link #TEMPORAL_FALLBACK} only when the dual-view
     * evidence did not itself confirm the region (North-only rule, West unavailable/invalid/below
     * its gate); {@link #TEMPORAL_DISABLED} never. {@code EXP-SKY-011} R6 motivates
     * {@code fallback}; it is not frozen.
     */
    @JsonProperty("temporal_confirmation")
    public String temporalConfirmation = TEMPORAL_REQUIRED;

    // ------------------------------------------------------------------ boundary

    /**
     * Largest allowed |skyline capture time − VO frame time| for a profile row paired by time
     * ({@code sync_dt_s} in the exported file). Rows beyond it are loaded as invalid observations
     * with the residual as the reason — explicit, never silently dropped.
     */
    @JsonProperty("skyline_sync_tolerance_s")
    public double skylineSyncToleranceS = 0.05;

    // ------------------------------------------------------------------ scheduling (§5, §13)

    /** {@code T_search_max}: elapsed-time maximum since the anchor; {@code null} disables it. */
    @JsonProperty("search_time_max_s")
    public Double searchTimeMaxS = null;

    /** {@code N_min_query_gap}: minimum frames between attempts after an unsuccessful one. */
    @JsonProperty("min_retry_gap_frames")
    public int minRetryGapFrames = 30;

    /**
     * {@code tau_I}: a {@code delta_rot_refit} value (degrees) at or above this marks unresolved
     * instability and requests an immediate search. {@code null} disables escalation; it never
     * disables the baseline schedule. Semantics frozen by {@code DEC-CONF-003}.
     */
    @JsonProperty("instability_threshold_deg")
    public Double instabilityThresholdDeg = null;

    // ------------------------------------------------------------------ acceptance (§8, §13)

    /**
     * {@code tau_match}: minimum best-region score of the primary ranking. 0.9 is
     * {@code accept-v2-margin}'s value, calibrated on C0 scores in {@code EXP-SKY-008} and
     * inherited, not re-validated, under C1.
     */
    @JsonProperty("match_threshold")
    public double matchThreshold = 0.9;

    /** {@code tau_margin}: minimum region-level margin over the best reference outside the top region. */
    @JsonProperty("margin_threshold")
    public double marginThreshold = 0.15;

    /** {@code N_confirm}: consecutive eligible queries the same region must persist over, when required. */
    @JsonProperty("confirm_queries")
    public int confirmQueries = 2;

    /**
     * Region identity across consecutive queries under {@code id_gap}: two regions are the same
     * place when any pair of their member reference ids lies within this many ids. Symmetric, so
     * forward, reverse and near-stationary traversal all count (design §7).
     */
    @JsonProperty("region_track_max_gap")
    public int regionTrackMaxGap = 2;

    public static RelocalizationConfig load(Path path) throws IOException {
        ObjectMapper mapper = new ObjectMapper()
                .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);
        RelocalizationConfig c = mapper.readValue(path.toFile(), RelocalizationConfig.class);
        c.validate();
        return c;
    }

    public boolean excludeByCount() {
        return MODE_COUNT.equals(recentExclusionMode);
    }

    public boolean excludeByTime() {
        return MODE_TIME.equals(recentExclusionMode);
    }

    /** The lag freedom actually searched: 0 under C0 whatever {@code max_lag_samples} says. */
    public int effectiveMaxLagSamples() {
        return MATCHER_C0.equals(matcher) ? 0 : maxLagSamples;
    }

    /**
     * The matcher's standing under {@code DEC-INT-007}, for the run manifest: C0 primary; C1-4 the
     * pre-registered fallback / comparison; C1-32 legacy / comparison (the recognition
     * recommendation of {@code DEC-SKY-007} and the 2026-09-07 starting policy); anything else a
     * non-standard selection.
     */
    public String matcherRole() {
        if (MATCHER_C0.equals(matcher)) {
            return "PRIMARY (DEC-INT-007 on EXP-SKY-013 / DEC-SKY-008): C0 pointwise NCC, no alignment search; "
                    + "gate must be C0-calibrated on DEV";
        }
        if (maxLagSamples == 4) {
            return "FALLBACK / COMPARISON (DEC-INT-007): C1-4, the pre-registered rung if C0 coverage is "
                    + "inadequate; gate must be C1-4-calibrated on DEV, never carried from C0 or C1-32";
        }
        if (maxLagSamples == 32) {
            return "LEGACY / COMPARISON (DEC-INT-007): C1-32, DEC-SKY-007's recognition matcher and the "
                    + "2026-09-07 starting policy; not a starting default for DISCRETE_REFERENCE since 2026-09-08";
        }
        return "NON-STANDARD SELECTION: C1 at |lag| <= " + maxLagSamples + " samples (no decision record names it)";
    }

    /** True when the West bank is scored at all (any rule but North-only). */
    public boolean fusesWest() {
        return !FUSION_NORTH_ONLY.equals(fusionRule);
    }

    /** True when the primary ranking is a fused one ({@code weakest_view} / {@code mean_score}). */
    public boolean usesFusedRanking() {
        return FUSION_WEAKEST.equals(fusionRule) || FUSION_MEAN.equals(fusionRule);
    }

    private static void checkRadius(String key, Double v) {
        if (v == null || !(v > 0.0) || !Double.isFinite(v)) {
            throw new IllegalArgumentException(key + " must be a finite radius > 0 metres when "
                    + "region_rule is \"" + REGION_POSITION + "\"; got " + v);
        }
    }

    /**
     * True when the three positional radii came from the single legacy {@code region_radius_m}
     * rather than being declared independently. Recorded in the run manifest so a run made under
     * one radius is never read as though three had been chosen.
     */
    public boolean regionRadiiAreLegacyMapped() {
        return REGION_POSITION.equals(regionRule) && regionRadiusM != null;
    }

    /** A: the ambiguity-grouping radius actually in force (metres), or {@code null} under id_gap. */
    @Nullable
    public Double effectiveAmbiguityRadiusM() {
        return REGION_POSITION.equals(regionRule)
                ? (regionRadiusM != null ? regionRadiusM : ambiguityRegionRadiusM) : null;
    }

    /** B: the North/West agreement radius actually in force (metres), or {@code null} under id_gap. */
    @Nullable
    public Double effectiveDualAgreementRadiusM() {
        return REGION_POSITION.equals(regionRule)
                ? (regionRadiusM != null ? regionRadiusM : dualAgreementRadiusM) : null;
    }

    /** C: the temporal-continuity radius actually in force (metres), or {@code null} under id_gap. */
    @Nullable
    public Double effectiveTemporalRadiusM() {
        return REGION_POSITION.equals(regionRule)
                ? (regionRadiusM != null ? regionRadiusM : temporalRegionRadiusM) : null;
    }

    /**
     * Whether a query with North evidence only can ever be accepted under this config: never
     * under a dual rule (the dual condition is the confirmation; {@code AcceptanceGate}), after
     * {@code N_confirm} North-only queries under {@code north_only} with temporal confirmation, on
     * a single capture under {@code north_only} / {@code disabled}.
     */
    public String singleViewAcceptance() {
        if (fusesWest()) {
            return "never: a missing, invalid or refusing West retains the candidate whatever the temporal support (EXP-SKY-012)";
        }
        return TEMPORAL_DISABLED.equals(temporalConfirmation)
                ? "single North capture (north_only + disabled: the comparison baseline EXP-SKY-010/012 measured as unsafe)"
                : "after " + confirmQueries + " consecutive gate-passing North queries (north_only; EXP-SKY-012 R5: not a substitute for a second view)";
    }

    /**
     * Where this configuration stands relative to the named policies: the {@code EXP-INT-001} C0
     * policy and its C1-4 fallback ({@code DEC-INT-007}, DEV-calibrated 2026-09-08), the 2026-09-07
     * C1-32 starting policy of {@code DEC-INT-003} (historical), or none of them.
     */
    public String policyStatus() {
        boolean expInt001Shape = FUSION_WEAKEST.equals(fusionRule) && TEMPORAL_FALLBACK.equals(temporalConfirmation)
                && matchThreshold == 0.90 && marginThreshold == 0.30 && REGION_POSITION.equals(regionRule)
                && regionRadiusM == null && ambiguityRegionRadiusM != null && ambiguityRegionRadiusM == 50.0
                && dualAgreementRadiusM != null && dualAgreementRadiusM == 15.0
                && temporalRegionRadiusM != null && temporalRegionRadiusM == 30.0;
        String expTail = " — selected on the five EXP-INT-001 DEV recordings by the pre-registered rule "
                + "(0 wrong-place, 0 severely harmful, most genuine revisits; ties to the conservative side), "
                + "before any evaluation-recording number was seen; a T2 development selection, not a universal constant";
        if (expInt001Shape && MATCHER_C0.equals(matcher)) {
            return "EXP-INT-001 C0 POLICY (DEC-INT-007 primary): C0, weakest-view fusion, region-level margin, "
                    + "0.90 / 0.30, radii 50 / 15 / 30 m, temporal fallback" + expTail;
        }
        if (expInt001Shape && MATCHER_C1.equals(matcher) && maxLagSamples == 4) {
            return "EXP-INT-001 C1-4 FALLBACK POLICY (DEC-INT-007): C1-4, weakest-view fusion, region-level margin, "
                    + "0.90 / 0.30, radii 50 / 15 / 30 m, temporal fallback" + expTail;
        }
        boolean starting = FUSION_WEAKEST.equals(fusionRule) && TEMPORAL_FALLBACK.equals(temporalConfirmation)
                && matchThreshold == 0.95 && marginThreshold == 0.20;
        boolean strictAlt = FUSION_STRICT.equals(fusionRule) && TEMPORAL_FALLBACK.equals(temporalConfirmation)
                && matchThreshold == 0.95 && marginThreshold == 0.20;
        String tail = " — the 2026-09-07 starting value selected on EXP-SKY-011/012 (DEC-INT-003 amended), calibrated "
                + "on C1-32 scores and HISTORICAL since DEC-INT-007 (2026-09-08): not a C0 gate, not a universal constant";
        if (starting) {
            return "2026-09-07 STARTING POLICY (historical): weakest-view fusion, region-level margin, 0.95 / 0.20, temporal fallback" + tail;
        }
        if (strictAlt) {
            return "2026-09-07 CONSERVATIVE ALTERNATIVE (historical): strict North+West agreement, 0.95 / 0.20, temporal fallback" + tail;
        }
        if (!fusesWest()) {
            return "SINGLE-VIEW BASELINE (north_only): the arm EXP-SKY-010/012 measured as unsafe on the flat city; comparison only";
        }
        return "NON-STANDARD SELECTION (" + fusionRule + ", " + matchThreshold + " / " + marginThreshold + ", temporal "
                + temporalConfirmation + ")" + tail;
    }

    public double effectiveWestMatchThreshold() {
        return westMatchThreshold == null ? matchThreshold : westMatchThreshold;
    }

    public double effectiveWestMarginThreshold() {
        return westMarginThreshold == null ? marginThreshold : westMarginThreshold;
    }

    public RelocalizationConfig validate() {
        if (descriptorLength < 2) {
            throw new IllegalArgumentException("descriptor_length must be >= 2");
        }
        if (degenerateStdFloor < 0.0 || !Double.isFinite(degenerateStdFloor)) {
            throw new IllegalArgumentException("degenerate_std_floor must be finite and >= 0");
        }
        if (!MATCHER_C0.equals(matcher) && !MATCHER_C1.equals(matcher)) {
            throw new IllegalArgumentException("matcher must be \"" + MATCHER_C1 + "\" or \""
                    + MATCHER_C0 + "\", got \"" + matcher + "\"");
        }
        if (maxLagSamples < 0) {
            throw new IllegalArgumentException("max_lag_samples must be >= 0");
        }
        if (!(minOverlapFrac > 0.0 && minOverlapFrac <= 1.0)) {
            throw new IllegalArgumentException("min_overlap_frac must be in (0, 1]");
        }
        if (noveltyMinDistance < 0.0 || noveltyMinDistance > 2.0) {
            throw new IllegalArgumentException("novelty_min_distance must be in [0, 2]");
        }
        if (maxSpacingFrames < 1) {
            throw new IllegalArgumentException("max_spacing_frames must be >= 1");
        }
        if (searchExposureBound < 0) {
            throw new IllegalArgumentException("search_exposure_bound must be >= 0");
        }
        if (!MODE_COUNT.equals(recentExclusionMode) && !MODE_TIME.equals(recentExclusionMode)) {
            throw new IllegalArgumentException("recent_exclusion_mode must be \"" + MODE_COUNT
                    + "\" or \"" + MODE_TIME + "\", got \"" + recentExclusionMode + "\"");
        }
        if (recentExclusionCount < 0) {
            throw new IllegalArgumentException("recent_exclusion_count must be >= 0");
        }
        if (excludeByTime()) {
            if (recentExclusionSeconds == null || recentExclusionSeconds < 0.0
                    || !Double.isFinite(recentExclusionSeconds)) {
                throw new IllegalArgumentException(
                        "recent_exclusion_mode \"time\" needs a finite recent_exclusion_seconds >= 0");
            }
        } else if (recentExclusionSeconds != null) {
            throw new IllegalArgumentException("recent_exclusion_seconds is set but "
                    + "recent_exclusion_mode is \"count\"; the two mechanisms are never combined — "
                    + "pick one mode and remove the other parameter");
        }
        if (!REGION_ID_GAP.equals(regionRule) && !REGION_POSITION.equals(regionRule)) {
            throw new IllegalArgumentException("region_rule must be \"" + REGION_ID_GAP + "\" or \""
                    + REGION_POSITION + "\", got \"" + regionRule + "\"");
        }
        if (regionGapReferences < 0) {
            throw new IllegalArgumentException("region_gap_references must be >= 0");
        }
        boolean anySpecific = ambiguityRegionRadiusM != null || dualAgreementRadiusM != null
                || temporalRegionRadiusM != null;
        if (REGION_POSITION.equals(regionRule)) {
            if (regionRadiusM != null && anySpecific) {
                throw new IllegalArgumentException("region_radius_m (legacy: one radius for all "
                        + "three positional questions) is set alongside ambiguity_region_radius_m / "
                        + "dual_agreement_radius_m / temporal_region_radius_m. The two forms are "
                        + "never combined — a legacy value is mapped as a whole and labelled as "
                        + "legacy, or the three are declared explicitly. Remove one form.");
            }
            if (regionRadiusM == null && !anySpecific) {
                throw new IllegalArgumentException("region_rule \"" + REGION_POSITION + "\" needs "
                        + "radii: either all three of ambiguity_region_radius_m, "
                        + "dual_agreement_radius_m and temporal_region_radius_m, or the legacy "
                        + "region_radius_m for all three — declared per terrain, never defaulted");
            }
            if (regionRadiusM != null) {
                checkRadius("region_radius_m", regionRadiusM);
            } else {
                checkRadius("ambiguity_region_radius_m", ambiguityRegionRadiusM);
                checkRadius("dual_agreement_radius_m", dualAgreementRadiusM);
                checkRadius("temporal_region_radius_m", temporalRegionRadiusM);
            }
        } else if (regionRadiusM != null || anySpecific) {
            throw new IllegalArgumentException("a positional radius is set but region_rule is \""
                    + REGION_ID_GAP + "\"; remove one");
        }
        if (topK < 1) {
            throw new IllegalArgumentException("top_k must be >= 1");
        }
        if (!FUSION_NORTH_ONLY.equals(fusionRule) && !FUSION_STRICT.equals(fusionRule)
                && !FUSION_WEAKEST.equals(fusionRule) && !FUSION_MEAN.equals(fusionRule)) {
            throw new IllegalArgumentException("fusion_rule must be one of \"" + FUSION_NORTH_ONLY
                    + "\", \"" + FUSION_STRICT + "\", \"" + FUSION_WEAKEST + "\", \"" + FUSION_MEAN
                    + "\", got \"" + fusionRule + "\"");
        }
        if (westMatchThreshold != null && (westMatchThreshold < -1.0 || westMatchThreshold > 1.0)) {
            throw new IllegalArgumentException("west_match_threshold must be in [-1, 1]");
        }
        if (westMarginThreshold != null && (westMarginThreshold < 0.0 || westMarginThreshold > 2.0)) {
            throw new IllegalArgumentException("west_margin_threshold must be in [0, 2]");
        }
        if (!TEMPORAL_REQUIRED.equals(temporalConfirmation) && !TEMPORAL_FALLBACK.equals(temporalConfirmation)
                && !TEMPORAL_DISABLED.equals(temporalConfirmation)) {
            throw new IllegalArgumentException("temporal_confirmation must be \"" + TEMPORAL_REQUIRED
                    + "\", \"" + TEMPORAL_FALLBACK + "\" or \"" + TEMPORAL_DISABLED + "\", got \""
                    + temporalConfirmation + "\"");
        }
        if (!(skylineSyncToleranceS >= 0.0) || !Double.isFinite(skylineSyncToleranceS)) {
            throw new IllegalArgumentException("skyline_sync_tolerance_s must be finite and >= 0");
        }
        if (searchTimeMaxS != null && !(searchTimeMaxS > 0.0)) {
            throw new IllegalArgumentException("search_time_max_s must be > 0 when set");
        }
        if (minRetryGapFrames < 0) {
            throw new IllegalArgumentException("min_retry_gap_frames must be >= 0");
        }
        if (instabilityThresholdDeg != null && !(instabilityThresholdDeg >= 0.0)) {
            throw new IllegalArgumentException("instability_threshold_deg must be >= 0 when set");
        }
        if (matchThreshold < -1.0 || matchThreshold > 1.0) {
            throw new IllegalArgumentException("match_threshold must be in [-1, 1]");
        }
        if (marginThreshold < 0.0 || marginThreshold > 2.0) {
            throw new IllegalArgumentException("margin_threshold must be in [0, 2]");
        }
        if (confirmQueries < 1) {
            throw new IllegalArgumentException("confirm_queries must be >= 1");
        }
        if (regionTrackMaxGap < 0) {
            throw new IllegalArgumentException("region_track_max_gap must be >= 0");
        }
        return this;
    }
}
