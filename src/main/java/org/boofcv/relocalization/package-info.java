/**
 * Relocalization-assisted navigation: the alignment layer between the metric, externally
 * heading-referenced stitching VO and the persistent navigation frame, plus the sparse
 * trusted-reference memory, the exact dual-view retrieval and the decision loop it re-anchors
 * through.
 *
 * <p><b>Maturity: {@code prototype}.</b> The 5 September 2026 final integration design
 * ({@code int_design_iterations/Final_Relocalization_Assisted_Stitching_VO_Design_2026-09-05.pdf};
 * {@code DEC-INT-001} as amended 2026-09-07, {@code DEC-INT-002}, {@code DEC-INT-003},
 * {@code COMP-INT-001}), rebased onto the merged VO contract of {@code DEC-VO-009}/{@code -010}.
 * Transform semantics and state transitions are pinned by synthetic (T1) tests; the plumbing has
 * been exercised once on a UE5 recording as a smoke test; no relocalization performance is claimed
 * and every policy threshold in {@link org.boofcv.relocalization.RelocalizationConfig} is an
 * unvalidated starting value.
 *
 * <h2>What this package does</h2>
 *
 * <pre>
 *   metric VO ──► LocalPoseSample (x_seg [m E], y_seg [m N], psi_nav [deg CW N], validity)
 *              ──► p_global = p_seg + t_e, or ABSENT;  heading = psi_nav, or ABSENT   (NavigationAligner)
 *                                      ▲
 *            accepted trusted reference ┘  t_e ← p_ref − p_seg   (translation only; binary; history untouched)
 * </pre>
 *
 * <p>The estimator is never modified. Scale is the height channel's and direction is the heading
 * channel's, so the alignment is a translation in metres and neither scale nor rotation is a
 * relocalization unknown. A hard VO loss opens the VO's next segment and leaves the alignment
 * UNKNOWN — no zero-motion transform is ever inserted across the gap — while the heading keeps its
 * own validity. The lineage ledger ({@code E_eff = E_anchor + E_since},
 * {@code I_eff = I_anchor OR I_since}) never decays.
 *
 * <p>Skyline evidence arrives through a file boundary as North (required) plus optional synchronised
 * West profiles; retrieval is exact and exhaustive per view; how the two views combine and whether
 * temporal confirmation is required are <em>configurable and unfrozen</em>.
 *
 * <h2>What this package deliberately does not do</h2>
 *
 * <p>No ANN, HMM or particle filter, no covariance, no confidence scalar, no pose blending, no pose
 * graph, no landmark map, no asynchronous worker, no visual/external heading fusion, no scale
 * estimation, no skyline yaw estimation, and no conversion of any C1 lag (single or paired) into a
 * position.
 *
 * <p>No BoofCV or georegression type is imported: the layer consumes
 * {@link org.boofcv.relocalization.LocalPoseSample} (built from the VO's
 * {@code MetricNavigationState}) and that is the whole of its coupling to the estimator
 * (Principle VII).
 */
package org.boofcv.relocalization;
