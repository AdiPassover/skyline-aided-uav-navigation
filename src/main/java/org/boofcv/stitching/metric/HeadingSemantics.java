package org.boofcv.stitching.metric;

/**
 * <b>What the heading column means</b>, declared rather than inferred.
 *
 * <p>The reason this type exists is the failure mode {@code EXP-VO-013} R6 measured: the heading of
 * the <b>airframe</b> is not the heading of the <b>camera</b>, and substituting one for the other
 * put every trajectory ~66° off from the first metre (ATE 439.84 m against 146.60 m). Two datasets
 * in this repository carry both quantities, and nothing but a declaration distinguishes them:
 * MARS-LVIG's {@code attitude.csv} is airframe, and the UE capture's {@code groundtruth.csv} is the
 * nadir camera's. A config pointing at the wrong one would otherwise produce a differently-armed run
 * that looked entirely ordinary.
 *
 * <p>This mirrors {@code MetricReadoutSettings.heightSemantics}, for the same reason and with the
 * same consequence: an unrecognised value is refused, and the run manifest records which arm ran.
 */
public enum HeadingSemantics {

    /**
     * <b>Production-like.</b> The source column is the <b>airframe's</b> heading, degrees clockwise
     * from North — a flight controller's fused attitude yaw, as published on MARS-LVIG by
     * {@code /dji_osdk_ros/attitude} and committed as {@code datasets/<id>/attitude.csv}'s
     * {@code yaw_compass_deg}.
     *
     * <p>It is <b>not</b> the navigation heading and must be converted:
     * {@code yaw_nav = wrap360(body_heading + delta_mount)}, where {@code delta_mount} is a
     * <b>declared external platform calibration</b> — never fitted at runtime, never inferred from
     * trajectory ground truth, and refused rather than defaulted when absent.
     *
     * <p><b>Caveat that travels with this arm:</b> on the only platform for which this stream exists
     * here, its GPS-independence is <b>not established</b>. The FC yaw's non-drift over 1189 s shows
     * it is absolutely referenced by something; {@code rtk_info_yaw = 50} on 100 % of samples in
     * every committed window means there is no RTK-degraded interval to test against.
     */
    FC_AHRS_BODY_COMPASS(true),

    /**
     * <b>Simulator.</b> The source column is already the <b>nadir (navigation) camera's</b> heading,
     * degrees clockwise from North — the UE capture's {@code groundtruth.csv:heading_deg}, defined by
     * the rig as <i>"projection of image-up axis onto horizontal plane; clockwise from simulator
     * North"</i> and verified against the camera quaternion at every frame by the ingest's C5 check.
     *
     * <p>It is used <b>directly</b>. No mounting correction is applied, and the airframe yaw in
     * {@code ue_body.csv} is diagnostic only: the UE nadir camera is
     * {@code world_stabilized_nadir_independent_yaw}, so body and camera yaw are related by no fixed
     * offset at all ({@code camera_body_yaw_is_fixed_offset: false} in the ingest provenance). Any
     * lag between simulated body yaw and nadir-camera yaw is a simulator/control artifact, not a
     * sensor mounting offset, and VO must not compensate it.
     */
    SIM_NADIR_CAMERA_HEADING(false);

    private final boolean requiresMountCalibration;

    HeadingSemantics(boolean requiresMountCalibration) {
        this.requiresMountCalibration = requiresMountCalibration;
    }

    /**
     * True when this arm needs a declared camera-to-body mounting yaw offset. A configuration that
     * omits it must be refused, not defaulted to zero — zero is a perfectly plausible-looking value
     * that would silently reintroduce {@code EXP-VO-013} R6's error.
     */
    public boolean requiresMountCalibration() {
        return requiresMountCalibration;
    }

    /** Parses the config spelling ({@code fc_ahrs_body_compass} / {@code sim_nadir_camera_heading}). */
    public static HeadingSemantics parse(String text) {
        if (text != null) {
            String t = text.trim();
            if ("fc_ahrs_body_compass".equalsIgnoreCase(t)) {
                return FC_AHRS_BODY_COMPASS;
            }
            if ("sim_nadir_camera_heading".equalsIgnoreCase(t)) {
                return SIM_NADIR_CAMERA_HEADING;
            }
        }
        throw new IllegalArgumentException("Unknown heading_readout.heading_semantics '" + text
                + "'; expected \"fc_ahrs_body_compass\" (airframe heading, needs delta_mount_deg) or "
                + "\"sim_nadir_camera_heading\" (already the navigation camera's, used directly)");
    }

    /** The config spelling, for round-tripping into a run manifest. */
    public String configName() {
        return name().toLowerCase(java.util.Locale.ROOT);
    }
}
