#!/usr/bin/env bash
# EXP-VO-007: acquire and build `datasets/amtown01-c` from the window frozen in DEC-VO-005.
#
# Every value below is either frozen in DEC-VO-005 (window, datum, altitude, speed) or measured
# for THIS sequence (yaw offset, intrinsics). Nothing is inherited from HKairport01 -- the yaw
# offset in particular must not be, and `--yaw-offset` is a required argument for that reason.
#
#   YAW_OFFSET=<deg> CACHE=<scratch> bash evaluation/tools/exp_vo_007/fetch_amtown01.sh
#
# Resumable: sub-windows are checkpointed, so re-running continues rather than restarting.
# Set N_SUBWINDOWS=<n> to build only the first n of the frozen 27-window tiling -- a technical
# truncation for an acquisition that cannot complete. The tiling is unchanged, so the prefix
# boundary is a tiling boundary rather than a number anyone chose, and every already-fetched
# window is reused from its checkpoint.
set -euo pipefail

PY="${PY:-python}"
CACHE="${CACHE:?set CACHE to a scratch directory}"
YAW_OFFSET="${YAW_OFFSET:?set YAW_OFFSET from derive_yaw_offset.py --scene AMtown01}"
DEFAULT_YAW_DERIVATION="Re-derived for AMtown01 by evaluation/tools/exp_vo_007/derive_yaw_offset.py (EXP-VO-007), by LIT-006 Stage 2's method: circular mean of rtk_yaw - compass(attitude) over straight legs spread across compass quadrants, gated on compass(attitude) tracking RTK course over ground. NOT inherited from HKairport01, whose offset LIT-006 states is sequence-specific until independently re-verified."
YAW_DERIVATION="${YAW_DERIVATION:-$DEFAULT_YAW_DERIVATION}"
INTRINSICS="${INTRINSICS:?set INTRINSICS to the AMtown01 camera_intrinsics JSON}"

"$PY" evaluation/tools/exp_vo_007/ingest_sequence.py \
  --scene AMtown01 \
  --dataset-id amtown01-c \
  --dataset-revision v1 \
  --out-root datasets \
  --t-start 1658137128.011 \
  --t-end   1658138317.359 \
  ${N_SUBWINDOWS:+--n-subwindows "$N_SUBWINDOWS"} \
  --subwindow-s 45 \
  --target-agl 80 \
  --nominal-speed 4.0 \
  --ground-datum 1074.116 \
  --yaw-offset "$YAW_OFFSET" \
  --yaw-derivation "$YAW_DERIVATION" \
  --intrinsics-json "$INTRINSICS" \
  --capture-date 2022-07-18 \
  --environment "village and semi-desert terrain, Urtsadzor, Ararat province, Armenia; low buildings, dirt roads, orchards, sparse vegetation" \
  --evidence-caveat "Third-party recorded flight (MARS-LVIG, Li et al. 2024): DJI M300 RTK, Hikvision CA-050-11UC, 5 mm lens, 80 m nadir, rigidly mounted (not gimbal-stabilised). T3 real-sensor data, but NOT of this project's own system - it characterises the algorithm, never the deployed system. Imagery is JPEG and auto-exposure was enabled. Per-frame parallax is sub-pixel at a 0.4 m baseline, but the scene is NOT flat and the flight is NOT at constant height above the ground it images: LiDAR-measured in-footprint depth spread is 11.3 m (15 % of flight height) and height above the imaged surface varies by a factor of 1.66 (p5-p95) over the cruise, against 1.26 on HKairport01 - LIT-VO-004 section 3. Height above the TAKEOFF datum is constant to 0.030 m; the two are not the same thing. Yaw is converted from the RTK antenna-baseline heading via an offset re-derived empirically for THIS sequence, not inherited." \
  --attitude-caveat "The evaluated Hikvision camera is rigidly mounted, not gimbal-stabilised (the gimbal carries the DJI L1), so airframe attitude tilts the evaluation camera directly. This sequence's own roll/pitch/tilt distribution is in attitude.csv and is summarised in the EXP-VO-007 ingest-gate report; it is NOT assumed to match HKairport01's (median tilt 6.54 deg, max 15.54 deg, EXP-003 R3). COMP-001 section 6 assumption 2." \
  --notes "EXP-VO-007: full eroded cruise of MARS-LVIG AMtown01, frozen in DEC-VO-005 before acquisition. Independent validation sequence for NavigationSource.RIGID_MOTION; not used to design it." \
  --provenance "EXP-VO-007: fetched from DapengFeng/MCAP AMtown01_0.mcap 2026-08-25 by evaluation/tools/exp_vo_007/ingest_sequence.py" \
  --attitude \
  --cache-dir "$CACHE"
