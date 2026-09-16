#!/usr/bin/env bash
# EXP-VO-013: build `datasets/amtown01-d` — the FULL frozen `AMtown01` cruise, all 27 sub-windows.
#
# Same scene, same frozen window, same tiling, same datum, same yaw offset and same intrinsics as
# `datasets/amtown01-c` (`EXP-VO-007`, `fetch_amtown01.sh`). The only difference is that `-c` is the
# first 14 of the 27 sub-windows — a technical truncation of an acquisition that ran out of time —
# and this builds all of them.
#
# Two reasons the full cruise is worth the remaining bandwidth, and they are independent:
#
#   1. `EXP-VO-013` needs REAL imagery over which the height above the *imaged surface* varies.
#      Over `-c` the LiDAR-measured AGL spans 74.2..88.2 m (x1.19); over the full cruise it spans
#      41.2..88.2 m (x2.14, p95/p5 1.66). The unbuilt half carries almost all of the variation.
#   2. It discharges a loose end standing since `EXP-VO-007`: `-c` is an OPEN path ending 599.7 m
#      (23.79 %) from its start, so its endpoint error is not comparable with `hkairport01-b`'s
#      near-closed 1.01 %. The full cruise removes that confound, and carries three predictions
#      attached to it in advance (`EXP-VO-008`'s heading-sensitivity explanation, `EXP-VO-009`'s
#      tilt moderator, `EXP-VO-011`'s claim that the scale bias is a per-frame property independent
#      of path shape).
#
# `-c` is NOT modified, and no committed run record changes. A new dataset id is used precisely so
# that every `amtown01-c` result stays exactly reproducible.
#
#   CACHE=<scratch> INTRINSICS=<amtown01_intrinsics.json> bash \
#       evaluation/tools/exp_vo_013/fetch_amtown01_full.sh
#
# Resumable. If `CACHE/checkpoints_amtown01-d` is primed from `-c`'s checkpoints and
# `datasets/amtown01-d/images/` is primed from `-c`'s imagery (hard links are enough), the first 14
# sub-windows resume for free and only the remaining 13 are fetched.
set -euo pipefail

PY="${PY:-python}"
CACHE="${CACHE:?set CACHE to a scratch directory}"
INTRINSICS="${INTRINSICS:?set INTRINSICS to the AMtown01 camera_intrinsics JSON}"
YAW_OFFSET="${YAW_OFFSET:-90.51}"
DEFAULT_YAW_DERIVATION="Re-derived for AMtown01 by evaluation/tools/exp_vo_007/derive_yaw_offset.py (EXP-VO-007), by LIT-006 Stage 2 method: circular mean of rtk_yaw - compass(attitude) over straight legs spread across compass quadrants, gated on compass(attitude) tracking RTK course over ground. NOT inherited from HKairport01. Reused verbatim for amtown01-d, which is the same flight over a longer window of the same tiling."
YAW_DERIVATION="${YAW_DERIVATION:-$DEFAULT_YAW_DERIVATION}"

"$PY" evaluation/tools/exp_vo_007/ingest_sequence.py \
  --scene AMtown01 \
  --dataset-id amtown01-d \
  --dataset-revision v1 \
  --out-root datasets \
  --t-start 1658137128.011 \
  --t-end   1658138317.359 \
  --subwindow-s 45 \
  --target-agl 80 \
  --nominal-speed 4.0 \
  --ground-datum 1074.116 \
  --yaw-offset "$YAW_OFFSET" \
  --yaw-derivation "$YAW_DERIVATION" \
  --intrinsics-json "$INTRINSICS" \
  --capture-date 2022-07-18 \
  --environment "village and semi-desert terrain, Urtsadzor, Ararat province, Armenia; low buildings, dirt roads, orchards, sparse vegetation" \
  --evidence-caveat "Third-party recorded flight (MARS-LVIG, Li et al. 2024): DJI M300 RTK, Hikvision CA-050-11UC, 5 mm lens, 80 m nadir, rigidly mounted (not gimbal-stabilised). T3 real-sensor data, but NOT of this project's own system - it characterises the algorithm, never the deployed system. Imagery is JPEG and auto-exposure was enabled. THE FULL CRUISE, all 27 sub-windows of the DEC-VO-005 tiling (amtown01-c is its first 14). Height above the TAKEOFF datum is constant at 80.00 +/- 0.03 m across the whole cruise; height above the IMAGED SURFACE, measured by the onboard Livox Avia inside the camera cone, spans 41.2..88.2 m (max/min 2.14, p95/p5 1.66). The two are not the same quantity and on this sequence they differ by up to a factor of two - LIT-VO-004 section 4. That difference is TERRAIN, not aircraft motion, and it is why this sequence tests the datum limitation of an altitude-constrained metric readout rather than its benefit. Yaw is converted from the RTK antenna-baseline heading via an offset re-derived empirically for AMtown01, not inherited from HKairport01." \
  --attitude-caveat "The evaluated Hikvision camera is rigidly mounted, not gimbal-stabilised (the gimbal carries the DJI L1), so airframe attitude tilts the evaluation camera directly. This sequence's own roll/pitch/tilt distribution is in attitude.csv; it is NOT assumed to match HKairport01's. COMP-001 section 6 assumption 2." \
  --notes "EXP-VO-013: the full frozen AMtown01 cruise. Window, tiling, datum, target altitude, nominal speed, yaw offset and intrinsics are all inherited unchanged from DEC-VO-005 / EXP-VO-007; nothing about the selection was chosen after seeing a VO result. Built for a real-image test of the altitude-constrained metric readout, and it also removes amtown01-c's open-path confound." \
  --provenance "EXP-VO-013: fetched from DapengFeng/MCAP AMtown01_0.mcap 2026-08-28 by evaluation/tools/exp_vo_007/ingest_sequence.py; sub-windows 1-14 resumed from the amtown01-c acquisition of 2026-08-25 (identical bytes, hard-linked imagery), 15-27 fetched fresh." \
  --attitude \
  --cache-dir "$CACHE"
