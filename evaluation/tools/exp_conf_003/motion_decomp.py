"""EXP-CONF-003 Phase A: decomposition of a transform disagreement into the
navigation-consumed (similarity) part and the projective residual.

Frozen conventions (EXP-CONF-003 front half):
- readout = exp_conf_002.readout (polar-decomposition rotation at the centre Jacobian,
  log-scale = half log det, centre flow), unchanged;
- similarity approximation S_H of a homography H at the image centre c: rotation rot(H),
  scale exp(logscale(H)), translation fixed by S_H(c) = H(c);
- grid = 16 x 16 over a 5% margin (the panel's xchk convention);
- d_tot  = median grid |H_a(p) - H_b(p)|          (image-space total disagreement)
  d_sim  = median grid |S_a(p) - S_b(p)|          (navigation-consumed disagreement)
  d_proj = median grid |(H_a - S_a)(p) - (H_b - S_b)(p)|   (projective residual disagreement)
  nonrigid(H) = median grid |H(p) - S_H(p)|       (how projective a single transform is)
- direction disagreement defined only when both centre-flow magnitudes >= DIR_MIN_FLOW_PX.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_002"))
from readout import apply_h, readout                                    # noqa: E402

GRID_N = 16
GRID_MARGIN = 0.05
DIR_MIN_FLOW_PX = 2.0


def grid_points(width: int, height: int) -> np.ndarray:
    xs = np.linspace(width * GRID_MARGIN, width * (1 - GRID_MARGIN), GRID_N)
    ys = np.linspace(height * GRID_MARGIN, height * (1 - GRID_MARGIN), GRID_N)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def similarity_approx(H: np.ndarray, width: int, height: int) -> tuple[np.ndarray, dict]:
    """The similarity transform carrying exactly the quantities the RIGID_MOTION readout
    consumes: rotation rot(H), scale exp(logscale(H)), translation such that S(c) = H(c)."""
    ro = readout(H, width, height)
    th = np.radians(ro["rot_deg"])
    s = float(np.exp(ro["log_scale"]))
    A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.array([width / 2.0, height / 2.0])
    hc = apply_h(H, c[None, :])[0]
    t = hc - A @ c
    S = np.eye(3)
    S[:2, :2] = A
    S[:2, 2] = t
    return S, ro


def wrap_deg(d: float) -> float:
    return float((d + 180.0) % 360.0 - 180.0)


def decompose(H_a: np.ndarray, H_b: np.ndarray, width: int, height: int) -> dict:
    """All Phase A disagreement quantities between two transforms of the same pair
    (a = the transform under test, b = the reference), same mapping direction."""
    S_a, ro_a = similarity_approx(H_a, width, height)
    S_b, ro_b = similarity_approx(H_b, width, height)
    c = np.array([[width / 2.0, height / 2.0]])
    g = grid_points(width, height)

    fv_a = apply_h(H_a, c)[0] - c[0]
    fv_b = apply_h(H_b, c)[0] - c[0]
    mag_a, mag_b = float(np.linalg.norm(fv_a)), float(np.linalg.norm(fv_b))
    if mag_a >= DIR_MIN_FLOW_PX and mag_b >= DIR_MIN_FLOW_PX:
        cosang = float(np.dot(fv_a, fv_b) / (mag_a * mag_b))
        dir_dis = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    else:
        dir_dis = float("nan")

    ga, gb = apply_h(H_a, g), apply_h(H_b, g)
    sa, sb = apply_h(S_a, g), apply_h(S_b, g)
    return {
        "rot_a_deg": ro_a["rot_deg"], "rot_b_deg": ro_b["rot_deg"],
        "rot_dis_deg": abs(wrap_deg(ro_a["rot_deg"] - ro_b["rot_deg"])),
        "flow_a_px": mag_a, "flow_b_px": mag_b,
        "flowvec_dis_px": float(np.linalg.norm(fv_a - fv_b)),
        "flowmag_dis_px": abs(mag_a - mag_b),
        "dir_dis_deg": dir_dis,
        "ls_a": ro_a["log_scale"], "ls_b": ro_b["log_scale"],
        "logscale_dis": abs(ro_a["log_scale"] - ro_b["log_scale"]),
        "nonrigid_a_px": float(np.median(np.linalg.norm(ga - sa, axis=1))),
        "nonrigid_b_px": float(np.median(np.linalg.norm(gb - sb, axis=1))),
        "d_tot_px": float(np.median(np.linalg.norm(ga - gb, axis=1))),
        "d_sim_px": float(np.median(np.linalg.norm(sa - sb, axis=1))),
        "d_proj_px": float(np.median(np.linalg.norm((ga - sa) - (gb - sb), axis=1))),
        "proper_a": ro_a["proper"], "proper_b": ro_b["proper"],
    }
