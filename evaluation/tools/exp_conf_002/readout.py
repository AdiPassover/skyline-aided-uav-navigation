"""EXP-CONF-002: shared geometric helpers for the reference diagnostic panel.

Increment readout convention (frozen in the EXP-CONF-002 front half): for a pair homography H,
- rotation = angle of the polar-decomposition rotation of the 2x2 Jacobian of H at the image
  centre;
- log-scale = 0.5 * log(det J) at the centre (det > 0 required; else flagged improper);
- centre flow = ||H(c) - c||.

This is the local analogue of the sidecar's rigid increment readout; when two homographies are
compared (VO vs independent pipeline) the same readout is applied to both, so convention deltas
cancel. numpy only.
"""
from __future__ import annotations

import numpy as np


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (n,2) point array."""
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def jacobian_at(H: np.ndarray, x: float, y: float) -> np.ndarray:
    """Analytic 2x2 Jacobian of the homography map at (x, y)."""
    a, b, c = H[0]
    d, e, f = H[1]
    g, h, i = H[2]
    D = g * x + h * y + i
    N1 = a * x + b * y + c
    N2 = d * x + e * y + f
    return np.array([
        [(a * D - N1 * g) / D**2, (b * D - N1 * h) / D**2],
        [(d * D - N2 * g) / D**2, (e * D - N2 * h) / D**2],
    ])


def readout(H: np.ndarray, width: int, height: int) -> dict:
    """Rotation (deg), log-scale, centre flow (px), proper flag — frozen convention."""
    cx, cy = width / 2.0, height / 2.0
    J = jacobian_at(H, cx, cy)
    det = float(np.linalg.det(J))
    proper = det > 0
    U, _s, Vt = np.linalg.svd(J)
    R = U @ Vt
    if np.linalg.det(R) < 0:            # reflection branch: fix the sign convention
        U[:, -1] *= -1
        R = U @ Vt
    rot_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    log_scale = 0.5 * np.log(det) if proper else np.nan
    centre = apply_h(H, np.array([[cx, cy]]))[0]
    flow = float(np.hypot(centre[0] - cx, centre[1] - cy))
    return {"rot_deg": rot_deg, "log_scale": float(log_scale), "flow_px": flow,
            "proper": proper}


def symmetric_transfer_px(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-point symmetric transfer distance (px): RMS of the two directional distances.

    H maps src -> dst. Returns sqrt((||H src - dst||^2 + ||H^-1 dst - src||^2) / 2).
    """
    Hi = np.linalg.inv(H)
    d1 = np.linalg.norm(apply_h(H, src) - dst, axis=1)
    d2 = np.linalg.norm(apply_h(Hi, dst) - src, axis=1)
    return np.sqrt((d1**2 + d2**2) / 2.0)


def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: similarity T so points have zero mean, mean dist sqrt(2)."""
    mean = pts.mean(axis=0)
    d = np.linalg.norm(pts - mean, axis=1).mean()
    s = np.sqrt(2.0) / d if d > 0 else 1.0
    T = np.array([[s, 0, -s * mean[0]], [0, s, -s * mean[1]], [0, 0, 1.0]])
    return (pts - mean) * s, T


def fit_h_ls(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares homography (normalized, 8-parameter h33=1) via normal equations.

    Fast enough for the bootstrap inner loop (O(n) build + 8x8 solve). The h33=1 chart is
    adequate for near-identity pair transforms; None on a singular system.
    """
    if len(src) < 4:
        return None
    sn, Ts = _normalise(src)
    dn, Td = _normalise(dst)
    x, y = sn[:, 0], sn[:, 1]
    u, v = dn[:, 0], dn[:, 1]
    n = len(x)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    A[0::2, 0] = x; A[0::2, 1] = y; A[0::2, 2] = 1
    A[0::2, 6] = -u * x; A[0::2, 7] = -u * y
    b[0::2] = u
    A[1::2, 3] = x; A[1::2, 4] = y; A[1::2, 5] = 1
    A[1::2, 6] = -v * x; A[1::2, 7] = -v * y
    b[1::2] = v
    AtA = A.T @ A
    Atb = A.T @ b
    try:
        h = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        return None
    Hn = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def circular_sd_deg(angles_deg: np.ndarray) -> float:
    """Circular standard deviation in degrees."""
    a = np.radians(angles_deg)
    R = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
    R = min(max(R, 1e-12), 1.0)
    return float(np.degrees(np.sqrt(-2.0 * np.log(R))))
