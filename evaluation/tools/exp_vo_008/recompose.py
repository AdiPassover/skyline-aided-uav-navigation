"""Re-derive every navigation readout offline from ONE recorded transform sequence (`EXP-VO-008`).

`EXP-VO-007` found legacy affine beating `RIGID_MOTION` affine on `AMtown01`, reversing
`HKairport01`. Gate G0 of this experiment showed the two runs' recorded logical transforms are
bit-identical, so the difference is entirely in how those transforms are *read*. This module makes
the readouts comparable by deriving all of them from the same `g00..g22` columns.

**The insight the code reading gave, which this module operationalises.** `MOSAIC_LEGACY`,
`RIGID_MOTION` and `LOGICAL_FRAME` are not three different architectures; they are one computation
at three granularities. All three integrate

    T_m = T_i + R(theta_i) * (S_{i->m}(c) - c)        theta_m = theta_i + rot(S_{i->m})

where `S_{i->m} : C_m -> C_i` is the composed transform since the last **reduction boundary** `i`,
and `c` is the image centre. What differs is only where the boundaries are:

- `RIGID_MOTION`  -> a boundary at **every frame** (N = 1);
- `MOSAIC_LEGACY` -> a boundary at every **mosaic re-origin** (~146 frames on `AMtown01` affine);
- `LOGICAL_FRAME` -> **no boundaries at all** (N = infinity), so the non-rigid part compounds.

Between boundaries the full transform is composed, so scale and shear act; at a boundary the state
is re-expressed rigidly and they are dropped. Granularity is therefore a **bias/variance dial**, and
`schedule_ate` sweeps it.

The second axis is which rotation estimator supplies `rot(S)`: the polar rotation of the Jacobian at
the image centre (what `RigidMotionDecomposition` uses) or the frame quad's top-edge angle (what the
legacy path uses, `COMP-001` section 3.6).

Nothing here is fitted. The schedules are fixed in the experiment record before execution, and the
recenter schedule is the estimator's own, taken from the recorded `event` column.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

G_COLS = ("g00", "g01", "g02", "g10", "g11", "g12", "g20", "g21", "g22")


@dataclass
class Recording:
    """One run's recorded logical transforms and events."""
    frame_index: np.ndarray
    G: np.ndarray               # (n, 3, 3), logical L -> C_k
    events: list[str]
    inc_log_scale: np.ndarray
    inc_anisotropy: np.ndarray

    def __len__(self) -> int:
        return self.G.shape[0]


def load_recording(run_dir: Path | str) -> Recording:
    with (Path(run_dir) / "logical_transform.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    G = np.empty((n, 3, 3))
    for i, r in enumerate(rows):
        m = np.array([float(r[k]) for k in G_COLS]).reshape(3, 3)
        G[i] = m / m[2, 2]      # normalise; the affine runs already have g22 == 1
    return Recording(
        frame_index=np.array([int(r["frame_index"]) for r in rows]),
        G=G,
        events=[r["event"] for r in rows],
        inc_log_scale=np.array([float(r["inc_log_scale"]) for r in rows]),
        inc_anisotropy=np.array([float(r["inc_anisotropy"]) for r in rows]),
    )


def _apply(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    w = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    return ((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / w,
            (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / w)


def _jacobian(M: np.ndarray, x: float, y: float) -> np.ndarray:
    """2x2 Jacobian of the projective map at `(x, y)` — `LIT-VO-003` eq. (5).

    Reduces to the linear block for an affine transform (bottom row 0, 0, 1), so the same code
    serves both motion models exactly as `MotionModelSupport` does in Java.
    """
    w = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    px, py = _apply(M, x, y)
    return np.array([
        [(M[0, 0] - px * M[2, 0]) / w, (M[0, 1] - px * M[2, 1]) / w],
        [(M[1, 0] - py * M[2, 0]) / w, (M[1, 1] - py * M[2, 1]) / w],
    ])


def polar_rotation(J: np.ndarray) -> float:
    """Nearest proper rotation in Frobenius norm — the same closed form as `RigidMotionDecomposition`."""
    return math.atan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1])


def edge_angle(M: np.ndarray, w: float, h: float) -> float:
    """Top-edge angle of the mapped frame quad — the legacy path's rotation estimator.

    `MotionModelStitchingEstimator.edgeAngleDegrees` takes `atan2(b.y - a.y, b.x - a.x)` over the
    quad's first two corners, i.e. the image's top edge from (0,0) to (W,0).
    """
    ax, ay = _apply(M, 0.0, 0.0)
    bx, by = _apply(M, w, 0.0)
    return math.atan2(by - ay, bx - ax)


def uniform_scale(J: np.ndarray) -> float:
    """`sqrt|det J|` — `DEC-VO-004`'s scale observable, multiplicative under composition."""
    return math.sqrt(abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]))


def boundaries_every(n: int, N: int) -> list[int]:
    """Reduction boundaries every `N` frames. `N >= n` gives a single segment (= FULL_LOGICAL)."""
    if N <= 0:
        raise ValueError("N must be positive")
    return list(range(0, n, N))


def boundaries_from_events(rec: Recording, kinds=("recenter", "restart")) -> list[int]:
    """Reduction boundaries at the estimator's own recorded events — the legacy schedule.

    GT-independent by construction: the event column is what the estimator did, not something
    chosen here.
    """
    b = [0] + [i for i, e in enumerate(rec.events) if e in kinds]
    return sorted(set(b))


def integrate(rec: Recording, boundaries: list[int], width: int, height: int,
              rotation: str = "polar", apply_scale: bool = False,
              external_scale: np.ndarray | None = None) -> dict:
    """Integrate the readout family at the given reduction boundaries.

    `apply_scale=True` additionally divides each segment displacement by the accumulated visual
    scale, i.e. normalises the pixel units back to the first frame's — `DEC-VO-004` alternative B.
    `external_scale`, when given, is used in place of the visual scale (the Phase-5 known-answer
    substitution). Both are **diagnostics**; neither is a production path.
    """
    n = len(rec)
    cx, cy = width / 2.0, height / 2.0
    bset = sorted(set(boundaries) | {0})
    seg_of = np.zeros(n, dtype=int)
    for k in range(1, len(bset)):
        seg_of[bset[k]:] = k

    Ginv = np.linalg.inv(rec.G)
    x = np.zeros(n)
    y = np.zeros(n)
    yaw = np.zeros(n)

    Tx = Ty = theta = 0.0
    anchor = 0                      # index of the current reduction boundary
    cum_log_scale = 0.0             # accumulated visual scale at the anchor (for apply_scale)
    for m in range(1, n):
        if seg_of[m] != seg_of[m - 1]:
            # close the previous segment at frame m-1, then re-anchor
            Tx, Ty, theta, cum_log_scale = _close(
                rec, Ginv, anchor, m - 1, cx, cy, Tx, Ty, theta, rotation,
                apply_scale, external_scale, cum_log_scale)
            anchor = m - 1
        px, py, dth = _segment(rec, Ginv, anchor, m, cx, cy, rotation)
        s = _scale_divisor(apply_scale, external_scale, cum_log_scale, anchor)
        c, si = math.cos(theta), math.sin(theta)
        x[m] = Tx + (c * px - si * py) / s
        y[m] = Ty + (si * px + c * py) / s
        yaw[m] = theta + dth
    return {
        "x": x, "y": -y,            # Pose3D's image-down -> pose-forward convention, as in Java
        "yaw_deg": np.degrees(yaw) % 360.0,
        "n_boundaries": len(bset),
        "boundaries": bset,
    }


def _scale_divisor(apply_scale, external_scale, cum_log_scale, anchor) -> float:
    if external_scale is not None:
        return float(external_scale[anchor])
    if apply_scale:
        return math.exp(cum_log_scale)
    return 1.0


def _segment(rec, Ginv, i, m, cx, cy, rotation):
    """Rigid part of `S_{i->m} : C_m -> C_i`, evaluated at the image centre."""
    S = rec.G[i] @ Ginv[m]
    qx, qy = _apply(S, cx, cy)
    if rotation == "polar":
        dth = polar_rotation(_jacobian(S, cx, cy))
    elif rotation == "edge":
        dth = edge_angle(S, 2 * cx, 2 * cy)
    else:
        raise ValueError(rotation)
    return qx - cx, qy - cy, dth


def _close(rec, Ginv, i, j, cx, cy, Tx, Ty, theta, rotation,
           apply_scale, external_scale, cum_log_scale):
    px, py, dth = _segment(rec, Ginv, i, j, cx, cy, rotation)
    s = _scale_divisor(apply_scale, external_scale, cum_log_scale, i)
    c, si = math.cos(theta), math.sin(theta)
    Tx += (c * px - si * py) / s
    Ty += (si * px + c * py) / s
    theta += dth
    if apply_scale:
        S = rec.G[i] @ Ginv[j]
        cum_log_scale += math.log(uniform_scale(_jacobian(S, cx, cy)))
    return Tx, Ty, theta, cum_log_scale
