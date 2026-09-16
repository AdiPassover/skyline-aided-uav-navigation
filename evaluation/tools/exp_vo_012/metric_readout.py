"""`EXP-VO-012`: the barometer-aided metric readout — `DEC-VO-007` D3, and nothing more.

One scalar per frame, multiplied into an increment `RIGID_MOTION` already produces:

    h_k    = h0 + h_baro(t_{k-1})                       # the REFERENCE frame's height
    gsd_k  = h_k / f_working                            # f_working = fx_native / downsampleFactor
    T_k    = T_{k-1} + R(theta_{k-1}) * (q_k - c) * gsd_k
    theta_k= theta_{k-1} + polarRotation(J(D_k^-1)(c))  # UNCHANGED, and not constrained here

Derived and verified in `LIT-VO-003` §10. Three things are easy to get wrong and all three are
checked by test rather than asserted here: the height belongs to frame `k−1` (the increment is
expressed in *that* frame's pixels), `f` is the **working-resolution** focal length, and the axes map
as `east = +T_x`, `north = −T_y`.

**Three arms, one code path.** They differ only in which array fills `h_agl`:

===============  =========================================================================
`fixed`          `h0` for every frame. What the system does today, made metric with a
                 declared constant. `EXP-VO-005` R4 measured its cost at 10–23 % of path.
`baro`           `h0 + h_baro(t)` — the design under test. Flat-terrain assumption.
`oracle`         the renderer's true `h_AGL`. DIAGNOSTIC ONLY: no sensor on the author's
                 platform supplies it at survey height (`LIT-VO-006` §2). It also measures
                 what a rangefinder would be worth.
===============  =========================================================================

**Visual scale has no authority here and cannot acquire any.** `lambda` never appears in the
computation below; it is carried alongside purely so Phase 7 can plot what the three channels
disagree about. `test_exp_vo_012.py` gate G4 asserts the output is bit-identical when the visual
series is replaced by arbitrary values — `DEC-VO-007` D4 enforced rather than intended.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVALUATION_DIR = HERE.parents[1]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(EVALUATION_DIR / "tools" / "exp_vo_008"))

import recompose as rc                                                   # noqa: E402

G_COLS = rc.G_COLS


@dataclass
class Increments:
    """Per-frame rigid increments, read once from a committed run record.

    `dq` is `(q_k − c)` in the REFERENCE frame's pixels; `dtheta` the polar rotation of the same
    inverse transform. Both are exactly what `RigidNavigationState` computes in Java — `recompose`
    reproduces the live readout to 2.8 × 10⁻¹¹ px (`VO_REPRODUCIBILITY.md` §5).
    """

    frame_index: np.ndarray
    events: list[str]
    dq: np.ndarray              # (n, 2), pixels, dq[0] = 0
    dtheta: np.ndarray          # (n,), radians, dtheta[0] = 0
    inc_log_scale: np.ndarray   # (n,) DIAGNOSTIC ONLY — never used below
    timestamps_s: np.ndarray

    def __len__(self) -> int:
        return self.dq.shape[0]


def load_increments(run_dir: Path | str, width: int, height: int) -> Increments:
    run_dir = Path(run_dir)
    with (run_dir / "logical_transform.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    g = np.empty((n, 3, 3))
    for i, r in enumerate(rows):
        m = np.array([float(r[k]) for k in G_COLS]).reshape(3, 3)
        g[i] = m / m[2, 2]
    cx, cy = width / 2.0, height / 2.0
    g_inv = np.linalg.inv(g)
    dq = np.zeros((n, 2))
    dtheta = np.zeros(n)
    for k in range(1, n):
        s = g[k - 1] @ g_inv[k]                       # C_k -> C_{k-1}
        qx, qy = rc._apply(s, cx, cy)
        dq[k] = (qx - cx, qy - cy)
        dtheta[k] = rc.polar_rotation(rc._jacobian(s, cx, cy))

    # The sidecar carries inc_log_scale only on runs captured after EXP-VO-006; recompute it from
    # the transforms otherwise, so both sidecar generations are readable.
    if "inc_log_scale" in rows[0]:
        inc_log = np.array([float(r["inc_log_scale"]) for r in rows])
    else:
        inc_log = np.zeros(n)
        for k in range(1, n):
            s = g[k - 1] @ g_inv[k]
            inc_log[k] = math.log(rc.uniform_scale(rc._jacobian(s, cx, cy)))

    ts = np.array([], float)
    fpath = run_dir / "frames.csv"
    if fpath.exists():
        with fpath.open(newline="") as f:
            by_idx = {int(r["frame_index"]): float(r["timestamp_s"]) for r in csv.DictReader(f)}
        ts = np.array([by_idx[int(r["frame_index"])] for r in rows])

    return Increments(frame_index=np.array([int(r["frame_index"]) for r in rows]),
                      events=[r["event"] for r in rows], dq=dq, dtheta=dtheta,
                      inc_log_scale=inc_log, timestamps_s=ts)


@dataclass
class MetricTrack:
    east_m: np.ndarray
    north_m: np.ndarray
    yaw_deg: np.ndarray
    gsd_m_per_px: np.ndarray
    h_used_m: np.ndarray
    valid: np.ndarray
    arm: str

    def xy(self) -> np.ndarray:
        return np.stack([self.east_m, self.north_m], axis=1)


def integrate_metric(inc: Increments, h_agl_m: np.ndarray, f_working_px: float,
                     arm: str = "baro", valid: np.ndarray | None = None) -> MetricTrack:
    """`DEC-VO-007` D3, verbatim. One multiply per frame; nothing else changes.

    `h_agl_m[k]` is the camera-to-imaged-surface height at frame `k`; the increment into frame `k`
    is converted with `h_agl_m[k-1]`, because `(q_k − c)` is expressed in frame `k−1`'s pixels
    (`LIT-VO-003` §10.1). Getting this backwards costs a factor of `h_k/h_{k−1}` per frame, which is
    small, systematic, and measurable — `test_exp_vo_012.py` pins it.

    A frame whose height is invalid (`valid=False`, i.e. beyond `tau_stale`) still integrates, using
    the last usable height, and is flagged. Refusing to produce a pose would make the arms
    non-comparable; flagging lets a consumer see exactly which frames are degraded, which is what
    `DEC-VO-007` D6 asks for.
    """
    n = len(inc)
    h = np.asarray(h_agl_m, float).copy()
    if h.shape != (n,):
        raise ValueError(f"h_agl_m has shape {h.shape}, expected ({n},)")
    ok = np.ones(n, bool) if valid is None else np.asarray(valid, bool).copy()

    # Carry the last usable height forward across invalid frames (and across NaN, which `baro.py`
    # writes beyond `tau_stale`). The FLAG is what marks them, not a silent substitution.
    last = None
    for k in range(n):
        if ok[k] and np.isfinite(h[k]):
            last = h[k]
        elif last is not None:
            h[k] = last
            ok[k] = False
        else:
            raise ValueError("no usable height at or before the reference frame")
    if not np.all(np.isfinite(h)):
        raise ValueError("height series still has non-finite values after hold")
    if np.any(h <= 0):
        raise ValueError("non-positive camera height")

    gsd = h / f_working_px
    east = np.zeros(n)
    north = np.zeros(n)
    yaw = np.zeros(n)
    tx = ty = th = 0.0
    for k in range(1, n):
        g = gsd[k - 1]                                   # the REFERENCE frame's ground sampling
        px, py = inc.dq[k, 0] * g, inc.dq[k, 1] * g
        c, s = math.cos(th), math.sin(th)
        tx += c * px - s * py
        ty += s * px + c * py
        th += inc.dtheta[k]
        east[k], north[k], yaw[k] = tx, -ty, th          # LIT-VO-003 eq. (10)
    return MetricTrack(east_m=east, north_m=north, yaw_deg=np.degrees(yaw) % 360.0,
                       gsd_m_per_px=gsd, h_used_m=h, valid=ok, arm=arm)


def h_agl_from_baro(h0_m: float, h_baro_m: np.ndarray) -> np.ndarray:
    """`LIT-VO-006` eq. (4) under the flat-terrain assumption. The assumption is the whole point:
    where terrain moves, this is wrong by `−e(x_c(t))` and nothing in the inputs can tell."""
    return h0_m + np.asarray(h_baro_m, float)


def f_working(fx_native_px: float, downsample_factor: int) -> float:
    """`LIT-VO-003` eq. (9). Forgetting the divisor halves every metric distance."""
    if downsample_factor < 1:
        raise ValueError("downsampleFactor must be >= 1")
    return fx_native_px / float(downsample_factor)
