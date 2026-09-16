"""`EXP-VO-012` Phase 5: two metric families, kept apart because mixing them would be the easiest way
to claim a metric result from a metric that fitted the scale away.

Every headline number this project has published uses **one global Sim(2) alignment** (`DEC-003`),
which removes four gauge freedoms **including scale**. That is right for a trajectory-*shape* claim
and continues to be reported here for continuity. It is **useless** for the question this experiment
asks, because it fits away exactly the quantity the barometer is supposed to supply: under Sim(2)
every arm — fixed height, barometer-aided, oracle — would score almost identically, and a completely
wrong `h₀` would score identically too.

So this module provides three alignments, and the *fewer* parameters an alignment fits, the more it
is trusted:

======================  =====  ===============================================================
`reference_initialised`  **0**  PRIMARY. The synthetic reference frame's position and heading
                                are known exactly, and the estimator's own datum IS that frame
                                (`est = (0,0)`, `θ = 0` at frame 0). So the estimate is placed
                                by construction: translate to GT frame 0, rotate by the known
                                initial heading, fit nothing.
`align_se2`              **3**  Secondary. Fits translation and rotation only — the two things
                                that genuinely are an unobservable initial datum for a local VO
                                with no global origin and no north reference. **Scale is held at
                                exactly 1 and is never fitted.**
`align_sim2`             **4**  Continuity with every prior VO result. Fits scale. **May not be
                                cited in support of a metric claim** and this module's callers
                                label it accordingly.
======================  =====  ===============================================================

`path_length_ratio` carries its own warning: `LIT-VO-003` §10.6 measured that the statistic is
inflated by per-frame noise — 76 % at a one-frame step on real imagery — so the step is a required
argument with no default, and `EXP-VO-012` pre-registered it at 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- alignments

@dataclass(frozen=True)
class Alignment:
    rotation_deg: float
    translation: np.ndarray
    scale: float
    n_fitted: int
    name: str

    def apply(self, xy: np.ndarray) -> np.ndarray:
        a = math.radians(self.rotation_deg)
        r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return self.scale * (np.asarray(xy, float) @ r.T) + self.translation


def _umeyama_rotation(est: np.ndarray, gt: np.ndarray) -> float:
    """Least-squares planar rotation taking centred `est` onto centred `gt` (Umeyama 1991)."""
    e = est - est.mean(axis=0)
    g = gt - gt.mean(axis=0)
    num = float((e[:, 0] * g[:, 1] - e[:, 1] * g[:, 0]).sum())
    den = float((e[:, 0] * g[:, 0] + e[:, 1] * g[:, 1]).sum())
    return math.degrees(math.atan2(num, den))


def align_se2(est: np.ndarray, gt: np.ndarray) -> Alignment:
    """Rotation + translation, **scale fixed at exactly 1**.

    Justified in `DEC-VO-007` and `EXP-VO-012` Phase 5: initial translation and heading are
    unobservable datum for a local VO; scale is not — it is what the external sensor supplies.
    """
    est = np.asarray(est, float)
    gt = np.asarray(gt, float)
    deg = _umeyama_rotation(est, gt)
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = gt.mean(axis=0) - (est.mean(axis=0) @ r.T)
    return Alignment(deg, t, 1.0, 3, "SE(2) (scale fixed at 1)")


def align_sim2(est: np.ndarray, gt: np.ndarray) -> Alignment:
    """Umeyama similarity — the historical alignment. **Fits scale**; shape metrics only."""
    est = np.asarray(est, float)
    gt = np.asarray(gt, float)
    deg = _umeyama_rotation(est, gt)
    e = est - est.mean(axis=0)
    g = gt - gt.mean(axis=0)
    a = math.radians(deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    var = float((e ** 2).sum())
    s = float(((e @ r.T) * g).sum() / var) if var > 0 else 1.0
    t = gt.mean(axis=0) - s * (est.mean(axis=0) @ r.T)
    return Alignment(deg, t, s, 4, "Sim(2) (fits scale — SHAPE ONLY)")


def reference_initialised(est: np.ndarray, gt: np.ndarray, heading0_deg: float) -> Alignment:
    """Zero fitted parameters: place the estimate by construction, not by optimisation.

    The estimator's frame is its own first frame with `θ = 0`; the dataset's first frame has a known
    compass heading. Rotating by that heading and translating to GT frame 0 puts the two in the same
    frame with nothing fitted at all, which is why this is the primary family.
    """
    est = np.asarray(est, float)
    gt = np.asarray(gt, float)
    a = math.radians(heading0_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = gt[0] - (est[0] @ r.T)
    return Alignment(heading0_deg, t, 1.0, 0, "reference-initialised (NOTHING fitted)")


# --------------------------------------------------------------------------- metrics

def path_length(xy: np.ndarray, step: int) -> float:
    """Summed displacement at a **declared** stride.

    `step` has no default on purpose. `LIT-VO-003` §10.6: at `step = 1` this statistic is inflated
    76 % on real imagery and 0.3–0.5 % on noise-free rendered imagery, because it sums `|Δ + noise|`.
    Any scale claim from it has to name the stride it used.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    p = np.asarray(xy, float)[::step]
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def rpe(est: np.ndarray, gt: np.ndarray, lengths_m) -> dict:
    """Relative pose error at fixed ground-truth path lengths. **No alignment at all**, so it is
    scale-sensitive by construction and measures local rather than accumulated error."""
    est = np.asarray(est, float)
    gt = np.asarray(gt, float)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    out = {}
    for length in lengths_m:
        errs = []
        j = 0
        for i in range(len(gt)):
            while j < len(gt) and dist[j] - dist[i] < length:
                j += 1
            if j >= len(gt):
                break
            de = est[j] - est[i]
            dg = gt[j] - gt[i]
            # Rotate the estimated increment into the GT frame by the local heading difference, the
            # standard construction: compare the DISPLACEMENT VECTORS after removing the segment's
            # own start orientation, so a heading error does not double-count as a length error.
            ang = math.atan2(dg[1], dg[0]) - math.atan2(de[1], de[0])
            c, s = math.cos(ang), math.sin(ang)
            de_rot = np.array([c * de[0] - s * de[1], s * de[0] + c * de[1]])
            errs.append(float(np.linalg.norm(de_rot - dg)))
        out[f"rpe_{int(length)}m"] = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    return out


def score(est_xy: np.ndarray, gt_xy: np.ndarray, heading0_deg: float, *,
          path_step: int, rpe_lengths=(10, 25, 50, 100)) -> dict:
    """Both families for one arm, with every alignment's fitted-parameter count carried along."""
    est = np.asarray(est_xy, float)
    gt = np.asarray(gt_xy, float)
    gt_path = path_length(gt, 1)
    out = {"gt_path_length_m": gt_path, "n_points": int(len(gt)),
           "path_length_step_frames": int(path_step)}

    for key, al in (("ref_init", reference_initialised(est, gt, heading0_deg)),
                    ("se2", align_se2(est, gt)),
                    ("sim2", align_sim2(est, gt))):
        a = al.apply(est)
        err = np.linalg.norm(a - gt, axis=1)
        out[key] = {
            "alignment": al.name,
            "n_fitted_parameters": al.n_fitted,
            "alignment_scale": al.scale,
            "alignment_rotation_deg": al.rotation_deg,
            "ate_rmse_m": float(np.sqrt(np.mean(err ** 2))),
            "ate_rmse_normalised": float(np.sqrt(np.mean(err ** 2)) / gt_path) if gt_path else float("nan"),
            "endpoint_error_m": float(err[-1]),
            "endpoint_error_pct_of_path": float(100.0 * err[-1] / gt_path) if gt_path else float("nan"),
            "max_error_m": float(err.max()),
        }

    # Scale-sensitive statistics that need no alignment at all.
    out["path_length_ratio"] = (path_length(est, path_step) / path_length(gt, path_step)
                                if path_length(gt, path_step) > 0 else float("nan"))
    out["path_length_ratio_step1"] = (path_length(est, 1) / path_length(gt, 1)
                                      if gt_path > 0 else float("nan"))
    out["rpe"] = rpe(reference_initialised(est, gt, heading0_deg).apply(est), gt, rpe_lengths)
    return out


def error_vs_distance(est_xy: np.ndarray, gt_xy: np.ndarray, heading0_deg: float):
    """The (travelled distance, horizontal metric error) curve — required figure 2."""
    est = np.asarray(est_xy, float)
    gt = np.asarray(gt_xy, float)
    a = reference_initialised(est, gt, heading0_deg).apply(est)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))])
    return dist, np.linalg.norm(a - gt, axis=1)
