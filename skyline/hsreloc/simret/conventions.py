"""The one place UE units and axes become project units and axes (PROT-SKY-001 §1.1, PROT-001 §2).

Every centimetre→metre division and every axis relabelling in the simulator path goes through this
module. That is the "explicit documented seam" the brief asks for: if the declaration turns out to be
wrong, exactly one file is wrong, and the fix is a re-conversion of the recorded raw values rather
than a re-render (PROT-001 §3.3).

What is *declared* here
-----------------------
``PROT-001`` §2.2 asks the UE project to declare **world +X = North, +Y = East**. Under that
declaration the conversion is a relabelling::

    east_m  = Y_ue_cm / 100
    north_m = X_ue_cm / 100
    up_m    = (Z_ue_cm - datum_cm) / 100
    yaw_deg = Yaw_ue mod 360                (compass, clockwise from North — DEC-004)

and nothing needs a sign flip: UE's yaw 0 faces +X = North and increases toward +Y = East, which *is*
compass. The swap of two axes also flips handedness, so left-handed UE becomes right-handed ENU. No
other declaration is accepted, and the refusal message says why: mapping +X to East instead would be
an even permutation and would leave the result **left**-handed, so it is not a relabelling but a
reflection, and a reflection cannot be corrected downstream by any sign convention.

What is *verified* rather than assumed
--------------------------------------
Field names are not evidence. Three things are checked against the data itself:

* **the quaternion convention** — ``identify_quaternion_convention`` reconstructs a quaternion from
  the logged Euler angles under each of four candidate conventions and reports which one (if any)
  reproduces the logged quaternions. A run whose Euler angles and quaternions disagree under every
  candidate is refused, and the report names the residual for each candidate so the simulator owner
  can see *which* half is unexpected;
* **the heading lock** — a session declaring ``heading_mode: world_locked_north`` must actually carry
  a constant compass yaw of 0. If it carries a constant 90 instead, the axis declaration above is not
  what was rendered, and the refusal says exactly that;
* **the camera calibration** — ``fx`` against ``width/2 / tan(hfov/2)``, and the principal point
  against the image centre.

Pitch and roll signs (positive = optical axis up, positive = image rotates clockwise) remain
**declared and unverified** until the ``PROT-SKY-001`` §4 R0-S renders exist. They are carried in
``SimConventions.verified_by`` and stamped into every session, so no downstream record can claim a
verification that never happened.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

CONVENTIONS_VERSION = "1.0.0"

#: The only accepted UE world-axis declaration (PROT-001 §2.2).
AXES_X_NORTH_Y_EAST = "x_north_y_east"

CM_PER_M = 100.0

#: Quaternion conventions a simulator export might plausibly be logging. The adapter does not choose
#: one; it identifies which of these reproduces the logged values, and refuses if none does.
QUATERNION_CONVENTIONS = ("ue_frotator", "ue_frotator_conjugate", "zyx_right_handed",
                          "zyx_right_handed_conjugate")


class ConventionError(Exception):
    """A declared convention is unsupported, or the data contradicts the declaration."""


# --------------------------------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SimConventions:
    """Everything the adapter must be *told* rather than guess, plus what verified it.

    ``altitude_datum_ue_cm`` is the UE world Z of the ground plane at the trajectory origin
    (PROT-001 §2.4). It is required and explicit: "height above terrain" and raw world Z are both
    level-dependent, so a datum chosen silently makes altitudes incomparable between runs.

    ``verified_by`` names the convention-check recording (PROT-SKY-001 §4) that established the pitch
    and roll signs. ``None`` means they are declared but unverified — legal, recorded as such, and
    loudly reported; it is never silently upgraded.
    """

    ue_world_axes: str = AXES_X_NORTH_Y_EAST
    altitude_datum_ue_cm: float = 0.0
    altitude_datum: str = "ue_world_z_zero"
    pitch_sign: str = "positive_up"
    roll_sign: str = "positive_clockwise_image"
    attitude_order: str = "ZYX"
    quaternion_convention: Optional[str] = None      # None = identify it from the data
    verified_by: Optional[str] = None

    def validate(self) -> None:
        if self.ue_world_axes != AXES_X_NORTH_Y_EAST:
            raise ConventionError(
                f"unsupported UE world-axis declaration {self.ue_world_axes!r}. Only "
                f"{AXES_X_NORTH_Y_EAST!r} (PROT-001 §2.2: world +X = North, +Y = East) is accepted: "
                f"ENU = (Y, X, Z)_ue swaps two axes, which flips UE's left-handedness to ENU's "
                f"right-handedness and makes compass yaw equal to UE yaw with no offset. Mapping +X "
                f"to East instead would be an even permutation, leaving the frame left-handed — that "
                f"is a reflection, not a relabelling, and no downstream sign convention repairs it. "
                f"If the simulator really renders another declaration, it must be changed there or "
                f"this module must gain a second, separately verified mapping.")
        if self.quaternion_convention is not None and self.quaternion_convention not in QUATERNION_CONVENTIONS:
            raise ConventionError(f"unknown quaternion convention {self.quaternion_convention!r}; "
                                  f"expected one of {QUATERNION_CONVENTIONS} or null (identify it)")
        if not math.isfinite(float(self.altitude_datum_ue_cm)):
            raise ConventionError("altitude_datum_ue_cm must be finite")

    @property
    def attitude_verified(self) -> bool:
        return bool(self.verified_by)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["conventions_version"] = CONVENTIONS_VERSION
        d["frame_convention"] = "ENU"
        d["heading_convention"] = "compass_cw_from_north"
        d["length_unit_seam"] = "centimetres divided by 100 in hsreloc.simret.conventions"
        d["attitude_verified"] = self.attitude_verified
        return d

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "SimConventions":
        raw = dict(raw or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConventionError(f"unknown convention keys {unknown}; expected a subset of {sorted(known)}")
        c = cls(**raw)
        c.validate()
        return c


# --------------------------------------------------------------------------------------------------
# unit and axis conversion — the seam
# --------------------------------------------------------------------------------------------------

def cm_to_m(value_cm: float) -> float:
    """The only centimetre→metre conversion in the simulator path."""
    return float(value_cm) / CM_PER_M


def normalise_0_360(deg: float) -> float:
    """Compass degrees in [0, 360)."""
    return float(deg) % 360.0


def ue_position_to_enu(x_cm: float, y_cm: float, z_cm: float,
                       conventions: SimConventions) -> tuple:
    """UE world centimetres → ENU metres ``(east, north, up)``; up is relative to the declared datum."""
    conventions.validate()
    return (cm_to_m(y_cm), cm_to_m(x_cm),
            cm_to_m(float(z_cm) - float(conventions.altitude_datum_ue_cm)))


def ue_attitude_to_enu(yaw_deg: float, pitch_deg: float, roll_deg: float,
                       conventions: SimConventions) -> tuple:
    """UE rotator degrees → ``(compass yaw [0,360), pitch, roll)``.

    Under the accepted declaration this is a relabelling of yaw and a pass-through of pitch and roll:
    UE's positive pitch is nose-up (the project's ``positive_up``) and positive roll is right-side-down
    (the project's ``positive_clockwise_image``). Both remain *unverified* until the R0-S renders.
    """
    conventions.validate()
    return (normalise_0_360(yaw_deg), float(pitch_deg), float(roll_deg))


# --------------------------------------------------------------------------------------------------
# quaternions — reconstruction, comparison, and convention identification
# --------------------------------------------------------------------------------------------------

def _half(deg: float) -> tuple:
    r = math.radians(float(deg)) * 0.5
    return math.sin(r), math.cos(r)


def ue_rotator_to_quaternion(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """UE ``FRotator::Quaternion()`` as ``(w, x, y, z)``.

    Transcribed from the engine's documented formula. It is *not* trusted on that basis: it is one of
    the candidates ``identify_quaternion_convention`` tests against the export's own quaternion
    columns, and the adapter refuses a run in which no candidate matches.
    """
    sp, cp = _half(pitch_deg)
    sy, cy = _half(yaw_deg)
    sr, cr = _half(roll_deg)
    return np.array([
        cr * cp * cy + sr * sp * sy,        # w
        cr * sp * sy - sr * cp * cy,        # x
        -cr * sp * cy - sr * cp * sy,       # y
        cr * cp * sy - sr * sp * cy,        # z
    ], dtype=np.float64)


def zyx_right_handed_quaternion(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """The textbook right-handed intrinsic Z-Y′-X″ quaternion as ``(w, x, y, z)``.

    The alternative an exporter written outside the engine's own maths is most likely to produce.
    """
    sy, cy = _half(yaw_deg)
    sp, cp = _half(pitch_deg)
    sr, cr = _half(roll_deg)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def quaternion_conjugate(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _builder(name: str):
    if name == "ue_frotator":
        return ue_rotator_to_quaternion
    if name == "ue_frotator_conjugate":
        return lambda y, p, r: quaternion_conjugate(ue_rotator_to_quaternion(y, p, r))
    if name == "zyx_right_handed":
        return zyx_right_handed_quaternion
    if name == "zyx_right_handed_conjugate":
        return lambda y, p, r: quaternion_conjugate(zyx_right_handed_quaternion(y, p, r))
    raise ConventionError(f"unknown quaternion convention {name!r}")


def quaternion_norm_error(q) -> float:
    """``| |q| - 1 |`` — how far from unit length a logged quaternion is."""
    q = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        return float("inf")
    return abs(float(np.linalg.norm(q)) - 1.0)


def quaternion_angle_deg(a, b) -> float:
    """Rotation angle between two unit quaternions, in degrees, sign-ambiguity removed.

    ``q`` and ``-q`` are the same rotation, so the comparison is on ``|a·b|``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf")
    dot = float(np.clip(abs(float(a @ b) / (na * nb)), 0.0, 1.0))
    return float(2.0 * math.degrees(math.acos(dot)))


def identify_quaternion_convention(samples: list, tolerance_deg: float = 0.5) -> dict:
    """Which candidate convention reproduces the logged quaternions?

    ``samples`` is a list of ``(yaw_deg, pitch_deg, roll_deg, (w, x, y, z))`` taken from the export's
    own columns. Returns the residual per candidate and the identified convention (or ``None``).

    A run in which *several* candidates fit is reported too — that happens legitimately when every
    logged attitude is identity-like (yaw 0, pitch 0, roll 0 reproduces under all four), and the
    caller must not then claim the convention has been established. ``discriminating`` says whether
    the sample set could tell the candidates apart at all.
    """
    if not samples:
        raise ConventionError("no attitude samples to identify a quaternion convention from")
    residuals = {}
    for name in QUATERNION_CONVENTIONS:
        build = _builder(name)
        worst = 0.0
        for yaw, pitch, roll, q in samples:
            worst = max(worst, quaternion_angle_deg(build(yaw, pitch, roll), q))
        residuals[name] = worst
    fits = sorted(n for n, r in residuals.items() if r <= tolerance_deg)
    spread = max(residuals.values()) - min(residuals.values())
    return {
        "residual_deg": residuals,
        "tolerance_deg": float(tolerance_deg),
        "fits": fits,
        "identified": fits[0] if len(fits) == 1 else (fits[0] if fits else None),
        "ambiguous": len(fits) > 1,
        "discriminating": bool(spread > tolerance_deg),
        "n_samples": len(samples),
    }
