"""Canonical coordinate and heading conventions (DEC-004).

Canonical position frame: **ENU** — x East, y North, z Up, right-handed, relative to a
declared local tangent-plane origin.

Canonical heading: **compass-style** — degrees clockwise from North, normalised to
[0, 360). This is a deliberate hybrid with the ENU position axes (see DEC-004): a unit
heading vector at heading psi is ``(sin(psi), cos(psi))`` in (East, North), *not* the
``(cos, sin)`` a reader might expect from ENU's own axes. Every conversion here is a
separately tested step (FR-064) precisely because that mismatch is a documented,
easy-to-get-wrong source of sign errors (see gap G13a and DEC-004's rationale).

No source format's convention is assumed downstream of this module — geodetic, NED, and
raw estimator yaw are all converted here, once, at the ingest/capture boundary.
"""

from __future__ import annotations

import math

# WGS84 semi-major axis, metres. Used for a flat-earth (equirectangular) tangent-plane
# approximation, adequate for the short-range UAV flights this project targets (a few
# kilometres at most). Not a full ellipsoidal geodesy implementation — sufficient here,
# and cheap to replace behind this function if longer-range flights ever require it.
_WGS84_SEMI_MAJOR_AXIS_M = 6378137.0


def geodetic_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_alt_m: float,
) -> tuple[float, float, float]:
    """Convert geodetic (lat, lon, alt) to local ENU metres relative to an origin.

    Flat-earth tangent-plane approximation: latitude offset scales directly by the
    WGS84 semi-major axis, longitude offset is additionally scaled by cos(origin
    latitude) to account for meridian convergence. Matches DEC-004 / FR-006.
    """
    origin_lat_rad = math.radians(origin_lat_deg)
    d_lat_rad = math.radians(lat_deg - origin_lat_deg)
    d_lon_rad = math.radians(lon_deg - origin_lon_deg)

    north_m = d_lat_rad * _WGS84_SEMI_MAJOR_AXIS_M
    east_m = d_lon_rad * _WGS84_SEMI_MAJOR_AXIS_M * math.cos(origin_lat_rad)
    up_m = alt_m - origin_alt_m

    return east_m, north_m, up_m


def ned_to_enu(north_m: float, east_m: float, down_m: float) -> tuple[float, float, float]:
    """Convert NED (north, east, down) to canonical ENU (east, north, up).

    A pure axis relabelling: swap the first two axes and negate the third. Verified
    explicitly (FR-064) rather than assumed, because getting this backwards silently
    reflects every trajectory through the flight path.
    """
    return east_m, north_m, -down_m


def normalize_heading_deg(heading_deg: float) -> float:
    """Normalise a heading to the canonical range [0, 360).

    Guards a genuine floating-point edge case: for an input whose true value modulo
    360 is a tiny negative number close to zero (e.g. -1e-14), Python's `%` can round
    the result up to exactly 360.0 rather than just under it, because 360.0's
    representable precision (~5.7e-14) is coarser than the residual being corrected
    for. Left unguarded, this silently violates the [0, 360) contract enforced
    downstream (runrecord.py rejects est_yaw_deg == 360.0) -- found via alignment.py's
    rotation_deg computation hitting exactly this case during test development.
    """
    result = heading_deg % 360.0
    if result >= 360.0:
        result -= 360.0
    return result


def heading_to_unit_vector(heading_deg: float) -> tuple[float, float]:
    """Canonical compass heading -> unit vector (east, north).

    Compass convention: 0 deg = North = (0, 1); 90 deg = East = (1, 0); clockwise
    positive. Hence (east, north) = (sin(psi), cos(psi)) -- the "reversed" pair
    relative to the ENU-math convention, and the crux of DEC-004's hybrid choice.
    """
    psi = math.radians(heading_deg)
    return math.sin(psi), math.cos(psi)


def unit_vector_to_heading_deg(east: float, north: float) -> float:
    """Unit vector (east, north) -> canonical compass heading in [0, 360)."""
    psi_deg = math.degrees(math.atan2(east, north))
    return normalize_heading_deg(psi_deg)


def compass_to_enu_math_deg(compass_deg: float) -> float:
    """Compass heading (CW from North) -> ENU-math angle (CCW from East).

    Self-inverse: applying it twice returns the original value (mod 360). Provided so
    the relationship DEC-004 documents is checkable in code, not only in prose.
    """
    return normalize_heading_deg(90.0 - compass_deg)


def enu_math_to_compass_deg(math_deg: float) -> float:
    """ENU-math angle (CCW from East) -> compass heading (CW from North). Self-inverse."""
    return normalize_heading_deg(90.0 - math_deg)


def shortest_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Shortest signed angular difference a - b, wrapped to (-180, 180].

    The building block for every yaw-error computation and for heading interpolation.
    A naive subtraction breaks at the 0/360 wrap (e.g. 1 deg vs 359 deg is a 2 deg
    difference, not 358).
    """
    diff = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    # Python's % can return -180.0 exactly at the boundary; canonicalise to +180.0
    # so the range is the conventional (-180, 180].
    if diff <= -180.0:
        diff += 360.0
    return diff


def interpolate_heading_deg(h0_deg: float, h1_deg: float, frac: float) -> float:
    """Shortest-arc interpolation between two headings at fraction frac in [0, 1].

    Linear interpolation of the raw degree values is wrong across the wrap (e.g.
    interpolating 350 deg -> 10 deg linearly passes through 180 deg, the opposite
    direction from the true 20 deg-wide shortest arc). Interpolating along the
    shortest-angle difference is correct by construction.
    """
    delta = shortest_angle_diff_deg(h1_deg, h0_deg)
    return normalize_heading_deg(h0_deg + frac * delta)
