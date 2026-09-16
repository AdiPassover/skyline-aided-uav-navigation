"""Ingestion core: geodetic->ENU normalization, derived heading, and route-stratified candidate
sampling (spec 007). Dataset-agnostic -- a Nordland adapter and a future sim/flight adapter both feed
these functions. No matcher/ranking anywhere; sampling only *selects* frames.

ENU conversion reuses ``naveval.frames`` (DEC-004). Heading is *derived* from the GPS track bearing
(the train/vehicle direction between consecutive fixes) and marked as such -- the source cameras carry no
measured yaw. FOV is left unknown by design.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

from naveval.frames import geodetic_to_enu, normalize_heading_deg

from hsreloc.observation import Observation


def compute_origin(observations: list[Observation]) -> tuple[float, float, float]:
    """Local-frame origin = the GPS centroid of the observations (DEC-004 local tangent plane)."""
    lats = [o.lat for o in observations if o.lat is not None]
    lons = [o.lon for o in observations if o.lon is not None]
    alts = [o.alt_m for o in observations if o.alt_m is not None]
    if not lats or not lons:
        raise ValueError("cannot compute origin: observations carry no lat/lon")
    return (sum(lats) / len(lats), sum(lons) / len(lons), (sum(alts) / len(alts)) if alts else 0.0)


def normalize_to_enu(observations: list[Observation],
                     origin: Optional[tuple[float, float, float]] = None) -> tuple[float, float, float]:
    """Fill ``pos_east_m/pos_north_m/up_m`` from lat/lon/alt for every observation. Returns the origin."""
    if origin is None:
        origin = compute_origin(observations)
    o_lat, o_lon, o_alt = origin
    for obs in observations:
        if obs.lat is None or obs.lon is None:
            continue
        e, n, u = geodetic_to_enu(obs.lat, obs.lon, obs.alt_m if obs.alt_m is not None else o_alt,
                                  o_lat, o_lon, o_alt)
        obs.pos_east_m, obs.pos_north_m, obs.up_m = e, n, u
    return origin


def _bearing_deg(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Initial compass bearing (CW from North, [0,360)) from point 0 to point 1."""
    phi1, phi2 = math.radians(lat0), math.radians(lat1)
    dlon = math.radians(lon1 - lon0)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return normalize_heading_deg(math.degrees(math.atan2(y, x)))


def derive_headings(observations: list[Observation]) -> None:
    """Set ``yaw_deg`` on each observation from the GPS track bearing (marked derived).

    Bearing at frame i = bearing from frame i to frame i+1 (last frame reuses the previous bearing).
    Observations must be in track order.
    """
    n = len(observations)
    for i, obs in enumerate(observations):
        j = i + 1 if i + 1 < n else i - 1
        if j < 0:
            continue
        a, b = (observations[i], observations[j]) if j > i else (observations[j], observations[i])
        if None in (a.lat, a.lon, b.lat, b.lon):
            continue
        obs.yaw_deg = _bearing_deg(a.lat, a.lon, b.lat, b.lon)
        obs.gt_yaw_sigma_deg = None
        obs.extra.setdefault("heading_source", "gps_track_bearing")  # honest: derived, not measured


def _enu_distance(a: Observation, b: Observation) -> float:
    if None in (a.pos_east_m, a.pos_north_m, b.pos_east_m, b.pos_north_m):
        return float("inf")
    return math.hypot(a.pos_east_m - b.pos_east_m, a.pos_north_m - b.pos_north_m)


def cumulative_route_distance(observations: list[Observation]) -> list[float]:
    """Cumulative along-track ENU distance (m) at each observation (requires ENU set, track order)."""
    dist = [0.0]
    for i in range(1, len(observations)):
        step = _enu_distance(observations[i - 1], observations[i])
        dist.append(dist[-1] + (0.0 if math.isinf(step) else step))
    return dist


def sample_candidates(
    observations: list[Observation],
    n_places: int,
    *,
    min_spacing_m: float = 1000.0,
    seed: int = 0,
    usable: Optional[Callable[[Observation], bool]] = None,
) -> list[Observation]:
    """Route-stratified place sampling (research R6): split the route into ``n_places`` equal
    along-track strata and pick one *usable* frame per stratum, enforcing a minimum inter-place
    ENU spacing so places are distinct (no near-duplicates). Deterministic given ``seed``.

    ``usable(obs)`` gates degenerate frames (e.g. tunnel/low-sky exclusion, from QC) -- default: all
    frames usable. This function performs **no matching/ranking**; it only selects frames spatially.
    """
    if not observations:
        return []
    rng = random.Random(seed)
    keep = [o for o in observations if (usable is None or usable(o))]
    if not keep:
        return []
    dist = cumulative_route_distance(keep)
    total = dist[-1] if dist[-1] > 0 else float(len(keep))
    edges = [total * k / n_places for k in range(n_places + 1)]

    chosen: list[Observation] = []
    for k in range(n_places):
        lo, hi = edges[k], edges[k + 1]
        bucket = [keep[i] for i in range(len(keep)) if lo <= dist[i] < hi] or \
                 [keep[i] for i in range(len(keep)) if lo <= dist[i] <= hi]
        if not bucket:
            continue
        rng.shuffle(bucket)
        for cand in bucket:  # first candidate respecting min spacing to already-chosen places
            if all(_enu_distance(cand, c) >= min_spacing_m for c in chosen):
                chosen.append(cand)
                break
    chosen.sort(key=lambda o: o.frame_index)
    return chosen


def nearest_reference_distance(candidate: Observation, reference_obs: list[Observation]) -> float:
    """The ENU distance from ``candidate`` to its nearest reference place (research R17: pairing/
    out-of-database decisions must use real position, never frame-index/timestamp proximity)."""
    return min((_enu_distance(candidate, r) for r in reference_obs), default=float("inf"))


def select_out_of_db(
    candidate_obs: list[Observation],
    reference_obs: list[Observation],
    min_distance_m: float,
) -> set[str]:
    """Candidate observation ids that are at least ``min_distance_m`` from **every** reference place
    (research R9's out-of-database construction rule, generalized/reusable). Distance is real ENU
    position (research R17), never frame-index or timestamp order. Pure selection -- no ranking, no
    matching; the complement (candidates *within* ``min_distance_m`` of some reference) are the
    in-database/matched candidates a caller may pair by nearest reference for provenance, but pairing
    identity itself is never asserted here (research R9: 007 emits positions + in_coverage only)."""
    return {
        c.observation_id for c in candidate_obs
        if nearest_reference_distance(c, reference_obs) >= min_distance_m
    }
