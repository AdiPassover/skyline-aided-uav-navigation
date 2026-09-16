"""Skyline extraction and description for UAV visual relocalization."""

from .extraction import extract_skyline, SkylineResult
from .descriptors import (
    height_profile,
    fourier_descriptor,
    slope_signature,
    shape_context,
    profile_distance,
    fourier_distance,
    shape_context_distance,
)

__all__ = [
    "extract_skyline",
    "SkylineResult",
    "height_profile",
    "fourier_descriptor",
    "slope_signature",
    "shape_context",
    "profile_distance",
    "fourier_distance",
    "shape_context_distance",
]
