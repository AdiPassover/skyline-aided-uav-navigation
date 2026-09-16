"""Reuse `EXP-VO-007`'s height-profile loader without duplicating it.

The LiDAR height series and its "this is height above the imaged surface, not `up_m`" contract are
`EXP-VO-007`'s; `EXP-VO-008` consumes them unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_vo_007"))

from height_profile import load_height_profile as load_profile   # noqa: E402,F401
