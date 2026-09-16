"""Load the LiDAR height profile a `--height-profile` argument points at.

Shared by `plot_groundtruth.py` and `analyse.py` so the two cannot disagree about which series they
are reading. Accepts either shape:

- a full `probe_candidates.py` report, from which the named sequence's profile is dug out; or
- a bare `{"t": [...], "height_m": [...]}` object, for a profile prepared by hand.

The distinction this series carries is the one `LIT-VO-004` §4 forced: `height_m` is height above
**the imaged surface**, which is what sets the pixel-to-metre factor, and is *not* the `up_m` column
of `groundtruth.csv`, which is height above the **takeoff datum**. They coincide only over flat
ground, and on no MARS-LVIG sequence do they coincide.
"""
from __future__ import annotations

import json
from pathlib import Path


class HeightProfileError(ValueError):
    pass


def load_height_profile(path: str | Path, scene: str | None = None) -> dict:
    """-> {"t": [...], "height_m": [...], "source": ...}."""
    obj = json.loads(Path(path).read_text())

    if "sequences" in obj:
        seqs = obj["sequences"]
        if scene is None:
            if len(seqs) != 1:
                raise HeightProfileError(
                    f"{path} covers {sorted(seqs)}; pass the scene name to disambiguate")
            scene = next(iter(seqs))
        if scene not in seqs:
            raise HeightProfileError(f"{path} has no sequence {scene!r}; has {sorted(seqs)}")
        prof = seqs[scene].get("lidar", {}).get("height_profile")
        if prof is None:
            raise HeightProfileError(
                f"{path} carries no height_profile for {scene} -- it was written by a "
                f"probe_candidates.py older than the field. Re-run the probe; the LiDAR clouds "
                f"are cached, so it costs nothing.")
        return {"t": prof["t"], "height_m": prof["height_m"],
                "source": f"{path} :: sequences.{scene}.lidar.height_profile"}

    if "t" in obj and "height_m" in obj:
        return {"t": obj["t"], "height_m": obj["height_m"], "source": str(path)}

    raise HeightProfileError(
        f"{path} is neither a probe_candidates report nor a bare "
        f"{{t, height_m}} object; keys are {sorted(obj)[:8]}")
