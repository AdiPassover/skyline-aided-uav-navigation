"""Stage-1 synthetic corpus: constructed skylines with known answers, and a real store to hold them.

Research **R11**. Used only by Stage-1 tests. **No real data is read, generated from, or referenced.**

Two things are built here, and the second is the important one:

1. **Curves with constructed answers.** Each "place" gets a deterministic silhouette; which query
   belongs to which place is a fact of construction, not a measurement. Four cases are built to
   exercise every acceptance branch: *unambiguous*, *aliased*, *out-of-coverage*, *degenerate*.

2. **A real spec-006/007 observation store, reference set and query set on disk**, produced by the
   *actual* ``hsreloc.build`` builders. Stage 1 therefore drives the same ``OracleCurveSource`` ->
   ``refset`` -> record path that Stage 2 will, over synthetic data. That is deliberate: if Stage 1
   used a bespoke shortcut pipeline, proving it would prove less about the run that matters.

**On silhouette design.** Places share a common base shape with a small place-specific signature on
top -- which is both realistic for a corridor and necessary for the fixture to be discriminating:
identical-by-construction wrong-pair scores would have zero spread, and the acceptance rule correctly
refuses a population it cannot standardise. The constants below shape the *test data* so each branch
is reachable; they are not, and must never become, adjustments to the frozen acceptance thresholds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from hsreloc import build, ingest, observation
from hsreloc.observation import Observation
from hsreloc.retrieval.fix import GeodeticOrigin, enu_to_geodetic

W, H = 128, 96
ORIGIN = GeodeticOrigin(lat_deg=63.0, lon_deg=10.0, alt_m=100.0)
PLACE_SPACING_M = 1200.0

_BASE_ROW = H * 0.55
_BASE_AMP = 10.0          # the shared corridor shape: makes wrong pairs correlate highly...
_SIGNATURE_AMP = 3.2      # ...and the place signature is what separates them
_JITTER = 0.05            # tiny per-place variation so wrong-pair scores have non-zero spread
_QUERY_NOISE = 0.10       # a query is its place's skyline, seen again imperfectly


def _x() -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, W)


def place_curve(place: int, *, seed: int = 0, noise: float = 0.0) -> np.ndarray:
    """The silhouette of one place: shared base + a place-specific signature + optional noise."""
    x = _x()
    rng = np.random.default_rng((seed + 1) * 9973 + place)
    base = _BASE_ROW + _BASE_AMP * np.sin(x + 0.4) + 0.5 * _BASE_AMP * np.sin(2.0 * x + 1.1)
    freq = 3.0 + 1.7 * place
    signature = _SIGNATURE_AMP * np.sin(freq * x + 0.31 * place)
    jitter = _JITTER * np.random.default_rng(place + 4242).standard_normal(W)
    curve = base + signature + jitter
    if noise:
        curve = curve + noise * rng.standard_normal(W)
    return np.clip(curve, 4.0, H - 4.0)


def out_of_coverage_curve() -> np.ndarray:
    """A place the reference corridor does not contain.

    The **base** shape differs, not merely the fine signature -- different terrain, not the same
    ridge seen differently. That distinction is what makes the case a fair test: a query whose
    signature alone is unusual still shares the corridor's dominant shape, so an amplitude-sensitive
    metric can find a spurious standout in it (measured: L1 returned a confident SUCCESS on such a
    construction). A genuinely different skyline is near-equidistant from every reference, which is
    the score population an honest refusal has to come from. Verified refused by all three baselines.
    """
    x = _x()
    curve = (_BASE_ROW
             + _BASE_AMP * np.sin(x + 0.4 + np.pi)
             + 0.5 * _BASE_AMP * np.sin(2.0 * x + 1.1 + np.pi)
             + _SIGNATURE_AMP * np.sin(5.5 * x))
    return np.clip(curve, 4.0, H - 4.0)


def flat_curve() -> np.ndarray:
    """A featureless skyline -- the degenerate case, which must be refused, not matched."""
    return np.full(W, _BASE_ROW, dtype=np.float64)


def _write_image(path: Path, curve: np.ndarray) -> None:
    """A tiny rendered stand-in. Nothing in the matcher ever opens it; only the builder copies it."""
    import cv2

    img = np.zeros((H, W, 3), dtype=np.uint8)
    rows = np.arange(H)[:, None]
    sky = rows < curve[None, :]
    img[..., 0] = np.where(sky, 220, 40)
    img[..., 1] = np.where(sky, 200, 80)
    img[..., 2] = np.where(sky, 180, 50)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _write_curve(path: Path, curve: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("col,row\n")
        for c, r in enumerate(curve):
            f.write(f"{c},{float(r)}\n")


def _observation(obs_id: str, session: str, index: int, east_m: float, north_m: float) -> Observation:
    lat, lon, alt = enu_to_geodetic(east_m, north_m, 0.0, ORIGIN)
    return Observation(
        observation_id=obs_id, session_id=session, frame_index=index, timestamp_s=float(index),
        image_path=f"{obs_id}.png", image_width_px=W, image_height_px=H,
        gt_source="synthetic", lat=lat, lon=lon, alt_m=alt,
        extra={"condition_season": session},
    )


def build_corpus(root: Path, n_places: int = 8) -> dict:
    """Write the observation store, reference set and query set; return the constructed answers.

    Query cases, by ``observation_id``:

    * ``q_unambiguous_<k>`` -- place ``k``'s skyline seen again; answer is reference ``ref_<k>``
    * ``q_aliased``         -- matches the deliberately duplicated pair (``ref_0`` / ``ref_alias``)
    * ``q_out_of_coverage`` -- a place absent from the reference set
    * ``q_degenerate``      -- a flat skyline
    """
    root = Path(root)
    obs_root = root / "observations"
    ref_session, query_session = "synth_ref", "synth_query"

    reference_obs, query_obs = [], []
    answers: dict = {}

    # --- references: one per place, plus a deliberate duplicate of place 0 for the aliased case ---
    for k in range(n_places):
        rid = f"ref_{k}"
        o = _observation(rid, ref_session, k, east_m=k * PLACE_SPACING_M, north_m=0.0)
        reference_obs.append(o)
        _write_curve(obs_root / ref_session / "skylines_oracle" / f"{rid}.csv", place_curve(k))
        _write_image(obs_root / ref_session / "images" / f"{rid}.png", place_curve(k))

    alias = _observation("ref_alias", ref_session, n_places,
                         east_m=0.0 + 40.0, north_m=15.0)   # same place, a few metres away
    reference_obs.append(alias)
    _write_curve(obs_root / ref_session / "skylines_oracle" / "ref_alias.csv", place_curve(0))
    _write_image(obs_root / ref_session / "images" / "ref_alias.png", place_curve(0))

    # --- queries ---
    def add_query(qid: str, curve: np.ndarray, index: int, east_m: float, north_m: float) -> None:
        o = _observation(qid, query_session, index, east_m=east_m, north_m=north_m)
        query_obs.append(o)
        _write_curve(obs_root / query_session / "skylines_oracle" / f"{qid}.csv", curve)
        _write_image(obs_root / query_session / "images" / f"{qid}.png", curve)

    index = 0
    for k in range(2, n_places):          # places 0/1 are reserved for the aliased pair's geometry
        qid = f"q_unambiguous_{k}"
        add_query(qid, place_curve(k, seed=7, noise=_QUERY_NOISE), index,
                  east_m=k * PLACE_SPACING_M + 5.0, north_m=3.0)
        answers[qid] = {"case": "unambiguous", "reference_id": f"ref_{k}",
                        "east_m": k * PLACE_SPACING_M, "north_m": 0.0}
        index += 1

    add_query("q_aliased", place_curve(0, seed=11, noise=_QUERY_NOISE * 0.5), index,
              east_m=20.0, north_m=8.0)
    answers["q_aliased"] = {"case": "aliased", "reference_id": None}
    index += 1

    # A place the reference set does not contain -- different terrain, and far away geographically.
    # No candidate stands out, which is exactly the refusal the acceptance rule must produce.
    add_query("q_out_of_coverage", out_of_coverage_curve(), index,
              east_m=250_000.0, north_m=190_000.0)
    answers["q_out_of_coverage"] = {"case": "out_of_coverage", "reference_id": None}
    index += 1

    add_query("q_degenerate", flat_curve(), index, east_m=5.0, north_m=1.0)
    answers["q_degenerate"] = {"case": "degenerate", "reference_id": None}

    # --- sessions on disk, ENU normalized against the shared origin ---
    origin = (ORIGIN.lat_deg, ORIGIN.lon_deg, ORIGIN.alt_m)
    for session, obs in ((ref_session, reference_obs), (query_session, query_obs)):
        ingest.normalize_to_enu(obs, origin)
        meta = {
            "session_id": session, "platform": "synthetic", "frame_convention": "ENU",
            "heading_convention": "compass_cw_from_north", "evidence_tier": "T1",
            "local_frame_origin": {"lat": origin[0], "lon": origin[1], "alt_m": origin[2]},
            "camera": {"calibration_id": None, "fov_deg": None, "orientation": "forward"},
        }
        observation.write_session(obs_root / session, meta, obs)

    oracle_dirs = {ref_session: obs_root / ref_session / "skylines_oracle",
                   query_session: obs_root / query_session / "skylines_oracle"}
    images_dirs = {ref_session: obs_root / ref_session / "images",
                   query_session: obs_root / query_session / "images"}
    caveat = ("Synthetic Stage-1 fixture with constructed answers (spec "
              "20260823-014933-sky-oracle-skyline-relocalization, research R11). NOT real data and "
              "never evidence about any real skyline, dataset, or platform.")

    build.build_reference_set(
        reference_obs, oracle_dirs, root / "skyline_refdb", "refset-synth-sky-v1",
        dataset_id="synthetic", extraction_mode="oracle:sim_exact", evidence_tier="T1",
        evidence_caveat=caveat, reference_spacing_m=PLACE_SPACING_M)
    build.build_query_set(
        query_obs, oracle_dirs, images_dirs, root / "datasets", "skyquery-synth-sky-v1",
        dataset_id="synthetic", dataset_revision="synth-v1",
        query_in_coverage={o.observation_id: (o.observation_id != "q_out_of_coverage")
                           for o in query_obs},
        evidence_tier="T1", source_type="synthetic", evidence_caveat=caveat,
        extraction_mode="oracle:sim_exact")

    with (root / "answers.json").open("w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2, sort_keys=True)

    return {
        "root": root,
        "observations": obs_root,
        "reference_set": root / "skyline_refdb" / "refset-synth-sky-v1",
        "query_set": root / "datasets" / "skyquery-synth-sky-v1",
        "answers": answers,
        "origin": ORIGIN,
    }


def query_index_of(query_set: Path, observation_id: str) -> int:
    """Map an ``observation_id`` back to the ``query_index`` the record will use as ``query_id``."""
    with (Path(query_set) / "frames.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if Path(row["image_path"]).stem == observation_id:
                return int(row["frame_index"])
    raise KeyError(observation_id)
