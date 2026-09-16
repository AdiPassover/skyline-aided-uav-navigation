"""Export canonical skyline profiles from validated SKY observation sessions for the Java INT side.

The file-based SKY -> INT boundary (``DEC-INT-002``; dual view per ``DEC-INT-003``). Reuses the SKY
lane's own, unmodified code for every step that touches a skyline -- the curve sources
(``hsreloc.retrieval.sources``, ``hsreloc.extraction.auto_source``, ``hsreloc.simret.sources``), the
profile stage (``hsreloc.retrieval.profile.normalize``: resample to ``n_samples``, mean removed,
image-fraction units) and the degeneracy rule (``profile.is_degenerate``) -- and writes one CSV row per
physical observation (schema 2):

    frame_index,timestamp_s,sync_dt_s,observation_id,valid,invalid_reason,profile,
    west_observation_id,west_available,west_valid,west_invalid_reason,west_profile

``frame_index`` is the **VO dataset's** frame index the observation is paired to, by exactly one of
four declared mechanisms, and the manifest names which and whether the identity is **exact**:

* ``--sim-observations <run>/skyline/observations.csv`` **with** ``--vo-frames`` -- **exact capture
  identity** (recordings since 2026-09-07, whose simulator consumes one pending skyline request on the
  next nadir/VO frame and logs that frame's ``vo_frame_id`` and ``vo_synchronized`` on the skyline
  row). Each observation's raw id (``sim_observation_id_raw``) is looked up in the run's own
  ``observations.csv``; the frame it names must exist in the VO dataset and carry the same timestamp
  to 1e-6 s, otherwise the export is **refused** -- an identity that contradicts the clock is a data
  fault, never something to repair by nearest-neighbour. A capture the simulator could not
  synchronise (``vo_frame_id = -1``) has no VO frame and therefore no row; it is listed in the
  manifest. ``sync_dt_s`` is written as the verified residual (0). North and West of one row are
  the same simulator row and hence the same ``vo_frame_id`` -- one skyline observation, one nadir/VO
  frame, one query-time local pose.
* ``--vo-frames`` alone -- **legacy / approximate**: the frame whose ``timestamp_s`` is nearest the
  capture time, within ``--sync-tolerance-s``; ``sync_dt_s`` = capture time minus frame time per row,
  and a pairing outside the tolerance becomes an invalid row naming the residual, never a silent
  drop. For the pre-2026-09-07 recordings, whose skyline ids and nadir frame ids are unrelated.
* ``--frame-map`` (explicit ``observation_id,frame_index``) or ``--frame-offset`` (the session's own
  index plus an offset) -- **declared** by the operator.

The North session is required. A West session (``--west-session-dir``) is paired to it observation
by observation through the shared capture identity (``sim_observation_id_raw`` for simulator
sessions, else ``frame_index``); an observation with no West partner is written
``west_available=false`` -- never fabricated from North -- and a West observation with no North
partner is a refusal, because the two must be one capture.
``profile`` fields are ``n_samples`` semicolon-separated samples, empty on an invalid row. A
``skyline_profiles.json`` manifest records both sources' ``describe()`` blocks, the pairing mechanism
and its residual statistics, the profile configuration, the counts, and the matcher hint (C1-32).
Nothing here matches anything; the C1 lags are the matcher's business and have no position semantics.

Run with the repository venv from the repository root::

    python evaluation/tools/int/export_skyline_profiles.py --session-dir observations_sim/<north-session> \\
        [--west-session-dir observations_sim/<west-session>] --source sim_exact \\
        --vo-frames datasets/<id>/frames.csv --out <dir>

``--source`` is one of ``oracle`` (``skylines_oracle/``, ``--provenance`` defaults to ``oracle:manual``),
``auto`` (``skylines_auto/``, ``--method-id`` required), or the simulator keys ``sim_exact`` / ``dp`` /
``segformer`` (``hsreloc.simret.sources.build_source``; ``segformer`` needs ``--mask-root``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HSRELOC_ROOT = REPO / "skyline"
if str(HSRELOC_ROOT) not in sys.path:
    sys.path.insert(0, str(HSRELOC_ROOT))

import numpy as np                                                        # noqa: E402

import hsreloc                                                            # noqa: E402,F401  (bootstraps sys.path)
from hsreloc.extraction.auto_source import AutomaticCurveSource           # noqa: E402
from hsreloc.observation import read_session                              # noqa: E402
from hsreloc.retrieval.profile import ProfileConfig, is_degenerate, normalize  # noqa: E402
from hsreloc.retrieval.skyline_curve import CurveError                    # noqa: E402
from hsreloc.retrieval.sources import OracleCurveSource                   # noqa: E402

CSV_NAME = "skyline_profiles.csv"
MANIFEST_NAME = "skyline_profiles.json"
HEADER = ["frame_index", "timestamp_s", "sync_dt_s", "observation_id", "valid", "invalid_reason", "profile",
          "west_observation_id", "west_available", "west_valid", "west_invalid_reason", "west_profile"]
SCHEMA_VERSION = "2.1.0"          # CSV unchanged since 2.0.0; the manifest gained frame_pairing.identity
PAIR_KEY = "sim_observation_id_raw"
EXACT_TOLERANCE_S = 1e-6
IDENTITY_EXACT = "exact"
IDENTITY_APPROXIMATE = "approximate"
IDENTITY_DECLARED = "declared"


class ExportError(RuntimeError):
    pass


# --------------------------------------------------------------------------- sources


def build_source(args, sessions: dict, width: int, height: int, store_root: Path):
    if args.source == "oracle":
        return OracleCurveSource(store_root, sessions, height, width, expect_provenance=args.provenance)
    if args.source == "auto":
        if not args.method_id:
            raise ExportError("--source auto needs --method-id")
        return AutomaticCurveSource(store_root, sessions, height, width, args.method_id)
    from hsreloc.simret.sources import build_source as sim_build_source
    spec = {"store_root": str(store_root)}
    if args.source == "segformer":
        if not args.mask_root:
            raise ExportError("--source segformer needs --mask-root")
        spec = {"mask_root": str(args.mask_root)}
    if args.source == "dp" and args.extraction_manifest:
        spec["extraction_manifest"] = str(args.extraction_manifest)
    return sim_build_source(args.source, spec, sessions, width, height,
                            store_root=store_root if args.source != "segformer" else None)


# --------------------------------------------------------------------------- pairing (pure)


def load_frame_map(path: Path | None) -> dict | None:
    if path is None:
        return None
    mapping = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["observation_id"]] = int(row["frame_index"])
    return mapping


def load_vo_frames(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``frame_index`` and ``timestamp_s`` of the VO dataset's ``frames.csv`` (contracts/dataset.md)."""
    idx, ts = [], []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for col in ("frame_index", "timestamp_s"):
            if reader.fieldnames is None or col not in reader.fieldnames:
                raise ExportError(f"{path}: no '{col}' column in {reader.fieldnames}")
        for row in reader:
            idx.append(int(row["frame_index"]))
            ts.append(float(row["timestamp_s"]))
    t = np.asarray(ts, dtype=float)
    if len(t) == 0:
        raise ExportError(f"{path}: no frames")
    if not np.all(np.diff(t) > 0):
        raise ExportError(f"{path}: frame timestamps are not strictly increasing")
    return np.asarray(idx, dtype=int), t


def pair_by_time(capture_time_s: float, frame_idx: np.ndarray, frame_t: np.ndarray) -> tuple[int, float]:
    """LEGACY / APPROXIMATE: the VO frame nearest the capture instant, and ``capture - frame`` in seconds."""
    k = int(np.argmin(np.abs(frame_t - capture_time_s)))
    return int(frame_idx[k]), float(capture_time_s - frame_t[k])


def load_sim_observations(path: Path) -> dict:
    """The simulator run's own ``skyline/observations.csv``, keyed by raw observation id: the exact
    nadir/VO frame each skyline capture was synchronised with. Refuses a file without the
    synchronisation columns -- that is a pre-2026-09-07 recording, which only the approximate
    mechanism can pair."""
    rows: dict = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        needed = ("observation_id", "sim_time_s", "vo_frame_id", "vo_synchronized")
        missing = [c for c in needed if reader.fieldnames is None or c not in reader.fieldnames]
        if missing:
            raise ExportError(f"{path}: not a synchronised simulator export (missing {missing}); a recording "
                              "without vo_frame_id can only be paired approximately (--vo-frames alone)")
        for row in reader:
            oid = row["observation_id"]
            if oid in rows:
                raise ExportError(f"{path}: duplicate observation_id {oid!r}")
            rows[oid] = {"sim_time_s": float(row["sim_time_s"]), "vo_frame_id": int(row["vo_frame_id"]),
                         "vo_synchronized": row["vo_synchronized"].strip() in ("1", "true", "True")}
    if not rows:
        raise ExportError(f"{path}: no observations")
    return rows


def pair_exact(obs, sim_rows: dict, frame_idx: np.ndarray, frame_t: np.ndarray) -> tuple[int | None, float | None, str | None]:
    """EXACT capture identity: the VO frame the simulator recorded for this capture, verified
    against the VO dataset's clock. Returns ``(frame, residual, None)``; ``(None, None, reason)``
    when the simulator recorded no synchronised frame; raises on any contradiction."""
    key = pair_key(obs)
    if key.startswith("frame_index:"):
        raise ExportError(f"{obs.observation_id}: no {PAIR_KEY} extra -- exact identity needs a simulator session")
    row = sim_rows.get(key)
    if row is None:
        raise ExportError(f"{obs.observation_id}: raw id {key!r} is not in the simulator observations file")
    if abs(row["sim_time_s"] - float(obs.timestamp_s)) > EXACT_TOLERANCE_S:
        raise ExportError(f"{obs.observation_id}: session capture time {obs.timestamp_s} differs from the "
                          f"simulator row's {row['sim_time_s']} -- not the same recording")
    if not row["vo_synchronized"] or row["vo_frame_id"] < 0:
        return None, None, f"no synchronised VO frame (vo_frame_id={row['vo_frame_id']}, vo_synchronized=" \
                           f"{int(row['vo_synchronized'])})"
    frame = row["vo_frame_id"]
    where = np.flatnonzero(frame_idx == frame)
    if len(where) != 1:
        raise ExportError(f"{obs.observation_id}: vo_frame_id {frame} is not a frame of the VO dataset "
                          f"({len(where)} matches) -- the skyline export and the VO dataset are not one recording")
    dt = float(obs.timestamp_s) - float(frame_t[int(where[0])])
    if abs(dt) > EXACT_TOLERANCE_S:
        raise ExportError(f"{obs.observation_id}: vo_frame_id {frame} carries timestamp {frame_t[int(where[0])]} "
                          f"but the capture was at {obs.timestamp_s} (residual {dt:.6f} s): the exact identity "
                          "contradicts the clock; refused rather than re-paired by proximity")
    return frame, 0.0, None            # verified to EXACT_TOLERANCE_S; the residual written is the contract's 0


def pair_key(obs) -> str:
    """The identity two views of one capture share: the raw simulator id, else the session index."""
    raw = obs.extra.get(PAIR_KEY) if getattr(obs, "extra", None) else None
    return str(raw) if raw not in (None, "") else f"frame_index:{obs.frame_index}"


def pair_west(north_obs: list, west_obs: list | None) -> dict:
    """North observation id -> West observation (or None). Refuses a West with no North partner and
    two Wests claiming one North."""
    if west_obs is None:
        return {o.observation_id: None for o in north_obs}
    by_key: dict = {}
    for w in west_obs:
        k = pair_key(w)
        if k in by_key:
            raise ExportError(f"two West observations share the capture key {k!r}: {by_key[k].observation_id}, "
                              f"{w.observation_id}")
        by_key[k] = w
    out: dict = {}
    used = set()
    for n in north_obs:
        k = pair_key(n)
        w = by_key.get(k)
        out[n.observation_id] = w
        if w is not None:
            used.add(k)
            if abs(float(w.timestamp_s) - float(n.timestamp_s)) > 1e-6:
                raise ExportError(f"North {n.observation_id} and West {w.observation_id} share key {k!r} but "
                                  f"not the capture time ({n.timestamp_s} vs {w.timestamp_s})")
    orphans = [w.observation_id for k, w in by_key.items() if k not in used]
    if orphans:
        raise ExportError(f"{len(orphans)} West observation(s) have no North partner (e.g. {orphans[0]}): the "
                          "two views must be one capture; a West-only observation cannot be represented")
    return out


def profile_or_reason(source, observation_id: str, config: ProfileConfig, floor: float) -> tuple[list | None, str | None]:
    """The canonical profile of one observation, or the reason there is none. Refusal is data."""
    try:
        curve = source.get(observation_id)
        profile = normalize(curve, config)
        if not np.all(np.isfinite(profile)):
            raise CurveError("profile has non-finite samples")
        if is_degenerate(profile, floor):
            raise CurveError(f"degenerate profile (std below {floor})")
        return [float(v) for v in profile], None
    except Exception as exc:                     # refusal is data, never a crash of the export
        return None, str(exc).replace(",", ";").replace("\n", " ")[:200]


def build_rows(north_obs: list, west_of: dict, north_source, west_source, config: ProfileConfig,
               floor: float, frame_map: dict | None, frame_offset: int,
               vo_frames: tuple[np.ndarray, np.ndarray] | None, sync_tolerance_s: float,
               sim_obs: dict | None = None) -> tuple[list, dict]:
    """One row per North observation, paired to a VO frame and to its West view. Pure: no I/O.
    ``sim_obs`` (with ``vo_frames``) selects the exact capture identity; ``vo_frames`` alone the
    approximate one."""
    rows, seen, stats = [], {}, {"n_valid": 0, "n_invalid": 0, "n_west_available": 0, "n_west_valid": 0,
                                "n_sync_rejected": 0, "sync_dt_abs_max_s": 0.0, "invalid_reasons": {},
                                "unsynchronized": {}}
    if sim_obs is not None and vo_frames is None:
        raise ExportError("exact capture identity needs the VO dataset's frames.csv (--vo-frames) to verify it")
    for obs in north_obs:
        sync_dt = None
        if frame_map is not None:
            if obs.observation_id not in frame_map:
                raise ExportError(f"{obs.observation_id}: not in --frame-map")
            frame = frame_map[obs.observation_id]
        elif sim_obs is not None:
            frame, sync_dt, why = pair_exact(obs, sim_obs, vo_frames[0], vo_frames[1])
            if frame is None:
                stats["unsynchronized"][obs.observation_id] = why      # no VO frame: no row, recorded
                continue
            stats["sync_dt_abs_max_s"] = max(stats["sync_dt_abs_max_s"], abs(sync_dt))
        elif vo_frames is not None:
            frame, sync_dt = pair_by_time(float(obs.timestamp_s), vo_frames[0], vo_frames[1])
            stats["sync_dt_abs_max_s"] = max(stats["sync_dt_abs_max_s"], abs(sync_dt))
        else:
            frame = obs.frame_index + frame_offset
        if frame in seen:
            raise ExportError(f"VO frame {frame} produced by two observations ({seen[frame]}, "
                              f"{obs.observation_id}); fix --frame-map / --frame-offset, or the capture is denser "
                              "than the VO frame rate")
        seen[frame] = obs.observation_id

        profile, reason = profile_or_reason(north_source, obs.observation_id, config, floor)
        if sync_dt is not None and abs(sync_dt) > sync_tolerance_s:
            reason = f"sync residual {sync_dt:.4f} s exceeds tolerance {sync_tolerance_s:.4f} s"
            profile = None
            stats["n_sync_rejected"] += 1
        if profile is None:
            stats["n_invalid"] += 1
            stats["invalid_reasons"][obs.observation_id] = reason
        else:
            stats["n_valid"] += 1

        w = west_of.get(obs.observation_id)
        if w is None:
            west = ["", "false", "", "", ""]
        else:
            stats["n_west_available"] += 1
            wprofile, wreason = profile_or_reason(west_source, w.observation_id, config, floor)
            if sync_dt is not None and abs(sync_dt) > sync_tolerance_s:
                wprofile, wreason = None, reason
            if wprofile is None:
                west = [w.observation_id, "true", "false", wreason, ""]
                stats["invalid_reasons"][w.observation_id] = wreason
            else:
                stats["n_west_valid"] += 1
                west = [w.observation_id, "true", "true", "", ";".join(repr(v) for v in wprofile)]

        rows.append([frame, repr(float(obs.timestamp_s)), "" if sync_dt is None else repr(sync_dt),
                     obs.observation_id, "true" if profile is not None else "false",
                     "" if profile is not None else reason,
                     ";".join(repr(v) for v in profile) if profile is not None else ""] + west)
    rows.sort(key=lambda r: r[0])
    return rows, stats


def write_csv(path: Path, rows: list) -> str:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(HEADER) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- main


def read_one(session_dir: Path):
    session_dir = session_dir.resolve()
    meta, observations = read_session(session_dir)
    widths = {o.image_width_px for o in observations}
    heights = {o.image_height_px for o in observations}
    if len(widths) != 1 or len(heights) != 1:
        raise ExportError(f"session {meta['session_id']} mixes image sizes {widths}x{heights}; one size per export")
    return session_dir, meta, observations, widths.pop(), heights.pop()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", required=True, type=Path,
                    help="the NORTH session: observations/<session_id> (session.json + observations.csv)")
    ap.add_argument("--west-session-dir", type=Path, default=None,
                    help="the synchronised WEST session of the same capture, when one exists")
    ap.add_argument("--source", required=True, choices=["oracle", "auto", "sim_exact", "dp", "segformer"])
    ap.add_argument("--provenance", default="oracle:manual", help="expected provenance for --source oracle")
    ap.add_argument("--method-id", default=None, help="extractor id for --source auto")
    ap.add_argument("--mask-root", type=Path, default=None, help="label-map root for --source segformer")
    ap.add_argument("--extraction-manifest", type=Path, default=None, help="dp extraction manifest (optional)")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--degenerate-std-floor", type=float, default=1e-3)
    ap.add_argument("--frame-offset", type=int, default=0,
                    help="added to each observation's frame_index to obtain the VO frame index")
    ap.add_argument("--frame-map", type=Path, default=None,
                    help="CSV observation_id,frame_index overriding the session's frame indices")
    ap.add_argument("--vo-frames", type=Path, default=None,
                    help="the VO dataset's frames.csv: with --sim-observations, verifies the exact identity; "
                         "alone, pairs each observation to the nearest frame by time (legacy / approximate)")
    ap.add_argument("--sim-observations", type=Path, default=None,
                    help="the simulator run's skyline/observations.csv carrying vo_frame_id / vo_synchronized: "
                         "exact capture identity (requires --vo-frames)")
    ap.add_argument("--sync-tolerance-s", type=float, default=0.05,
                    help="largest |capture time - frame time| accepted under --vo-frames alone")
    ap.add_argument("--max-lag-samples", type=int, default=32, help="matcher hint recorded in the manifest")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if sum(x is not None for x in (args.frame_map, args.vo_frames)) + (args.frame_offset != 0) > 1:
        raise ExportError("declare exactly one pairing mechanism: --frame-map, --vo-frames "
                          "[--sim-observations] or --frame-offset")
    if args.sim_observations is not None and args.vo_frames is None:
        raise ExportError("--sim-observations needs --vo-frames (the VO dataset the identity is verified against)")

    north_dir, north_meta, north_obs, width, height = read_one(args.session_dir)
    north_sessions = {o.observation_id: north_meta["session_id"] for o in north_obs}
    north_source = build_source(args, north_sessions, width, height, north_dir.parent)

    west_source, west_meta, west_obs = None, None, None
    if args.west_session_dir is not None:
        west_dir, west_meta, west_obs, w_width, w_height = read_one(args.west_session_dir)
        if (w_width, w_height) != (width, height):
            raise ExportError("North and West sessions differ in image size")
        if west_meta.get("skyline_view", "west") != "west" or north_meta.get("skyline_view", "north") != "north":
            raise ExportError(f"expected a north and a west session, got {north_meta.get('skyline_view')!r} and "
                              f"{west_meta.get('skyline_view')!r}")
        west_sessions = {o.observation_id: west_meta["session_id"] for o in west_obs}
        west_source = build_source(args, west_sessions, w_width, w_height, west_dir.parent)
    west_of = pair_west(north_obs, west_obs)

    profile_config = ProfileConfig(n_samples=args.n_samples, normalize_mean=True, detrend=False)
    profile_config.validate()
    frame_map = load_frame_map(args.frame_map)
    vo_frames = load_vo_frames(args.vo_frames) if args.vo_frames is not None else None
    sim_obs = load_sim_observations(args.sim_observations) if args.sim_observations is not None else None

    rows, stats = build_rows(north_obs, west_of, north_source, west_source, profile_config,
                             args.degenerate_std_floor, frame_map, args.frame_offset, vo_frames,
                             args.sync_tolerance_s, sim_obs)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / CSV_NAME
    digest = write_csv(csv_path, rows)
    if frame_map is not None:
        pairing = {"mechanism": "frame_map", "identity": IDENTITY_DECLARED, "path": str(args.frame_map)}
    elif sim_obs is not None:
        pairing = {"mechanism": "sim_vo_frame_id_exact", "identity": IDENTITY_EXACT,
                   "sim_observations": str(args.sim_observations), "vo_frames": str(args.vo_frames),
                   "exact_tolerance_s": EXACT_TOLERANCE_S, "sync_dt_abs_max_s": stats["sync_dt_abs_max_s"],
                   "n_unsynchronized": len(stats["unsynchronized"]),
                   "unsynchronized_observations": stats["unsynchronized"],
                   "note": "one skyline observation <-> the nadir/VO frame the simulator consumed it on "
                           "(vo_frame_id) <-> that frame's local pose; North and West are one simulator row; "
                           "verified against the VO dataset's clock, never re-paired by proximity"}
    elif vo_frames is not None:
        pairing = {"mechanism": "vo_frames_nearest_time", "identity": IDENTITY_APPROXIMATE,
                   "path": str(args.vo_frames),
                   "sync_tolerance_s": args.sync_tolerance_s, "sync_dt_abs_max_s": stats["sync_dt_abs_max_s"],
                   "n_sync_rejected": stats["n_sync_rejected"],
                   "note": "LEGACY / APPROXIMATE (pre-2026-09-07 recordings): sync_dt_s = skyline capture time - "
                           "VO frame time; the VO pose used for a query is the paired frame's, so a residual of "
                           "dt at speed v displaces the query by v*dt"}
    else:
        pairing = {"mechanism": "frame_offset", "identity": IDENTITY_DECLARED, "frame_offset": args.frame_offset}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "producer": "evaluation/tools/int/export_skyline_profiles.py",
        "north_session_id": north_meta["session_id"],
        "north_session_dir": str(north_dir),
        "west_session_id": None if west_meta is None else west_meta["session_id"],
        "west_session_dir": None if args.west_session_dir is None else str(args.west_session_dir.resolve()),
        "west_pairing_key": PAIR_KEY + " (else frame_index)",
        "session_evidence_tier": north_meta.get("evidence_tier"),
        "session_evidence_caveat": north_meta.get("evidence_caveat"),
        "north_source": north_source.describe(),
        "west_source": None if west_source is None else west_source.describe(),
        "profile": profile_config.as_dict(),
        "degenerate_std_floor": args.degenerate_std_floor,
        "frame_pairing": pairing,
        "image_width_px": width,
        "image_height_px": height,
        "n_rows": len(rows),
        "n_valid": stats["n_valid"],
        "n_invalid": stats["n_invalid"],
        "n_west_available": stats["n_west_available"],
        "n_west_valid": stats["n_west_valid"],
        "invalid_reasons": stats["invalid_reasons"],
        "csv_sha256": digest,
        "matcher_hint": {"variant": "c1_bounded_lag_ncc", "max_lag_samples": args.max_lag_samples,
                         "min_overlap_frac": 0.6,
                         "note": "neither view's C1 lag carries position semantics (DEC-INT-002)"},
    }
    (args.out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[int] exported {len(rows)} rows ({stats['n_valid']} North-valid, {stats['n_west_available']} with "
          f"West, {stats['n_west_valid']} West-valid, {stats['n_sync_rejected']} beyond sync tolerance, "
          f"{len(stats['unsynchronized'])} unsynchronised captures without a row; pairing "
          f"{pairing['mechanism']} / identity {pairing['identity']}) from "
          f"{north_meta['session_id']} via {north_source.describe().get('kind')} -> {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        print(f"[int] refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
