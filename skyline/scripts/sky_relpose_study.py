"""EXP-SKY-009 — how much relative horizontal position a matched skyline pair actually carries.

Offline, deterministic, three stages, each re-runnable from the previous stage's saved outputs::

    PYTHONPATH=. python scripts/sky_relpose_study.py --config configs/sky-relpose.json --stage pairs
    PYTHONPATH=. python scripts/sky_relpose_study.py --config configs/sky-relpose.json --stage estimate
    PYTHONPATH=. python scripts/sky_relpose_study.py --config configs/sky-relpose.json --stage report

``pairs``    builds every reference/query pair from the EXP-SKY-008 extended batch (anchor → swipe,
             the INT scenario; and swipe-internal, for 2-D coverage), runs the **frozen C1-32 matcher**
             on the frozen 256-sample profile and records what it returns plus what it discards (the
             score-vs-lag peak shape, eight windowed local lags, the raw row offset, the range the
             pair's own height difference implies). Ground truth ``Δp = p_query − p_ref`` is written
             beside every row in ENU metres.
``estimate`` applies the estimator ladder (global lag → windowed lags → linear check), with every
             fitted quantity fitted on DEV (village) only, and the INT safety comparison.
``report``   metrics.json, report.md and JPEG figures from the saved CSVs.

Nothing frozen is modified: ``hsreloc/retrieval``, ``hsreloc/matchers`` and every EXP-SKY-008 artifact
are read only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsreloc.observation import read_session                                    # noqa: E402
from hsreloc.retrieval.skyline_curve import CurveError                          # noqa: E402
from hsreloc.simret import relpose                                              # noqa: E402
from hsreloc.simret.geometry import assign_bin                                  # noqa: E402
from hsreloc.simret.sources import (SessionSilverSource, SimExactSource,       # noqa: E402
                                    session_map)

LEVEL_ORDER = ("village", "mountains", "city")
LEVEL_COLOR = {"village": "#3366aa", "mountains": "#117744", "city": "#aa5511"}
DAY_CLEAR = "DAY+CLEAR"


def _resolve(base: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def _load_cfg(path: Path) -> tuple:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    final_cfg = json.loads(_resolve(base, cfg["sim_final_config"]).read_text(encoding="utf-8"))
    return cfg, base, final_cfg


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return str(v)


def _open(path: Path, mode: str):
    """``.csv.gz`` for the two large tables (tens of MB plain), plain ``.csv`` otherwise."""
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def _write_csv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fields})


def _read_csv(path: Path) -> list:
    with _open(path, "r") as f:
        return list(csv.DictReader(f))


#: Tables that are gzip-compressed on disk (regenerable; kept tracked for traceability).
_GZ = {"pairs_swipe_internal", "estimates"}


def _table_path(out: Path, name: str) -> Path:
    return out / (f"{name}.csv.gz" if name in _GZ else f"{name}.csv")


def _f(v):
    try:
        return float(v) if v not in ("", None) else math.nan
    except ValueError:
        return math.nan


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


# ==================================================================================================
# stage 1 — pairs
# ==================================================================================================

def _index(cfg: dict, base: Path) -> list:
    rows = _read_csv(_resolve(base, cfg["index_csv"]))
    for r in rows:
        for k in ("east_m", "north_m", "up_m", "lateral_east_m", "longitudinal_north_m", "vertical_up_m",
                  "total_displacement_m", "swipe_center_anchor_distance_m", "hour"):
            r[k] = _f(r.get(k))
        r["level_key"] = cfg["level_keys"][r["level"]]
        r["condition"] = f"{r['time_of_day']}+{r['clouds']}"
        # anchor_capture rows carry the Anchor dataclass repr in anchor_id; swipe rows carry the bare id
        m = re.match(r"Anchor\(anchor_id='([^']+)'", r.get("anchor_id") or "")
        if m:
            r["anchor_id"] = m.group(1)
    return rows


def _camera(store: Path, session_ids: list, n_samples: int) -> relpose.ProfileCamera:
    cams = set()
    cam = None
    for sid in session_ids:
        meta, _ = read_session(store / sid)
        cam = relpose.ProfileCamera.from_session_meta(meta, n_samples)
        cams.add(tuple(sorted(cam.as_dict().items())))
    if len(cams) != 1:
        raise SystemExit(f"sessions disagree on the camera model: {cams}")
    return cam


def _sources(cfg: dict, base: Path, final_cfg: dict, sessions: dict, cam) -> dict:
    store = _resolve(base, cfg["store_root"])
    out = {}
    for key in cfg["sources"]:
        if key == "sim_exact":
            out[key] = SimExactSource(store, sessions, cam.width_px, cam.height_px)
        elif key == "segformer":
            spec = final_cfg["sources"]["segformer"]
            out[key] = SessionSilverSource(
                _resolve(_resolve(base, cfg["sim_final_config"]).parent, spec["mask_root"]), sessions,
                cam.width_px, cam.height_px, scale_short_side=spec.get("scale_short_side", 512),
                closing_px=spec.get("closing_px", 0), min_valid_frac=spec.get("min_valid_frac", 0.5),
                expected_model_revision=spec.get("expected_model_revision"))
        else:
            raise SystemExit(f"unsupported source {key!r} for this study")
    return out


class _Curves:
    """Per-source curve + frozen profile cache with refusal bookkeeping."""

    def __init__(self, source):
        self.source = source
        self.curves: dict = {}
        self.profiles: dict = {}
        self.refused: dict = {}

    def get(self, oid: str):
        if oid in self.refused:
            return None
        if oid not in self.curves:
            try:
                c = self.source.get(oid)
            except CurveError as exc:
                self.refused[oid] = str(exc)[:160]
                return None
            self.curves[oid] = c
            self.profiles[oid] = relpose.profile_of(c)
        return self.curves[oid]


def _site_range_maps(index_rows: list, curves_gt: _Curves, cfg: dict, cam) -> dict:
    """Per swipe: a per-column horizontal range map from the swipe's own vertical legs (GT curves).

    This is the *reference-side* range a database could carry (a DEM-rendered reference has it
    exactly); here it is measured from the simulator's vertical excursions at the swipe centre.
    """
    min_dz = float(cfg["site_range_min_abs_dz_m"])
    max_h = float(cfg["site_range_max_horizontal_m"])
    maps, notes = {}, {}
    by_session = defaultdict(list)
    for r in index_rows:
        if r["kind"] == "swipe":
            by_session[r["session_id"]].append(r)
    for sid, rows in by_session.items():
        start = min(rows, key=lambda r: (r["total_displacement_m"], r["observation_id"]))
        vertical = [r for r in rows if r["leg_axis"] == "vertical"]
        if not vertical:
            notes[sid] = "no vertical leg"
            continue
        # the star's legs meet at the anchor, not at the run-start frame: the base of the vertical
        # column is the centre-phase observation under it at the start altitude
        vx, vy = float(np.median([r["east_m"] for r in vertical])), float(np.median([r["north_m"] for r in vertical]))
        bases = [r for r in rows if r["phase"] == "center" and math.hypot(r["east_m"] - vx, r["north_m"] - vy) <= max_h]
        if not bases:
            notes[sid] = "no centre-phase observation under the vertical leg"
            continue
        centre = min(bases, key=lambda r: (abs(r["up_m"] - start["up_m"]), r["observation_id"]))
        c = curves_gt.get(centre["observation_id"])
        if c is None:
            notes[sid] = "centre curve refused"
            continue
        stack = []
        for r in rows:
            dz = r["up_m"] - centre["up_m"]
            horiz = math.hypot(r["east_m"] - centre["east_m"], r["north_m"] - centre["north_m"])
            if abs(dz) < min_dz or horiz > max_h:
                continue
            q = curves_gt.get(r["observation_id"])
            if q is None:
                continue
            stack.append(relpose.range_from_vertical_parallax(c.row_per_col, q.row_per_col, dz, cam.fy,
                                                              min_abs_dz_m=min_dz))
        if not stack:
            notes[sid] = "no vertical-leg observation with |dz| >= threshold"
            continue
        S = np.vstack(stack)
        with np.errstate(all="ignore"):
            med = np.nanmedian(S, axis=0)
        maps[sid] = {"range_per_column": med, "n_vertical_obs": len(stack),
                     "centre": centre["observation_id"],
                     "valid_frac": float(np.isfinite(med).mean()),
                     "median_range_m": float(np.nanmedian(med)) if np.isfinite(med).any() else math.nan}
    return {"maps": maps, "notes": notes}


_FROZEN_CHECK_EVERY = 20


def _pair_row(ref, qry, ref_curve, q_curve, ref_prof, q_prof, cfg, cam, site_map, extra: dict,
              use_frozen: bool = True) -> dict:
    """One pair's measurement row.

    ``use_frozen`` runs the actual ``BoundedLagNccMatcher`` for the primary lag/score (every
    anchor→swipe pair — the INT scenario). For the much larger swipe-internal family the vectorised
    search (bit-equal by test) supplies them, and every ``_FROZEN_CHECK_EVERY``-th pair is still
    cross-checked against the frozen matcher; a disagreement aborts the run.
    """
    max_lag = int(cfg["max_lag_samples"])
    nw = int(cfg["n_windows"])
    dE = qry["east_m"] - ref["east_m"]
    dN = qry["north_m"] - ref["north_m"]
    dz = qry["up_m"] - ref["up_m"]
    search = relpose.bounded_lag_search(q_prof, ref_prof, max_lag, float(cfg["min_overlap_frac"]))
    _pair_row.counter = getattr(_pair_row, "counter", 0) + 1
    if use_frozen or _pair_row.counter % _FROZEN_CHECK_EVERY == 0:
        frozen = relpose.frozen_c1(q_prof, ref_prof, max_lag)
        if np.isfinite(frozen.score) and (search.lag != frozen.shift or abs(search.score - frozen.score) > 1e-9):
            raise SystemExit(f"vectorised search disagrees with the frozen matcher on {ref['observation_id']} / "
                             f"{qry['observation_id']}: {search.lag}/{search.score} vs {frozen.shift}/{frozen.score}")
        c1_score, c1_lag, c1_overlap, c1_zero = frozen.score, frozen.shift, frozen.overlap, frozen.diagnostics.get("score_at_zero_lag")
        c1_from = "frozen"
    else:
        c1_score, c1_lag, c1_overlap, c1_zero = search.score, search.lag, search.overlap, search.score_at_zero
        c1_from = "vectorised(bit-equal)"
    wins = relpose.windowed_lag_search(q_prof, ref_prof, nw, max_lag, float(cfg["min_overlap_frac"]))
    d_row = q_curve.row_per_col - ref_curve.row_per_col
    pair_D = relpose.range_from_vertical_parallax(ref_curve.row_per_col, q_curve.row_per_col, dz, cam.fy,
                                                  min_abs_dz_m=float(cfg["range_from_dz_min_abs_m"]))
    pair_Dw = relpose.window_ranges(pair_D, cam, nw)
    site_Dw = (relpose.window_ranges(site_map["range_per_column"], cam, nw)
               if site_map is not None else np.full(nw, np.nan))
    row = {
        **extra,
        "ref_id": ref["observation_id"], "query_id": qry["observation_id"],
        "ref_session": ref["session_id"], "query_session": qry["session_id"],
        "ref_condition": ref["condition"], "query_condition": qry["condition"],
        "query_leg_axis": qry.get("leg_axis") or "", "query_phase": qry.get("phase") or "",
        "hard_tag": qry.get("hard_tag") or "",
        "ref_up_m": ref["up_m"], "query_up_m": qry["up_m"],
        "dE_m": dE, "dN_m": dN, "dz_m": dz, "sep_m": math.hypot(dE, dN),
        "c1_score": c1_score, "c1_lag": c1_lag, "c1_overlap": c1_overlap,
        "c1_score_at_zero": c1_zero, "c1_from": c1_from,
        "c1_saturated": bool(abs(c1_lag) >= max_lag) if np.isfinite(c1_score) else False,
        "c1_secondary_margin": search.secondary_margin, "c1_curvature": search.curvature,
        "c1_n_searched": search.n_searched,
        "row_offset_px": float(np.median(d_row)),
        "row_med_abs_offset_removed_px": float(np.median(np.abs(d_row - np.median(d_row)))),
        "site_range_median_m": site_map["median_range_m"] if site_map is not None else math.nan,
    }
    for k, w in enumerate(wins):
        row[f"w{k}_lag"] = w["lag"]
        row[f"w{k}_score"] = w["score"]
        row[f"w{k}_sat"] = w["saturated"]
        row[f"w{k}_flat"] = w["flat"]
        row[f"w{k}_secondary"] = w["secondary_margin"]
        row[f"w{k}_site_D"] = site_Dw[k]
        row[f"w{k}_pair_D"] = pair_Dw[k]
    return row


def _pair_fields(nw: int) -> list:
    base = ["family", "level", "split", "source", "swipe_session", "ref_id", "query_id", "ref_session",
            "query_session", "ref_condition", "query_condition", "query_leg_axis", "query_phase", "hard_tag",
            "ref_up_m", "query_up_m", "dE_m", "dN_m", "dz_m", "sep_m",
            "c1_score", "c1_lag", "c1_overlap", "c1_score_at_zero", "c1_from", "c1_saturated", "c1_secondary_margin",
            "c1_curvature", "c1_n_searched", "row_offset_px", "row_med_abs_offset_removed_px",
            "site_range_median_m"]
    for k in range(nw):
        base += [f"w{k}_lag", f"w{k}_score", f"w{k}_sat", f"w{k}_flat", f"w{k}_secondary",
                 f"w{k}_site_D", f"w{k}_pair_D"]
    return base


def stage_pairs(cfg: dict, base: Path, final_cfg: dict) -> None:
    t0 = time.time()
    out = _resolve(base, cfg["out_dir"])
    store = _resolve(base, cfg["store_root"])
    index_rows = _index(cfg, base)
    session_ids = sorted({r["session_id"] for r in index_rows})
    cam = _camera(store, session_ids, int(cfg["profile_n_samples"]))
    sessions = session_map(store, session_ids)
    sources = _sources(cfg, base, final_cfg, sessions, cam)
    caches = {k: _Curves(s) for k, s in sources.items()}
    nw = int(cfg["n_windows"])
    tau = float(cfg["tau_pos_m"])
    max_sep = float(cfg["max_pair_separation_m"])
    stride = int(cfg["swipe_internal_stride"])

    print(f"[pairs] camera {cam.as_dict()}")
    site = _site_range_maps(index_rows, caches["sim_exact"], cfg, cam)
    print(f"[pairs] site range maps: {len(site['maps'])} swipes with maps, notes={site['notes']}")
    _json(out / "site_range_maps.json",
          {"per_swipe": {sid: {k: v for k, v in m.items() if k != "range_per_column"} for sid, m in site["maps"].items()},
           "notes": site["notes"], "method": "median over the swipe's own vertical-leg observations of "
           "fy*dz/(row_q-row_ref) per column (GT curves); a reference-side range a DEM-backed database would carry",
           "min_abs_dz_m": cfg["site_range_min_abs_dz_m"]})
    np.savez_compressed(out / "site_range_maps.npz",
                        **{sid: m["range_per_column"] for sid, m in site["maps"].items()})

    anchors = {}
    for r in index_rows:
        if r["kind"] == "anchor_capture":
            anchors[(r["level"], r["anchor_id"], r["condition"])] = r
    swipes = defaultdict(list)
    for r in index_rows:
        if r["kind"] == "swipe":
            swipes[r["session_id"]].append(r)

    refusals = {k: {} for k in caches}
    families = {"anchor_swipe": [], "swipe_internal": []}
    stats = defaultdict(int)
    for sid, rows in sorted(swipes.items()):
        rows.sort(key=lambda r: r["observation_id"])
        level, split = rows[0]["level"], rows[0]["split"]
        lk = rows[0]["level_key"]
        site_map = site["maps"].get(sid)
        # --- anchor → swipe (the INT scenario)
        centre = min(rows, key=lambda r: (r["total_displacement_m"], r["observation_id"]))
        aid = centre.get("swipe_center_anchor_id") or ""
        adist = centre["swipe_center_anchor_distance_m"]
        if aid and np.isfinite(adist) and adist <= tau:
            ref_conditions = [DAY_CLEAR]
            for skey, cache in caches.items():
                conds = ref_conditions + (list(cfg["appearance_reference_conditions"]) if skey == "segformer" else [])
                for cond in conds:
                    ref = anchors.get((level, aid, cond))
                    if ref is None:
                        stats[f"missing_anchor_capture:{level}:{aid}:{cond}"] += 1
                        continue
                    rc = cache.get(ref["observation_id"])
                    if rc is None:
                        stats[f"ref_refused:{skey}"] += 1
                        continue
                    for q in rows:
                        qc = cache.get(q["observation_id"])
                        if qc is None:
                            stats[f"query_refused:{skey}"] += 1
                            continue
                        sep = math.hypot(q["east_m"] - ref["east_m"], q["north_m"] - ref["north_m"])
                        if sep > max_sep:
                            stats["anchor_pair_beyond_max_sep"] += 1
                            continue
                        families["anchor_swipe"].append(_pair_row(
                            ref, q, rc, qc, cache.profiles[ref["observation_id"]], cache.profiles[q["observation_id"]],
                            cfg, cam, site_map, {"family": "anchor_swipe", "level": lk, "split": split,
                                                 "source": skey, "swipe_session": sid}))
        else:
            stats[f"swipe_out_of_coverage:{sid}"] += 1
        # --- swipe-internal pairs (2-D coverage; correlated, characterisation only)
        sub = rows[::stride]
        for skey, cache in caches.items():
            for i in range(len(sub)):
                ri = sub[i]
                rc = cache.get(ri["observation_id"])
                if rc is None:
                    continue
                for j in range(i + 1, len(sub)):
                    q = sub[j]
                    sep = math.hypot(q["east_m"] - ri["east_m"], q["north_m"] - ri["north_m"])
                    if sep > max_sep:
                        continue
                    qc = cache.get(q["observation_id"])
                    if qc is None:
                        continue
                    families["swipe_internal"].append(_pair_row(
                        ri, q, rc, qc, cache.profiles[ri["observation_id"]], cache.profiles[q["observation_id"]],
                        cfg, cam, site_map, {"family": "swipe_internal", "level": lk, "split": split,
                                             "source": skey, "swipe_session": sid}, use_frozen=False))
        print(f"[pairs] {sid} ({lk}): anchor_swipe={sum(1 for r in families['anchor_swipe'] if r['swipe_session']==sid)} "
              f"swipe_internal={sum(1 for r in families['swipe_internal'] if r['swipe_session']==sid)} "
              f"t={time.time()-t0:.0f}s")
    for k, cache in caches.items():
        refusals[k] = cache.refused
    fields = _pair_fields(nw)
    for fam, rows in families.items():
        _write_csv(_table_path(out, f"pairs_{fam}"), rows, fields)
    _json(out / "pairs_manifest.json", {
        "relpose_version": relpose.RELPOSE_VERSION, "camera": cam.as_dict(),
        "n_pairs": {k: len(v) for k, v in families.items()},
        "n_pairs_by_level_source": {fam: dict(sorted(_count(rows, ("level", "source")).items()))
                                    for fam, rows in families.items()},
        "curve_refusals": {k: len(v) for k, v in refusals.items()},
        "curve_refusal_examples": {k: dict(list(v.items())[:5]) for k, v in refusals.items()},
        "stats": dict(stats), "config": cfg,
        "sources": {k: s.describe() for k, s in sources.items()},
        "sign_conventions": {"dE_dN_dz": "query minus reference, ENU metres (hsreloc.simret.conventions)",
                             "c1_lag": "profile samples; q[i] ~ r[i + lag]; positive for an eastward query",
                             "row_offset_px": "median(row_query - row_ref); positive for a higher query"},
        "elapsed_s": time.time() - t0})
    print(f"[pairs] done in {time.time()-t0:.0f}s: {json.dumps({k: len(v) for k, v in families.items()})}")


def _count(rows, keys):
    c = defaultdict(int)
    for r in rows:
        c["|".join(str(r[k]) for k in keys)] += 1
    return c


# ==================================================================================================
# stage 2 — estimators + safety comparison
# ==================================================================================================

def _load_pairs(out: Path, nw: int) -> dict:
    fams = {}
    for fam in ("anchor_swipe", "swipe_internal"):
        p = _table_path(out, f"pairs_{fam}")
        if not p.exists():
            continue
        rows = _read_csv(p)
        for r in rows:
            for k in ("ref_up_m", "query_up_m", "dE_m", "dN_m", "dz_m", "sep_m", "c1_score", "c1_lag",
                      "c1_score_at_zero", "c1_secondary_margin", "c1_curvature", "row_offset_px",
                      "row_med_abs_offset_removed_px", "site_range_median_m"):
                r[k] = _f(r.get(k))
            r["c1_saturated"] = r["c1_saturated"] == "True"
            r["windows"] = []
            for k in range(nw):
                r["windows"].append({"lag": _f(r[f"w{k}_lag"]), "score": _f(r[f"w{k}_score"]),
                                     "saturated": r[f"w{k}_sat"] == "True", "flat": r[f"w{k}_flat"] == "True",
                                     "centre_sample": None, "site_D": _f(r[f"w{k}_site_D"]),
                                     "pair_D": _f(r[f"w{k}_pair_D"])})
        fams[fam] = rows
    return fams


def _window_centres(cam, nw):
    return [(a + b - 1) / 2.0 for a, b in relpose.window_bounds(cam.n_samples, nw)]


def fit_dev_range(rows: list, cam, cfg: dict) -> dict:
    """The scalar range prior: OLS through the origin of ΔE on −Δα_centre, DEV pure-lateral pairs."""
    dev = cfg["dev_level"]
    xs, ys = [], []
    centre = (cam.n_samples - 1) / 2.0
    for r in rows:
        if r["level"] != dev or r["source"] != "sim_exact":
            continue
        if abs(r["dN_m"]) > 1.0 or abs(r["dz_m"]) > 1.0 or r["c1_saturated"] or not np.isfinite(r["c1_score"]):
            continue
        if r["c1_score"] < cfg["gate"]["min_c1_score"]:
            continue
        xs.append(-float(cam.lag_to_azimuth_shift_rad(centre, r["c1_lag"])))
        ys.append(r["dE_m"])
    xs, ys = np.asarray(xs), np.asarray(ys)
    if xs.size < 10 or float(xs @ xs) == 0.0:
        raise SystemExit("too few DEV pure-lateral pairs to fit a range prior")
    D = float((xs @ ys) / (xs @ xs))
    resid = ys - D * xs
    return {"range_m": D, "n": int(xs.size), "resid_rms_m": float(np.sqrt(np.mean(resid ** 2))),
            "rule": "OLS through origin of dE on -(azimuth shift at profile centre), DEV sim_exact "
                    "pure-lateral pairs (|dN|,|dz| <= 1 m) with C1 score >= gate and unsaturated lag"}


def _features(r: dict, cam, nw: int) -> np.ndarray:
    """Low-capacity feature vector for the linear check: global lag + window lags (nan→0) + valid flags."""
    f = [r["c1_lag"] if np.isfinite(r["c1_lag"]) else 0.0]
    for w in r["windows"]:
        ok = np.isfinite(w["lag"]) and np.isfinite(w["score"])
        f.append(w["lag"] if ok else 0.0)
        f.append(1.0 if ok else 0.0)
    f.append(1.0)
    return np.asarray(f, dtype=np.float64)


def fit_linear(rows: list, cam, cfg: dict, nw: int) -> dict:
    dev = cfg["dev_level"]
    X, Y = [], []
    for r in rows:
        if r["level"] != dev or r["source"] != "sim_exact" or not np.isfinite(r["c1_score"]):
            continue
        X.append(_features(r, cam, nw))
        Y.append([r["dE_m"], r["dN_m"]])
    X, Y = np.asarray(X), np.asarray(Y)
    lam = 1e-3 * X.shape[0]
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    resid = Y - X @ W
    return {"weights": W, "n": int(X.shape[0]),
            "dev_fit_rms_m": [float(np.sqrt(np.mean(resid[:, 0] ** 2))), float(np.sqrt(np.mean(resid[:, 1] ** 2)))],
            "rule": "ridge (lambda = 1e-3 n) from [global lag, 8 window lags, 8 validity flags, 1] to (dE, dN), "
                    "DEV sim_exact pairs of both families"}


ESTIMATORS = ("glob_Ddev", "glob_Dsite", "win_Ddev", "win_Dsite", "win_Dpair", "lin_dev")


def _estimate(r: dict, cam, nw: int, D_dev: float, lin_W, cfg: dict) -> dict:
    gate = cfg["gate"]
    centres = _window_centres(cam, nw)
    for k, w in enumerate(r["windows"]):
        w["centre_sample"] = centres[k]
    out = {}
    sat = r["c1_saturated"]
    out["glob_Ddev"] = relpose.estimate_global_lag(r["c1_lag"], cam, D_dev, saturated=sat)
    out["glob_Dsite"] = relpose.estimate_global_lag(r["c1_lag"], cam, r["site_range_median_m"], saturated=sat)
    kw = dict(min_score=float(gate["min_window_score"]), min_equations=int(gate["min_windows"]),
              max_residual_rad=float(gate["max_residual_rad"]))
    out["win_Ddev"] = relpose.estimate_from_windows(r["windows"], cam, D_dev, **kw)
    out["win_Dsite"] = relpose.estimate_from_windows(r["windows"], cam, [w["site_D"] for w in r["windows"]], **kw)
    out["win_Dpair"] = relpose.estimate_from_windows(r["windows"], cam, [w["pair_D"] for w in r["windows"]], **kw)
    if np.isfinite(r["c1_score"]):
        p = _features(r, cam, nw) @ lin_W
        out["lin_dev"] = relpose.Estimate(float(p[0]), float(p[1]), True, None, None, 0, None)
    else:
        out["lin_dev"] = relpose.Estimate(None, None, False, "no_score")
    return out


def stage_estimate(cfg: dict, base: Path, final_cfg: dict) -> None:
    out = _resolve(base, cfg["out_dir"])
    nw = int(cfg["n_windows"])
    manifest = json.loads((out / "pairs_manifest.json").read_text(encoding="utf-8"))
    cam = relpose.ProfileCamera(**{k: v for k, v in manifest["camera"].items() if k != "columns_per_sample"})
    fams = _load_pairs(out, nw)
    all_rows = [r for rows in fams.values() for r in rows]
    D_dev = fit_dev_range(all_rows, cam, cfg)
    lin = fit_linear(all_rows, cam, cfg, nw)
    print(f"[estimate] DEV range prior {D_dev['range_m']:.0f} m from {D_dev['n']} pairs (resid {D_dev['resid_rms_m']:.1f} m)")
    print(f"[estimate] linear check fitted on {lin['n']} DEV pairs, DEV fit RMS {lin['dev_fit_rms_m']}")
    gate = cfg["gate"]
    est_rows = []
    for fam, rows in fams.items():
        for r in rows:
            ests = _estimate(r, cam, nw, D_dev["range_m"], lin["weights"], cfg)
            gated_ok = (np.isfinite(r["c1_score"]) and r["c1_score"] >= gate["min_c1_score"] and not r["c1_saturated"])
            base_row = {k: r[k] for k in ("family", "level", "split", "source", "swipe_session", "ref_id", "query_id",
                                          "ref_condition", "query_condition", "query_leg_axis", "hard_tag",
                                          "dE_m", "dN_m", "dz_m", "sep_m", "c1_score", "c1_lag", "c1_saturated",
                                          "c1_secondary_margin", "site_range_median_m")}
            base_row["gate_c1"] = gated_ok
            for name, e in ests.items():
                base_row[f"{name}_dE"] = e.delta_east_m
                base_row[f"{name}_dN"] = e.delta_north_m
                base_row[f"{name}_valid"] = e.valid
                base_row[f"{name}_refusal"] = e.refusal or ""
                base_row[f"{name}_resid"] = e.residual_rad
                base_row[f"{name}_neq"] = e.n_equations
                if e.valid:
                    base_row[f"{name}_eE"] = abs(e.delta_east_m - r["dE_m"])
                    base_row[f"{name}_eN"] = abs(e.delta_north_m - r["dN_m"])
                    base_row[f"{name}_eH"] = math.hypot(e.delta_east_m - r["dE_m"], e.delta_north_m - r["dN_m"])
                else:
                    base_row[f"{name}_eE"] = base_row[f"{name}_eN"] = base_row[f"{name}_eH"] = math.nan
            est_rows.append(base_row)
    fields = list(est_rows[0].keys())
    _write_csv(_table_path(out, "estimates"), est_rows, fields)
    _json(out / "fits.json", {"dev_range_prior": D_dev, "linear_check": {**lin, "weights": lin["weights"].tolist()},
                              "estimators": {
                                  "glob_Ddev": "global C1 lag at the profile centre × DEV scalar range prior → dE only (dN := 0)",
                                  "glob_Dsite": "global C1 lag × the reference site's median range from its vertical legs (oracle scalar) → dE only",
                                  "win_Ddev": "8 windowed lags, least squares (dE, dN) with the DEV scalar range prior",
                                  "win_Dsite": "8 windowed lags with the reference site's per-window range map (oracle reference-side range, as a DEM-backed database would carry)",
                                  "win_Dpair": "8 windowed lags with the range the pair's OWN known height difference implies (barometric case; refuses when |dz| < threshold)",
                                  "lin_dev": "ridge regression on lag features fitted on DEV village only — the low-capacity data-driven check"},
                              "gate": gate})
    print(f"[estimate] wrote {len(est_rows)} rows")


# ==================================================================================================
# stage 3 — metrics, report, figures
# ==================================================================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return {"n": int(ok.sum()), "r": None, "slope": None, "resid_std": None}
    A = np.vstack([x[ok], np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    resid = y[ok] - A @ coef
    return {"n": int(ok.sum()), "r": float(np.corrcoef(x[ok], y[ok])[0, 1]), "slope": float(coef[0]),
            "intercept": float(coef[1]), "resid_std": float(np.std(resid))}


def observability(pairs: dict, cam, cfg: dict) -> dict:
    """What the global lag predicts — and does not — by level, source and displacement axis."""
    centre = (cam.n_samples - 1) / 2.0
    res = {}
    rows = [r for rows in pairs.values() for r in rows]
    for src in cfg["sources"]:
        for lk in LEVEL_ORDER:
            sel = [r for r in rows if r["source"] == src and r["level"] == lk and np.isfinite(r["c1_score"])
                   and r["c1_score"] >= cfg["gate"]["min_c1_score"] and not r["c1_saturated"]]
            lat = [r for r in sel if abs(r["dN_m"]) <= 1.0 and abs(r["dz_m"]) <= 1.0]
            lon = [r for r in sel if abs(r["dE_m"]) <= 1.0 and abs(r["dz_m"]) <= 1.0]
            ver = [r for r in sel if abs(r["dE_m"]) <= 1.0 and abs(r["dN_m"]) <= 1.0 and abs(r["dz_m"]) > 1.0]
            entry = {
                "n_gated": len(sel),
                "lag_vs_dE_pure_lateral": _corr([r["dE_m"] for r in lat], [r["c1_lag"] for r in lat]),
                "lag_vs_dN_pure_longitudinal": _corr([r["dN_m"] for r in lon], [r["c1_lag"] for r in lon]),
                "lag_vs_dz_pure_vertical": _corr([r["dz_m"] for r in ver], [r["c1_lag"] for r in ver]),
                "row_offset_vs_dz_pure_vertical": _corr([r["dz_m"] for r in ver], [r["row_offset_px"] for r in ver]),
                "lag_vs_dE_all_2d": _corr([r["dE_m"] for r in sel], [r["c1_lag"] for r in sel]),
                "lag_vs_dN_all_2d": _corr([r["dN_m"] for r in sel], [r["c1_lag"] for r in sel]),
            }
            # edge-window lag difference (right − left) as the scale/longitudinal proxy
            diff = [(r["dN_m"], r["windows"][-1]["lag"] - r["windows"][0]["lag"]) for r in lon
                    if np.isfinite(r["windows"][-1]["lag"]) and np.isfinite(r["windows"][0]["lag"])]
            entry["edge_window_lag_diff_vs_dN_pure_longitudinal"] = _corr([d[0] for d in diff], [d[1] for d in diff])
            # per-swipe implied range from the lateral slope — how much a scalar prior can vary
            per_swipe = {}
            for sid in sorted({r["swipe_session"] for r in lat}):
                s = [r for r in lat if r["swipe_session"] == sid]
                c = _corr([r["dE_m"] for r in s], [-float(cam.lag_to_azimuth_shift_rad(centre, r["c1_lag"])) for r in s])
                per_swipe[sid] = {"n": c["n"], "implied_range_m": (1.0 / c["slope"]) if c.get("slope") else None,
                                  "r": c["r"]}
            entry["implied_range_by_swipe_from_lateral_slope"] = per_swipe
            # conditional distribution: lag when the query is (nearly) at the reference laterally
            near = [r["c1_lag"] for r in sel if abs(r["dE_m"]) <= 5.0]
            entry["lag_when_abs_dE_le_5m"] = relpose.error_stats(np.abs(near)) if near else {"n": 0}
            res[f"{src}|{lk}"] = entry
    return res


def _by_bins(rows, key_fn, edges):
    out = defaultdict(list)
    for r in rows:
        b = assign_bin(key_fn(r), edges)
        if b is not None:
            out[b].append(r)
    return out


def metrics(est_rows: list, cfg: dict) -> dict:
    cat = float(cfg["catastrophic_m"])
    sep_bins = cfg["separation_bins_m"]
    dz_bins = cfg["dz_bins_m"]
    grid = cfg["e_before_grid_m"]
    res = {}
    for fam in ("anchor_swipe", "swipe_internal"):
        for src in cfg["sources"]:
            for lk in LEVEL_ORDER:
                for cond in ("DAY+CLEAR", "appearance"):
                    sel = [r for r in est_rows if r["family"] == fam and r["source"] == src and r["level"] == lk
                           and ((r["ref_condition"] == DAY_CLEAR) if cond == "DAY+CLEAR" else (r["ref_condition"] != DAY_CLEAR))]
                    if not sel:
                        continue
                    entry = {"n_pairs": len(sel), "n_gate_c1": int(sum(1 for r in sel if r["gate_c1"])),
                             "snap_error_all": relpose.error_stats([r["sep_m"] for r in sel], cat),
                             "estimators": {}}
                    for name in ESTIMATORS:
                        for gate_name, gate_fn in (("ungated", lambda r: r[f"{name}_valid"]),
                                                   ("gated", lambda r: r[f"{name}_valid"] and r["gate_c1"])):
                            ok = [r for r in sel if gate_fn(r)]
                            e = {"n_valid": len(ok), "coverage": len(ok) / len(sel),
                                 "refusals": dict(sorted(_count([r for r in sel if not gate_fn(r)], (f"{name}_refusal",)).items()))}
                            if ok:
                                eH = np.array([r[f"{name}_eH"] for r in ok])
                                e["eE"] = relpose.error_stats([r[f"{name}_eE"] for r in ok], cat)
                                e["eN"] = relpose.error_stats([r[f"{name}_eN"] for r in ok], cat)
                                e["eH"] = relpose.error_stats(eH, cat)
                                e["snap"] = relpose.snap_comparison([r["sep_m"] for r in ok], eH, grid)
                                e["by_separation_bin"] = {b: {"n": len(rs), "eH_median": float(np.median([r[f"{name}_eH"] for r in rs])),
                                                              "eH_p90": float(np.percentile([r[f"{name}_eH"] for r in rs], 90)),
                                                              "snap_median": float(np.median([r["sep_m"] for r in rs])),
                                                              "frac_refined_better": float(np.mean([r[f"{name}_eH"] < r["sep_m"] for r in rs]))}
                                                          for b, rs in sorted(_by_bins(ok, lambda r: r["sep_m"], sep_bins).items())}
                                e["by_abs_dz_bin"] = {b: {"n": len(rs), "eH_median": float(np.median([r[f"{name}_eH"] for r in rs])),
                                                          "eH_p90": float(np.percentile([r[f"{name}_eH"] for r in rs], 90))}
                                                      for b, rs in sorted(_by_bins(ok, lambda r: abs(r["dz_m"]), dz_bins).items())}
                            entry["estimators"][f"{name}|{gate_name}"] = e
                    res[f"{fam}|{src}|{lk}|{cond}"] = entry
    return res


def _fig_lag_vs_axis(out: Path, pairs: dict, cam, cfg: dict, axis: str, src: str, D_dev: float):
    plt = _plt()
    rows = [r for rows in pairs.values() for r in rows if r["source"] == src and np.isfinite(r["c1_score"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, lk in zip(axes, LEVEL_ORDER):
        if axis == "dE":
            sel = [r for r in rows if r["level"] == lk and abs(r["dN_m"]) <= 1 and abs(r["dz_m"]) <= 1]
        elif axis == "dN":
            sel = [r for r in rows if r["level"] == lk and abs(r["dE_m"]) <= 1 and abs(r["dz_m"]) <= 1]
        else:
            sel = [r for r in rows if r["level"] == lk and abs(r["dE_m"]) <= 1 and abs(r["dN_m"]) <= 1]
        g = [r for r in sel if r["c1_score"] >= cfg["gate"]["min_c1_score"] and not r["c1_saturated"]]
        b = [r for r in sel if r not in g]
        ax.scatter([r[f"{axis}_m"] for r in b], [r["c1_lag"] for r in b], s=6, color="#bbbbbb", label="score < 0.9 or saturated")
        ax.scatter([r[f"{axis}_m"] for r in g], [r["c1_lag"] for r in g], s=7, color=LEVEL_COLOR[lk], label="gated (score ≥ 0.9, |lag| < 32)")
        if axis == "dE":
            xs = np.linspace(-150, 150, 50)
            centre = (cam.n_samples - 1) / 2.0
            # DEV prior line: lag such that -Δα·D_dev = dE  (small-angle inverse)
            ys = [np.interp(-x / D_dev, [float(cam.lag_to_azimuth_shift_rad(centre, l)) for l in np.arange(-32, 33)][::-1],
                            np.arange(-32, 33)[::-1]) for x in xs]
            ax.plot(xs, ys, color="#222222", lw=1.2, ls="--", label=f"DEV prior D = {D_dev:.0f} m")
        ax.axhline(0, color="#999999", lw=0.6); ax.axvline(0, color="#999999", lw=0.6)
        ax.set_title(f"{lk} — {src}", fontsize=10)
        ax.set_xlabel(f"true {axis} (m, query − reference)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("frozen C1-32 lag (samples)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Global C1 lag vs {axis} on pure-{axis} pairs (both families)", fontsize=11)
    return _save(fig, out / "figures" / f"fig_lag_vs_{axis}_{src}.jpg")


def _fig_error_vs_sep(out: Path, m: dict, fam: str, src: str, names: list, cfg: dict):
    plt = _plt()
    bins = cfg["separation_bins_m"]
    labels = [f"{a:g}-{b:g}" for a, b in zip(bins[:-1], bins[1:])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, lk in zip(axes, LEVEL_ORDER):
        entry = m.get(f"{fam}|{src}|{lk}|DAY+CLEAR")
        if not entry:
            ax.set_title(f"{lk}: no pairs"); continue
        snap = [entry["estimators"][f"{names[0]}|gated"].get("by_separation_bin", {}).get(b, {}).get("snap_median", np.nan) for b in labels]
        ax.plot(range(len(labels)), snap, color="#222222", lw=1.5, ls="--", marker="s", ms=4, label="snap (= separation)")
        for name, style in zip(names, ("-", "-", "-", "-", ":", "-.")):
            e = entry["estimators"].get(f"{name}|gated", {})
            bs = e.get("by_separation_bin", {})
            ys = [bs.get(b, {}).get("eH_median", np.nan) for b in labels]
            ax.plot(range(len(labels)), ys, lw=1.4, ls=style, marker="o", ms=3, label=f"{name} (n={e.get('n_valid', 0)})")
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(f"{lk} — {src}", fontsize=10); ax.grid(alpha=0.25)
        ax.set_xlabel("true separation bin (m)")
    axes[0].set_ylabel("median horizontal error after relocalization (m)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{fam}: refined error vs snap error by separation (gated pairs, DAY+CLEAR reference)", fontsize=11)
    return _save(fig, out / "figures" / f"fig_error_vs_separation_{fam}_{src}.jpg")


def _fig_harmful(out: Path, m: dict, fam: str, src: str, name: str, cfg: dict):
    plt = _plt()
    grid = [float(x) for x in cfg["e_before_grid_m"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for lk in LEVEL_ORDER:
        entry = m.get(f"{fam}|{src}|{lk}|DAY+CLEAR")
        if not entry:
            continue
        e = entry["estimators"].get(f"{name}|gated", {}).get("snap")
        if not e:
            continue
        h = e["harmful_rate_vs_e_before"]
        ax.plot(grid, [h[f"{g:g}"]["snap"] for g in grid], color=LEVEL_COLOR[lk], ls="--", marker="s", ms=4, lw=1.2, label=f"{lk}: snap")
        ax.plot(grid, [h[f"{g:g}"]["refined"] for g in grid], color=LEVEL_COLOR[lk], ls="-", marker="o", ms=4, lw=1.6, label=f"{lk}: {name}")
    ax.set_xlabel("assumed VO error before relocalization, e_before (m)")
    ax.set_ylabel("harmful-correction rate  P(e_after > e_before)")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{fam} / {src}: discrete snap vs skyline-refined ({name}, gated)", fontsize=10)
    return _save(fig, out / "figures" / f"fig_harmful_rate_{fam}_{src}_{name}.jpg")


def _fig_delta_e(out: Path, est_rows: list, fam: str, src: str, name: str):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, lk in zip(axes, LEVEL_ORDER):
        sel = [r for r in est_rows if r["family"] == fam and r["source"] == src and r["level"] == lk
               and r["ref_condition"] == DAY_CLEAR and r["gate_c1"] and r[f"{name}_valid"]]
        if not sel:
            ax.set_title(f"{lk}: none"); continue
        imp = np.array([r["sep_m"] - r[f"{name}_eH"] for r in sel])
        ax.hist(imp, bins=np.linspace(-100, 100, 41), color=LEVEL_COLOR[lk], alpha=0.85)
        ax.axvline(0, color="#222222", lw=1)
        ax.set_title(f"{lk}: n={len(sel)}, better {100*np.mean(imp>0):.0f}%, median {np.median(imp):+.1f} m", fontsize=9)
        ax.set_xlabel("e_snap − e_refined (m)  (> 0: refinement helps)")
    axes[0].set_ylabel("pairs")
    fig.suptitle(f"{fam} / {src}: improvement over the discrete snap ({name}, gated)", fontsize=11)
    return _save(fig, out / "figures" / f"fig_improvement_{fam}_{src}_{name}.jpg")


def _md_stats(s: dict) -> str:
    if not s or s.get("n", 0) == 0:
        return "n=0"
    return (f"n={s['n']}, med {s['median']:.1f}, mean {s['mean']:.1f}, p90 {s['p90']:.1f}, p95 {s['p95']:.1f}, "
            f"max {s['max']:.0f}, >{s['catastrophic_m']:g} m: {100*s['frac_over_catastrophic']:.0f} %")


def stage_report(cfg: dict, base: Path, final_cfg: dict) -> None:
    out = _resolve(base, cfg["out_dir"])
    nw = int(cfg["n_windows"])
    manifest = json.loads((out / "pairs_manifest.json").read_text(encoding="utf-8"))
    fits = json.loads((out / "fits.json").read_text(encoding="utf-8"))
    cam = relpose.ProfileCamera(**{k: v for k, v in manifest["camera"].items() if k != "columns_per_sample"})
    pairs = _load_pairs(out, nw)
    est_rows = _read_csv(_table_path(out, "estimates"))
    for r in est_rows:
        for k in list(r.keys()):
            if k.endswith(("_dE", "_dN", "_eE", "_eN", "_eH", "_resid", "_m", "c1_score", "c1_lag", "c1_secondary_margin")):
                r[k] = _f(r[k])
        r["gate_c1"] = r["gate_c1"] == "True"
        for name in ESTIMATORS:
            r[f"{name}_valid"] = r[f"{name}_valid"] == "True"
    obs = observability(pairs, cam, cfg)
    m = metrics(est_rows, cfg)
    D_dev = fits["dev_range_prior"]["range_m"]
    figs = []
    for src in cfg["sources"]:
        for axis in ("dE", "dN", "dz"):
            figs.append(_fig_lag_vs_axis(out, pairs, cam, cfg, axis, src, D_dev))
        for fam in ("anchor_swipe", "swipe_internal"):
            figs.append(_fig_error_vs_sep(out, m, fam, src, list(ESTIMATORS), cfg))
            for name in ("glob_Ddev", "win_Dsite", "win_Dpair", "lin_dev"):
                figs.append(_fig_harmful(out, m, fam, src, name, cfg))
                figs.append(_fig_delta_e(out, est_rows, fam, src, name))
    _json(out / "metrics.json", {"observability": obs, "metrics": m, "fits": fits, "figures": [str(p.name) for p in figs],
                                 "n_estimate_rows": len(est_rows)})
    # ---- report.md
    L = []
    L.append("# EXP-SKY-009 — relative horizontal position from a matched skyline pair: study outputs\n")
    L.append(f"Generated by `scripts/sky_relpose_study.py` (relpose {relpose.RELPOSE_VERSION}); camera {manifest['camera']}; "
             f"pairs: {manifest['n_pairs']}; curve refusals: {manifest['curve_refusals']}.\n")
    L.append(f"DEV range prior (village, sim_exact, pure-lateral, gated): **{D_dev:.0f} m** from {fits['dev_range_prior']['n']} pairs, "
             f"residual RMS {fits['dev_range_prior']['resid_rms_m']:.1f} m. Linear check fitted on {fits['linear_check']['n']} DEV pairs "
             f"(DEV fit RMS dE/dN {fits['linear_check']['dev_fit_rms_m']}).\n")
    L.append("## Observability of the global C1 lag (gated pairs: score ≥ 0.9, |lag| < 32)\n")
    L.append("| source | level | n | lag~dE pure-lateral r / slope (samples/m) / resid σ | lag~dN pure-long. r | lag~dz pure-vert. r | row-offset~dz r / slope px/m | edge-window Δlag~dN r | implied range by swipe (m) | |lag| when \\|dE\\| ≤ 5 m: med / p90 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, e in obs.items():
        src, lk = key.split("|")
        a = e["lag_vs_dE_pure_lateral"]; b = e["lag_vs_dN_pure_longitudinal"]; c = e["lag_vs_dz_pure_vertical"]
        d = e["row_offset_vs_dz_pure_vertical"]; f = e["edge_window_lag_diff_vs_dN_pure_longitudinal"]
        rng = ", ".join(f"{v['implied_range_m']:.0f}" if v.get("implied_range_m") else "—" for v in e["implied_range_by_swipe_from_lateral_slope"].values())
        near = e["lag_when_abs_dE_le_5m"]
        fmt = lambda v, p=2: ("—" if v is None else f"{v:.{p}f}")
        L.append(f"| {src} | {lk} | {e['n_gated']} | {fmt(a['r'])} / {fmt(a['slope'],3)} / {fmt(a['resid_std'],1)} (n={a['n']}) | "
                 f"{fmt(b['r'])} (n={b['n']}) | {fmt(c['r'])} (n={c['n']}) | {fmt(d['r'])} / {fmt(d['slope'],3)} (n={d['n']}) | "
                 f"{fmt(f['r'])} (n={f['n']}) | {rng} | "
                 f"{near.get('median', float('nan')):.1f} / {near.get('p90', float('nan')):.1f} (n={near.get('n',0)}) |")
    L.append("")
    for fam in ("anchor_swipe", "swipe_internal"):
        L.append(f"## Estimator results — `{fam}`\n")
        L.append("Errors in metres on gated pairs (C1 score ≥ 0.9, unsaturated, estimator valid); coverage = valid ∧ gated / all pairs.\n")
        L.append("| source | level | ref cond | pairs | gated | estimator | coverage | eH | eE | eN | refined better than snap | harmful @ e_before=20 m snap / refined | @ 37.5 m snap / refined |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key, entry in m.items():
            f2, src, lk, cond = key.split("|")
            if f2 != fam:
                continue
            for name in ESTIMATORS:
                e = entry["estimators"].get(f"{name}|gated", {})
                if not e or e.get("n_valid", 0) == 0:
                    L.append(f"| {src} | {lk} | {cond} | {entry['n_pairs']} | {entry['n_gate_c1']} | {name} | {100*e.get('coverage',0):.0f} % | — | — | — | — | — | — |")
                    continue
                h = e["snap"]["harmful_rate_vs_e_before"]
                L.append(f"| {src} | {lk} | {cond} | {entry['n_pairs']} | {entry['n_gate_c1']} | {name} | {100*e['coverage']:.0f} % | "
                         f"{_md_stats(e['eH'])} | {_md_stats(e['eE'])} | {_md_stats(e['eN'])} | {100*e['snap']['frac_refined_better']:.0f} % | "
                         f"{100*h['20']['snap']:.0f} % / {100*h['20']['refined']:.0f} % | {100*h['37.5']['snap']:.0f} % / {100*h['37.5']['refined']:.0f} % |")
        L.append("")
    L.append("## Figures\n")
    for p in figs:
        L.append(f"- `figures/{p.name}`")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[report] wrote {out / 'report.md'} and {len(figs)} figures")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=("pairs", "estimate", "report", "all"), default="all")
    a = ap.parse_args(argv)
    cfg, base, final_cfg = _load_cfg(Path(a.config).resolve())
    if a.stage in ("pairs", "all"):
        stage_pairs(cfg, base, final_cfg)
    if a.stage in ("estimate", "all"):
        stage_estimate(cfg, base, final_cfg)
    if a.stage in ("report", "all"):
        stage_report(cfg, base, final_cfg)


if __name__ == "__main__":
    main()
