"""EXP-SKY-010 — recognition radius of the frozen C1-32 matcher versus true horizontal separation.

Offline, deterministic, two stages, the second re-runnable from the first's saved tables::

    PYTHONPATH=. python scripts/sky_recog_radius.py --config configs/sky-recog.json --stage score
    PYTHONPATH=. python scripts/sky_recog_radius.py --config configs/sky-recog.json --stage report

``score``  ranks every swipe query of the EXP-SKY-008 batch against (a) the level's sparse
           DAY+CLEAR anchor database with the frozen ``BoundedLagNccMatcher.match()`` itself
           (cross-checked against the committed C1 columns of ``evaluations/sim-ext-exp-variant``),
           plus the two appearance-condition anchor databases for SegFormer; and (b) a dense
           path database (horizontal-leg + centre swipe frames + anchors) through the batched
           bit-equal search, with hold-out radii applied *after* scoring. Writes one row per
           query x database instance with every region/exact quantity, and the raw score matrices.
``report`` bins by the actual nearest-reference distance, applies the pre-registered support and
           radius rules, the spacing geometry, sensitivity and slices; writes metrics.json,
           report.md and <= 8 figures.

Nothing frozen is modified; no range map or relative-position estimate is used (that is
EXP-SKY-009's question, not this one).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hsreloc.simret import recog                                                # noqa: E402
from hsreloc.simret.relpose import frozen_c1                                    # noqa: E402
from hsreloc.simret.sources import session_map                                  # noqa: E402
from sky_relpose_study import (LEVEL_COLOR, LEVEL_ORDER, _Curves, _camera,      # noqa: E402
                               _f, _index, _json, _load_cfg, _open, _plt,
                               _read_csv, _resolve, _save, _sources, _write_csv)

STUDY = "EXP-SKY-010"
DENSE_TABLE = "queries_dense.csv.gz"
SPARSE_TABLE = "queries_sparse.csv.gz"
BOOL_FIELDS = {"attainable", "region_top1", "region_in_top_k", "false_region", "ambiguous", "accepted_frame",
               "accepted_region", "exact_top1", "exact_in_top_k", "hard_tag"}


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------

def _query_meta(r: dict) -> dict:
    return {"query_id": r["observation_id"], "site": r["session_id"], "level": r["level_key"],
            "split": r["split"], "query_condition": r["condition"], "phase": r["phase"],
            "leg_axis": r["leg_axis"], "hard_tag": r["hard_tag"] == "True",
            "query_east_m": r["east_m"], "query_north_m": r["north_m"], "query_up_m": r["up_m"],
            "swipe_center_anchor_id": r.get("swipe_center_anchor_id") or ""}


def _profiles(cache: _Curves, rows: list) -> tuple:
    """(ids, profiles, xy, up, holes) for the rows whose curve the source provides."""
    ids, profs, xy, up, holes = [], [], [], [], {}
    for r in rows:
        oid = r["observation_id"]
        if cache.get(oid) is None:
            holes[oid] = cache.refused[oid]
            continue
        ids.append(oid)
        profs.append(cache.profiles[oid])
        xy.append((r["east_m"], r["north_m"]))
        up.append(r["up_m"])
    return ids, profs, np.asarray(xy, dtype=np.float64).reshape(-1, 2), np.asarray(up), holes


def _row_fields(k: int, extra_taus: list) -> list:
    base = ["db", "level", "source", "ref_condition", "holdout_m", "query_id", "site", "split", "query_condition",
            "phase", "leg_axis", "hard_tag", "query_east_m", "query_north_m", "query_up_m",
            "swipe_center_anchor_id", "n_references", "outcome", "d_near_m", "nearest_id", "nearest_score",
            "nearest_lag", "exact_rank", "exact_top1", "exact_in_top_k", "top1_id", "top1_score", "top1_lag",
            "top1_distance_m", "top2_score", "frame_margin", "accepted_frame", "c1_check"]
    region = ["attainable", "region_rank", "region_top1", "region_in_top_k", "false_region",
              "best_in_region_score", "best_out_region_score", "region_margin", "ambiguous",
              "region_level_margin", "accepted_region"]
    fields = base + region
    for t in extra_taus:
        fields += [f"{f}_tau{int(round(t))}" for f in region]
    for j in range(1, k + 1):
        fields += [f"top{j}_id", f"top{j}_score", f"top{j}_distance_m"]
    return fields


# --------------------------------------------------------------------------------------------------
# score stage
# --------------------------------------------------------------------------------------------------

def _variant_rows(cfg: dict, base: Path, level: str, split: str, source: str) -> dict:
    p = _resolve(base, cfg["variant_dir"]) / ("dev" if split == "dev" else "final") / \
        f"variant_final-swipes-{level}_{source}.csv"
    if not p.exists():
        return {}
    return {r["query_id"]: r for r in _read_csv(p)}


def _crosscheck(mine: list, theirs: dict) -> dict:
    """The sparse DAY+CLEAR run must reproduce the committed C1 variant columns exactly."""
    n = 0
    for row in mine:
        t = theirs.get(row["query_id"])
        if t is None or row["outcome"] == "REFUSED":
            continue
        n += 1
        their_rank = t["c1_bounded_lag_ncc_rank"]
        their_rank = None if their_rank in ("", "None") else int(float(their_rank))
        if t["correct_reference"] != row["nearest_id"]:
            raise SystemExit(f"cross-check: nearest reference differs for {row['query_id']}: "
                             f"{t['correct_reference']} vs {row['nearest_id']}")
        if their_rank != row["exact_rank"]:
            raise SystemExit(f"cross-check: rank of nearest differs for {row['query_id']}: "
                             f"{their_rank} vs {row['exact_rank']}")
        if t["c1_bounded_lag_ncc_best_reference"] != row["top1_id"]:
            raise SystemExit(f"cross-check: top-1 differs for {row['query_id']}")
        if abs(float(t["c1_bounded_lag_ncc_best_score"]) - row["top1_score"]) > 1e-9:
            raise SystemExit(f"cross-check: top-1 score differs for {row['query_id']}")
    return {"n_compared": n, "agree": True}


def _score_sparse(level, source, cond, refs, queries, cache, cfg, cam, extra_taus) -> list:
    k = int(cfg["recall_k"])
    acc = recog.Acceptance(float(cfg["acceptance"]["theta_s"]), float(cfg["acceptance"]["theta_m"]))
    max_lag = int(cfg["max_lag_samples"])
    tau = float(cfg["tau_region_m"])
    ids, profs, xy, _, holes = _profiles(cache, refs)
    rows = []
    for q in queries:
        meta = {"db": "sparse", "level": level, "source": source, "ref_condition": cond, "holdout_m": None,
                **_query_meta(q), "c1_check": "frozen_match"}
        if cache.get(q["observation_id"]) is None:
            rows.append({**meta, "n_references": len(ids), "outcome": "REFUSED"})
            continue
        qp = cache.profiles[q["observation_id"]]
        res = [frozen_c1(qp, rp, max_lag=max_lag) for rp in profs]
        scores = np.array([r.score for r in res], dtype=np.float64)
        lags = np.array([r.shift for r in res], dtype=np.float64)
        out = recog.retrieval_outcome((q["east_m"], q["north_m"]), ids, xy, scores, lags, tau_region=tau,
                                      k=k, acceptance=acc, extra_taus=extra_taus)
        rows.append({**meta, **out})
    return rows, holes


def _score_dense(level, source, refs, queries, cache, cfg, out_dir, extra_taus) -> tuple:
    k = int(cfg["recall_k"])
    acc = recog.Acceptance(float(cfg["acceptance"]["theta_s"]), float(cfg["acceptance"]["theta_m"]))
    max_lag = int(cfg["max_lag_samples"])
    tau = float(cfg["tau_region_m"])
    every = int(cfg["frozen_check_every"])
    ids, profs, xy, _, holes = _profiles(cache, refs)
    bank = recog.ReferenceBank(ids, profs, max_lag=max_lag, min_overlap_frac=float(cfg["min_overlap_frac"]))
    q_ids, S, L, q_xy, checks = [], [], [], [], 0
    rows = []
    usable = [q for q in queries if cache.get(q["observation_id"]) is not None]
    refused = [q["observation_id"] for q in queries if cache.get(q["observation_id"]) is None]
    for qi, q in enumerate(usable):
        qp = cache.profiles[q["observation_id"]]
        scores, lags = bank.score(qp)
        if qi % every == 0:                       # the batched search must equal the frozen matcher
            j = qi % len(ids)
            fr = frozen_c1(qp, profs[j], max_lag=max_lag)
            if not (abs(fr.score - scores[j]) <= 1e-12 and fr.shift == lags[j]):
                raise SystemExit(f"dense cross-check FAILED for {q['observation_id']} vs {ids[j]}: "
                                 f"{fr.score}/{fr.shift} vs {scores[j]}/{lags[j]}")
            checks += 1
        q_ids.append(q["observation_id"])
        S.append(scores)
        L.append(lags)
        q_xy.append((q["east_m"], q["north_m"]))
        dist = np.hypot(xy[:, 0] - q["east_m"], xy[:, 1] - q["north_m"])
        not_self = np.array([rid != q["observation_id"] for rid in ids])
        meta = {"db": "dense", "level": level, "source": source, "ref_condition": "DAY+CLEAR(anchors)+swipes",
                **_query_meta(q), "c1_check": "bank_lag_exact_score_1e-12"}
        for h in cfg["dense_holdout_radii_m"]:
            keep = not_self & (dist > float(h))
            out = recog.retrieval_outcome((q["east_m"], q["north_m"]), ids, xy, scores, lags, keep=keep,
                                          tau_region=tau, k=k, acceptance=acc, extra_taus=extra_taus)
            rows.append({**meta, "holdout_m": float(h), **out})
    np.savez_compressed(out_dir / f"scores_dense_{level}_{source}.npz",
                        scores=np.asarray(S), lags=np.asarray(L), query_ids=np.asarray(q_ids),
                        reference_ids=np.asarray(ids), reference_xy=xy, query_xy=np.asarray(q_xy))
    return rows, {"holes": holes, "refused_queries": refused, "n_references": len(ids),
                  "n_queries": len(usable), "frozen_cross_checks": checks}


def _cell_write(cell_dir: Path, name: str, rows: list, fields: list, info: dict) -> None:
    import csv
    cell_dir.mkdir(parents=True, exist_ok=True)
    with _open(cell_dir / f"{name}.csv.gz", "w") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    _json(cell_dir / f"{name}.json", info)


def _cell_done(cell_dir: Path, name: str) -> bool:
    return (cell_dir / f"{name}.csv.gz").exists() and (cell_dir / f"{name}.json").exists()


def _assemble(cell_dir: Path, names: list, target: Path, fields: list) -> int:
    """Stream the per-cell tables into one table without holding them all in memory."""
    import csv
    n = 0
    with _open(target, "w") as out_fh:
        w = csv.DictWriter(out_fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for name in names:
            with _open(cell_dir / f"{name}.csv.gz", "r") as fh:
                for r in csv.DictReader(fh):
                    w.writerow(r)
                    n += 1
    return n


def stage_score(cfg: dict, base: Path) -> None:
    """Per-cell checkpointing: every (level, source, database/condition) cell is written as soon as it
    is scored and skipped on a re-run, so an interrupted run resumes instead of restarting."""
    t0 = time.time()
    out = _resolve(base, cfg["out_dir"])
    cell_dir = out / "cells"
    out.mkdir(parents=True, exist_ok=True)
    final_cfg = json.loads(_resolve(base, cfg["sim_final_config"]).read_text(encoding="utf-8"))
    store = _resolve(base, cfg["store_root"])
    index_rows = _index(cfg, base)
    session_ids = sorted({r["session_id"] for r in index_rows})
    cam = _camera(store, session_ids, int(cfg["profile_n_samples"]))
    sessions = session_map(store, session_ids)
    sources = _sources(cfg, base, final_cfg, sessions, cam)
    caches = {k: _Curves(s) for k, s in sources.items()}
    extra_taus = [float(t) for t in cfg["tau_region_sensitivity_m"]]
    fields = _row_fields(int(cfg["recall_k"]), extra_taus)

    anchors = defaultdict(list)
    swipes = defaultdict(list)
    for r in index_rows:
        if r["kind"] == "anchor_capture":
            anchors[(r["level_key"], r["condition"])].append(r)
        elif r["kind"] == "swipe":
            swipes[r["level_key"]].append(r)
    manifest = {"study": STUDY, "recog_version": recog.RECOG_VERSION, "camera": cam.as_dict(),
                "sparse": {}, "dense": {}, "crosscheck": {}, "wall_s": None}
    sparse_names, dense_names = [], []
    for level in LEVEL_ORDER:
        queries = sorted(swipes[level], key=lambda r: r["observation_id"])
        split = queries[0]["split"]
        for source, cache in caches.items():
            conds = [cfg["reference_condition"]]
            if source == "segformer":
                conds += list(cfg["appearance_reference_conditions"])
            for cond in conds:
                name = f"sparse_{level}_{source}_{cond}"
                sparse_names.append(name)
                if _cell_done(cell_dir, name):
                    info = json.loads((cell_dir / f"{name}.json").read_text(encoding="utf-8"))
                    print(f"[score] {name}: reused checkpoint")
                else:
                    refs = sorted(anchors[(level, cond)], key=lambda r: r["observation_id"])
                    rows, holes = _score_sparse(level, source, cond, refs, queries, cache, cfg, cam, extra_taus)
                    info = {"n_references": len(refs) - len(holes), "holes": holes, "n_queries": len(rows),
                            "n_refused_queries": sum(1 for r in rows if r["outcome"] == "REFUSED")}
                    if cond == cfg["reference_condition"]:
                        info["crosscheck"] = _crosscheck(rows, _variant_rows(cfg, base, level, split, source))
                    _cell_write(cell_dir, name, rows, fields, info)
                    del rows
                    print(f"[score] {name}: {info['n_references']} refs, {info['n_queries']} queries"
                          + (f"; variant cross-check agree on {info['crosscheck']['n_compared']}"
                             if "crosscheck" in info else "") + f" ({time.time() - t0:.0f}s)")
                manifest["sparse"][f"{level}/{source}/{cond}"] = {k: v for k, v in info.items() if k != "crosscheck"}
                if "crosscheck" in info:
                    manifest["crosscheck"][f"{level}/{source}"] = info["crosscheck"]
            name = f"dense_{level}_{source}"
            dense_names.append(name)
            if _cell_done(cell_dir, name):
                info = json.loads((cell_dir / f"{name}.json").read_text(encoding="utf-8"))
                print(f"[score] {name}: reused checkpoint")
            else:
                horiz = [q for q in queries if q["leg_axis"] != "vertical"]
                dense_refs = horiz + sorted(anchors[(level, cfg["reference_condition"])],
                                            key=lambda r: r["observation_id"])
                rows, info = _score_dense(level, source, dense_refs, horiz, cache, cfg, out, extra_taus)
                _cell_write(cell_dir, name, rows, fields, info)
                del rows
                print(f"[score] {name}: {info['n_references']} refs x {info['n_queries']} queries, "
                      f"{info['frozen_cross_checks']} frozen cross-checks ({time.time() - t0:.0f}s)")
            manifest["dense"][f"{level}/{source}"] = info
    n_sparse = _assemble(cell_dir, sparse_names, out / SPARSE_TABLE, fields)
    n_dense = _assemble(cell_dir, dense_names, out / DENSE_TABLE, fields)
    manifest["wall_s"] = round(time.time() - t0, 1)
    manifest["n_rows"] = {"sparse": n_sparse, "dense": n_dense}
    _json(out / "score_manifest.json", manifest)
    print(f"[score] done: {n_sparse} sparse rows, {n_dense} dense rows in {manifest['wall_s']}s")


# --------------------------------------------------------------------------------------------------
# report stage
# --------------------------------------------------------------------------------------------------

def _parse(rows: list) -> list:
    out = []
    for r in rows:
        p = {}
        for k, v in r.items():
            if k in BOOL_FIELDS or k.startswith(("attainable_tau", "region_top1_tau", "region_in_top_k_tau",
                                                  "false_region_tau", "ambiguous_tau", "accepted_region_tau",
                                                  "exact_top1", "exact_in_top_k")):
                p[k] = (v == "True")
            elif k in ("db", "level", "source", "ref_condition", "query_id", "site", "split", "query_condition",
                       "phase", "leg_axis", "swipe_center_anchor_id", "outcome", "nearest_id", "top1_id",
                       "c1_check") or k.endswith("_id"):
                p[k] = v
            else:
                x = _f(v)
                p[k] = None if (isinstance(x, float) and math.isnan(x)) else x
        out.append(p)
    return out


def _load_tables(out: Path) -> tuple:
    sparse = _parse(_read_csv(out / SPARSE_TABLE))
    import csv
    with _open(out / DENSE_TABLE, "rt") as fh:
        dense = _parse(list(csv.DictReader(fh)))
    return sparse, dense


def _primary(rows: list) -> list:
    return [r for r in rows if r["outcome"] != "REFUSED" and not r["hard_tag"] and r["leg_axis"] != "vertical"]


def _dedupe_dense(rows: list) -> list:
    """Two hold-out radii that leave the same nearest reference give the identical instance."""
    seen, out = set(), []
    for r in rows:
        key = (r["query_id"], r["source"], r["nearest_id"], r["n_references"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _remap_tau(rows: list, tau: float) -> list:
    suf = f"_tau{int(round(tau))}"
    keys = ["attainable", "region_top1", "region_in_top_k", "false_region", "best_in_region_score",
            "region_margin", "ambiguous", "accepted_region"]
    out = []
    for r in rows:
        r2 = dict(r)
        for k in keys:
            r2[k] = r[k + suf]
        out.append(r2)
    return out


def _pooled(rows: list, k: int) -> dict:
    n = len(rows)
    att = [r for r in rows if r["attainable"]]
    acc = [r for r in rows if r["accepted_frame"]]
    return {"n": n, "n_sites": len({r["site"] for r in rows}), "n_attainable": len(att),
            "region_top1": recog._rate([r["region_top1"] for r in rows]),
            "region_recall_k": recog._rate([r["region_in_top_k"] for r in rows]),
            "exact_top1": recog._rate([r["exact_top1"] for r in rows]),
            "exact_recall_k": recog._rate([r["exact_in_top_k"] for r in rows]),
            "coverage_frame": recog._rate([r["accepted_frame"] for r in rows]),
            "coverage_region": recog._rate([r["accepted_region"] for r in rows]),
            "p_false_given_accepted": (sum(1 for r in acc if r["false_region"]) / len(acc)) if acc else None,
            "n_accepted": len(acc), "k": k}


def _group(rows: list, cfg: dict, k: int, bins_key: str = "bins_m") -> dict:
    table = recog.bin_table(rows, cfg[bins_key], cfg["support"], k=k)
    radius = recog.radius_from_bins(table, cfg["radius_rule"])
    return {"bins": table, "radius": radius,
            "spacing_conservative": recog.spacing_implication(radius["conservative_m"]),
            "spacing_recoverable": recog.spacing_implication(radius["recoverable_m"]),
            "pooled": _pooled(rows, k)}


def _by(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def _pct(v) -> str:
    return "—" if v is None else f"{100 * v:.0f} %"


def _num(v, nd=2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _bin_md(table: list) -> list:
    md = ["| bin (m) | n | sites | attain. | region top-1 (site min) | region R@5 | exact top-1 | top-1 score med | "
          "corr.-region score med | region margin med | false region | acc. false region | P(false|acc) | "
          "coverage frame / region | P(false|acc region) | ambiguous | fail w/ nearby ref | no ref |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in table:
        if e["n"] == 0:
            continue
        sup = "" if e["supported"] else " ⚠"
        md.append(f"| {e['bin']}{sup} | {e['n']} | {e['n_sites']} | {e['n_attainable']} | "
                  f"{_pct(e['region_top1'])} ({_pct(e['site_region_top1_min'])}) | {_pct(e['region_recall_k'])} | "
                  f"{_pct(e['exact_top1'])} | {_num(e['top1_score_median'], 3)} | "
                  f"{_num(e['correct_region_score_median'], 3)} | {_num(e['region_margin_median'], 3)} | "
                  f"{_pct(e['false_region_rate'])} | {_pct(e['accepted_false_region_rate'])} | "
                  f"{_pct(e['p_false_given_accepted'])} | {_pct(e['coverage_frame'])} / {_pct(e['coverage_region'])} | "
                  f"{_pct(e['p_false_given_accepted_region'])} | "
                  f"{_pct(e['ambiguity_rate'])} | {e['n_matcher_failed_with_nearby_reference']} | "
                  f"{e['n_no_suitable_reference']} |")
    return md


def _radius_md(name: str, g: dict) -> str:
    r, sc = g["radius"], g["spacing_conservative"]
    return (f"| {name} | {_num(r['conservative_m'], 1)} | {_num(r['recoverable_m'], 1)} | "
            f"{(r['conservative_stop'] or {}).get('bin', '—')}: {(r['conservative_stop'] or {}).get('reason', '—')} | "
            f"{_num(sc['theoretical_max_spacing_m'], 0)} | {_num(sc['conservative_spacing_m'], 0)} | "
            f"{sc['assessment']} |")


def _fig_rate_vs_bin(out: Path, name: str, groups: dict, key: str, ylabel: str, title: str, bins: list,
                     panels: list, hline=None):
    plt = _plt()
    labels = [e["bin"] for e in next(iter(groups.values()))["bins"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (ptitle, selector) in zip(axes, panels):
        for level in LEVEL_ORDER:
            for src, ls in (("sim_exact", "-"), ("segformer", "--")):
                g = groups.get(selector(level, src))
                if g is None:
                    continue
                ys = [e[key] if (e["n"] > 0 and e[key] is not None) else np.nan for e in g["bins"]]
                ax.plot(range(len(labels)), ys, ls, marker="o", color=LEVEL_COLOR[level],
                        label=f"{level} {'GT' if src == 'sim_exact' else 'SegFormer'}", ms=4)
                for i, e in enumerate(g["bins"]):
                    if e["n"] and not e["supported"]:
                        ax.plot(i, ys[i], "x", color="k", ms=8)
        if hline is not None:
            ax.axhline(hline, color="grey", lw=0.8, ls=":")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_xlabel("distance to nearest stored reference d_near (m)")
        ax.set_title(ptitle, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title} — x marks an unsupported bin (< 30 queries or < 3 sites)", fontsize=9)
    return _save(fig, out / "figures" / name)


def _fig_scatter(out: Path, sparse_rows: list, cfg: dict):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), sharey=True)
    for ax, level in zip(axes, LEVEL_ORDER):
        rows = [r for r in sparse_rows if r["level"] == level and r["source"] == "sim_exact"
                and r["ref_condition"] == cfg["reference_condition"]]
        for flag, color, lab in ((True, LEVEL_COLOR[level], "region-correct top-1"), (False, "#cc2222", "false region")):
            sel = [r for r in rows if r["region_top1"] == flag and r["top1_score"] is not None
                   and np.isfinite(r["top1_score"])]
            ax.scatter([r["d_near_m"] for r in sel], [r["top1_score"] for r in sel], s=9, alpha=0.55,
                       color=color, label=lab)
        ax.axhline(cfg["acceptance"]["theta_s"], color="grey", ls=":", lw=0.8)
        ax.set_title(f"{level} — sparse anchor database, GT curves", fontsize=10)
        ax.set_xlabel("d_near (m)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("top-1 C1-32 score")
    axes[0].legend(fontsize=8)
    return _save(fig, out / "figures" / "fig8_score_vs_distance_scatter.jpg")


def _fig_radius(out: Path, summary: dict):
    plt = _plt()
    names = list(summary.keys())
    cons = [summary[n]["radius"]["conservative_m"] or 0 for n in names]
    reco = [summary[n]["radius"]["recoverable_m"] or 0 for n in names]
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(names)), 4.2))
    x = np.arange(len(names))
    ax.bar(x - 0.2, cons, 0.4, label="conservative (region top-1 ≥ 0.9 ∧ accepted false ≤ 2 %)", color="#335588")
    ax.bar(x + 0.2, reco, 0.4, label="recoverable (region R@5 ≥ 0.9)", color="#88aacc")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("radius (m) — 0 = none")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("EXP-SKY-010 recognition radii by the pre-registered rule", fontsize=10)
    return _save(fig, out / "figures" / "fig7_radius_summary.jpg")


def stage_report(cfg: dict, base: Path) -> None:
    out = _resolve(base, cfg["out_dir"])
    k = int(cfg["recall_k"])
    sparse, dense = _load_tables(out)
    manifest = json.loads((out / "score_manifest.json").read_text(encoding="utf-8"))
    ref_cond = cfg["reference_condition"]

    sp = _primary(sparse)
    dn = _dedupe_dense(_primary(dense))
    groups = {}
    for (cond, level, src), rows in _by(sp, "ref_condition", "level", "source").items():
        groups[("sparse", cond, level, src)] = _group(rows, cfg, k)
    for (level, src), rows in _by(dn, "level", "source").items():
        groups[("dense", "path", level, src)] = _group(rows, cfg, k)

    # sensitivity of the region rule
    sensitivity = {}
    for tau in cfg["tau_region_sensitivity_m"]:
        for (level, src), rows in _by([r for r in sp if r["ref_condition"] == ref_cond], "level", "source").items():
            sensitivity[f"sparse/{level}/{src}/tau{int(tau)}"] = _group(_remap_tau(rows, tau), cfg, k)
        for (level, src), rows in _by(dn, "level", "source").items():
            sensitivity[f"dense/{level}/{src}/tau{int(tau)}"] = _group(_remap_tau(rows, tau), cfg, k)

    # slices on the sparse DAY+CLEAR database
    slices = {}
    base_sp = [r for r in sparse if r["outcome"] != "REFUSED" and r["ref_condition"] == ref_cond]
    for (level, src), rows in _by([r for r in base_sp if r["hard_tag"]], "level", "source").items():
        slices[f"hard/{level}/{src}"] = _group(rows, cfg, k)
    for (level, src), rows in _by([r for r in base_sp if r["leg_axis"] == "vertical"], "level", "source").items():
        dz = [abs(r["query_up_m"] - 0.0) for r in rows]
        slices[f"vertical/{level}/{src}"] = {**_pooled(rows, k),
                                             "d_near_median_m": float(np.median([r["d_near_m"] for r in rows])),
                                             "note": "vertical-leg queries: d_near ~ 0-2 m, |dz| up to ~100 m; "
                                                     "reported, excluded from the radius"}
    # dense: exact vs region gap and frame-margin coverage by hold-out radius
    dense_by_h = {}
    for (level, src, h), rows in _by(_primary(dense), "level", "source", "holdout_m").items():
        dense_by_h[f"{level}/{src}/h{h:g}"] = {**_pooled(rows, k),
                                               "d_near_median_m": float(np.median([r["d_near_m"] for r in rows]))}
    coincide = {}
    for (level, src), rows in _by([r for r in sp if r["ref_condition"] == ref_cond], "level", "source").items():
        coincide[f"{level}/{src}"] = sum(1 for r in rows if r["exact_top1"] != r["region_top1"])
    # POST-HOC (declared in the config comment): the merged 0-10 m first bin, sparse and dense alike
    posthoc = {}
    if "posthoc_bins_m" in cfg:
        for (cond, level, src), rows in _by(sp, "ref_condition", "level", "source").items():
            posthoc[f"sparse/{cond}/{level}/{'GT' if src == 'sim_exact' else 'SegF'}"] = _group(rows, cfg, k, "posthoc_bins_m")
        for (level, src), rows in _by(dn, "level", "source").items():
            posthoc[f"dense/path/{level}/{'GT' if src == 'sim_exact' else 'SegF'}"] = _group(rows, cfg, k, "posthoc_bins_m")

    summary = {}
    for key, g in groups.items():
        db, cond, level, src = key
        name = f"{db}/{cond}/{level}/{'GT' if src == 'sim_exact' else 'SegF'}"
        summary[name] = g
    metrics = {"study": STUDY, "recog_version": recog.RECOG_VERSION, "config": cfg, "manifest": manifest,
               "n_primary_rows": {"sparse": len(sp), "dense_instances": len(dn)},
               "groups": summary, "sensitivity": sensitivity, "slices": slices, "dense_by_holdout": dense_by_h,
               "exact_vs_region_disagreements_sparse": coincide, "posthoc_merged_first_bin": posthoc}
    _json(out / "metrics.json", metrics)

    # figures (7 + scatter)
    bins = cfg["bins_m"]
    sel_sparse = lambda level, src: ("sparse", ref_cond, level, src)   # noqa: E731
    sel_dense = lambda level, src: ("dense", "path", level, src)       # noqa: E731
    panels = [("sparse anchor database (10/30/27 refs)", sel_sparse), ("dense path database (hold-out radii)", sel_dense)]
    figs = [
        _fig_rate_vs_bin(out, "fig1_region_top1_vs_distance.jpg", groups, "region_top1", "region-correct top-1",
                         "Region-correct top-1 vs separation", bins, panels, hline=cfg["radius_rule"]["region_top1_min"]),
        _fig_rate_vs_bin(out, "fig2_region_recall5_vs_distance.jpg", groups, "region_recall_k",
                         f"region-correct Recall@{k}", f"Region-correct Recall@{k} vs separation", bins, panels,
                         hline=cfg["radius_rule"]["region_recall_k_min"]),
        _fig_rate_vs_bin(out, "fig3_top1_score_vs_distance.jpg", groups, "top1_score_median", "median top-1 C1-32 score",
                         "Match strength vs separation", bins, panels, hline=cfg["acceptance"]["theta_s"]),
        _fig_rate_vs_bin(out, "fig4_region_margin_vs_distance.jpg", groups, "region_margin_median",
                         "median region margin (best in-region − best out-of-region)",
                         "Ambiguity vs separation", bins, panels, hline=cfg["acceptance"]["theta_m"]),
        _fig_rate_vs_bin(out, "fig5_false_region_vs_distance.jpg", groups, "false_region_rate", "false-region rate",
                         "False-region rate vs separation", bins, panels),
        _fig_rate_vs_bin(out, "fig5b_accepted_false_region_vs_distance.jpg", groups, "accepted_false_region_rate",
                         "accepted-false-region rate (frozen accept-v2-margin)",
                         "Confident false regions vs separation", bins, panels,
                         hline=cfg["radius_rule"]["accepted_false_region_max"]),
    ]
    app_panels = [(f"SegFormer, references {c}", (lambda c: (lambda level, src: ("sparse", c, level, src)))(c))
                  for c in [ref_cond] + list(cfg["appearance_reference_conditions"])]
    seg_groups = {kk: v for kk, v in groups.items() if kk[3] == "segformer" and kk[0] == "sparse"}
    figs.append(_fig_rate_vs_bin(out, "fig6_appearance_region_top1.jpg", seg_groups, "region_top1",
                                 "region-correct top-1", "Appearance check (DAY+CLEAR queries vs reference condition)",
                                 bins, app_panels, hline=cfg["radius_rule"]["region_top1_min"]))
    figs.append(_fig_radius(out, summary))
    figs.append(_fig_scatter(out, sparse, cfg))

    # report.md
    md = [f"# {STUDY} — C1-32 recognition radius vs true horizontal separation", "",
          f"Generated by `scripts/sky_recog_radius.py --stage report` from `{SPARSE_TABLE}` / `{DENSE_TABLE}`. "
          f"Region rule τ = {cfg['tau_region_m']:g} m; acceptance `accept-v2-margin` "
          f"θ_s = {cfg['acceptance']['theta_s']}, θ_m = {cfg['acceptance']['theta_m']}; K = {k}; "
          f"support ≥ {cfg['support']['min_queries']} queries ∧ ≥ {cfg['support']['min_sites']} sites (⚠ = unsupported).",
          "", "Primary population: non-hard sites, horizontal legs + centre frames. Sparse = the level's DAY+CLEAR "
          "anchor database (frozen `match()`); dense = horizontal swipe frames + anchors, hold-out radii applied "
          "after scoring, one row per distinct nearest-reference instance.", "",
          "## Cross-check against the committed C1 variant tables", "",
          "| level/source | queries compared | agree |", "|---|---|---|"]
    for kk, v in manifest["crosscheck"].items():
        md.append(f"| {kk} | {v['n_compared']} | {v['agree']} |")
    md += ["", f"Dense frozen cross-checks: " + ", ".join(f"{kk}: {v['frozen_cross_checks']}"
                                                          for kk, v in manifest["dense"].items()), "",
           "Exact-top-1 ≠ region-top-1 disagreements on the sparse database (expected 0 — anchors ≥ 265 m apart): "
           + json.dumps(coincide), ""]
    md += ["## Recognition radii (pre-registered rule)", "",
           "| database / condition / level / source | R_conservative (m) | R_recoverable (m) | conservative walk stopped at | "
           "2R (m) | s ≈ R (m) | 10 m capture spacing |", "|---|---|---|---|---|---|---|"]
    for name, g in summary.items():
        md.append(_radius_md(name, g))
    md.append("")
    if posthoc:
        md += ["## POST-HOC: merged 0–10 m first bin (not pre-registered — see config comment)", "",
               "| database / condition / level / source | R_conservative (m) | R_recoverable (m) | conservative walk stopped at | "
               "2R (m) | s ≈ R (m) | 10 m capture spacing |", "|---|---|---|---|---|---|---|"]
        for name, g in posthoc.items():
            md.append(_radius_md(name, g))
        md.append("")
    for name, g in summary.items():
        p = g["pooled"]
        md += [f"## {name}", "",
               f"Pooled: n = {p['n']} ({p['n_sites']} sites, {p['n_attainable']} attainable) — region top-1 "
               f"{_pct(p['region_top1'])}, region R@{k} {_pct(p['region_recall_k'])}, exact top-1 {_pct(p['exact_top1'])}, "
               f"coverage frame/region {_pct(p['coverage_frame'])}/{_pct(p['coverage_region'])}, "
               f"P(false region | accepted) {_pct(p['p_false_given_accepted'])} (n_acc = {p['n_accepted']}).", ""]
        md += _bin_md(g["bins"])
        md.append("")
    md += ["## Sensitivity of the region rule", "",
           "| group | τ (m) | pooled region top-1 | R_conservative | R_recoverable |", "|---|---|---|---|---|"]
    for name, g in sensitivity.items():
        md.append(f"| {name.rsplit('/', 1)[0]} | {name.rsplit('tau', 1)[1]} | {_pct(g['pooled']['region_top1'])} | "
                  f"{_num(g['radius']['conservative_m'], 1)} | {_num(g['radius']['recoverable_m'], 1)} |")
    md += ["", "## Slices (sparse DAY+CLEAR database)", "",
           "| slice | n | sites | region top-1 | region R@5 | coverage frame | P(false|acc) | note |",
           "|---|---|---|---|---|---|---|---|"]
    for name, g in slices.items():
        p = g["pooled"] if "pooled" in g else g
        md.append(f"| {name} | {p['n']} | {p['n_sites']} | {_pct(p['region_top1'])} | {_pct(p['region_recall_k'])} | "
                  f"{_pct(p['coverage_frame'])} | {_pct(p['p_false_given_accepted'])} | "
                  f"{g.get('note', '')} |")
    md += ["", "## Dense database by hold-out radius (frame-level vs region-level acceptance)", "",
           "| level/source/h | n | d_near median (m) | region top-1 | exact top-1 | coverage frame | coverage region | "
           "P(false|acc frame) |", "|---|---|---|---|---|---|---|---|"]
    for name, p in dense_by_h.items():
        md.append(f"| {name} | {p['n']} | {p['d_near_median_m']:.1f} | {_pct(p['region_top1'])} | {_pct(p['exact_top1'])} | "
                  f"{_pct(p['coverage_frame'])} | {_pct(p['coverage_region'])} | {_pct(p['p_false_given_accepted'])} |")
    md += ["", "## Figures", ""] + [f"![{p.name}](figures/{p.name})" for p in figs]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[report] wrote metrics.json, report.md, {len(figs)} figures under {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=("score", "report", "all"), default="all")
    a = ap.parse_args(argv)
    cfg, base, _final = _load_cfg(Path(a.config))
    if a.stage in ("score", "all"):
        stage_score(cfg, base)
    if a.stage in ("report", "all"):
        stage_report(cfg, base)


if __name__ == "__main__":
    main()
