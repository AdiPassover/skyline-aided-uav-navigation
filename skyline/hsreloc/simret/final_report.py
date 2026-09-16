"""The reader's report for the simulator pilot (T029; research R10 selection rules).

Assembles ``evaluations/sim-retrieval-report/`` — ``report.md`` plus JPEG figures — from committed
experiment outputs only. Every example is selected by a rule stated in the report itself:

* **pair panels**: per direction (lateral / longitudinal / vertical), the median-displacement and
  the maximum-displacement translation pair;
* **retrieval examples**: every rank-1 miss of the GT and SegFormer sources; DP's first miss per
  time of day in query order; the first rank-1-correct query per source in query order;
* **condition montage**: anchor a2 (the one anchor with a complete, duplicate-free 6-cell matrix).

Nothing here recomputes an experiment: rank-1 correctness is derived from the committed records'
candidate lists exactly as the EXP-SKY-006 report derived it, and every table carries the pilot
caveat from the pre-registration.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from hsreloc.matchers import FrozenNccMatcher
from hsreloc.retrieval.profile import ProfileConfig, normalize
from hsreloc.retrieval.skyline_curve import CurveError
from hsreloc.simret import groups, render, sources
from hsreloc.simret.prereg import LIMITATIONS

FIGSIZE_MAP = (7.5, 7.5)


def _resolve(base: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def _read_csv(path: Path) -> list:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------------------------------
# rank-1 correctness from the committed records (the EXP-SKY-006 shortlist convention)
# --------------------------------------------------------------------------------------------------

def rank1_rows(cfg: dict, base: Path, source_key: str) -> list:
    task, spacing = cfg["task_name"], float(cfg["tasks"]["spacings_m"][0])
    ds = _resolve(base, cfg["datasets_root"]) / f"sim-{task}-s{spacing:g}-{source_key}"
    side = _read_csv(ds / "skyline_queries.csv")
    rec = _read_csv(_resolve(base, cfg["runs_root"])
                    / f"sim-{task}-s{spacing:g}-ncc-{source_key}" / "queries.csv")
    refs = {r["reference_id"]: (float(r["east_m"]), float(r["north_m"]))
            for r in _read_csv(_resolve(base, cfg["refdb_root"])
                               / f"sim-refs-{task}-s{spacing:g}-{source_key}" / "references.csv")}
    rows = []
    for sd, rr in zip(side, rec):
        row = {"observation_id": sd["condition_observation_id"],
               "session_id": sd["condition_session_id"],
               "stage": sd["condition_stage"],
               "translation_m": float(sd["condition_translation_m"]),
               "translation_bin": sd["condition_translation_bin"],
               "along_m": float(sd["condition_along_m"] or 0.0),
               "lateral_m": float(sd["condition_lateral_m"] or 0.0),
               "up_diff_m": float(sd["condition_up_diff_m"] or 0.0),
               "time_of_day": sd["condition_time_of_day"], "clouds": sd["condition_clouds"],
               "anchor_id": sd["condition_anchor_id"],
               "correct_reference": sd["condition_nearest_reference_id"],
               "outcome": rr["outcome"],
               "best_score": float(rr["best_score"]) if rr["best_score"] else None,
               "score_margin": float(rr["score_margin"]) if rr["score_margin"] else None,
               "scored": bool(rr["cand0_east_m"])}
        if row["scored"]:
            ce, cn = refs[row["correct_reference"]]
            row["top1_correct"] = math.hypot(float(rr["cand0_east_m"]) - ce,
                                             float(rr["cand0_north_m"]) - cn) < 1.0
            row["accepted"] = rr["outcome"] == "SUCCESS"
            row["confident_false"] = row["accepted"] and not row["top1_correct"]
        else:
            row["top1_correct"] = None
            row["accepted"] = False
            row["confident_false"] = False
        rows.append(row)
    return rows


def _rate_table(rows: list, key) -> dict:
    out: dict = {}
    for r in rows:
        k = key(r) if callable(key) else r.get(key)
        g = out.setdefault(str(k), {"n": 0, "scored": 0, "top1": 0, "accepted": 0, "conf_false": 0})
        g["n"] += 1
        if r["scored"]:
            g["scored"] += 1
            g["top1"] += bool(r["top1_correct"])
            g["accepted"] += bool(r["accepted"])
            g["conf_false"] += bool(r["confident_false"])
    return dict(sorted(out.items()))


def _md_rate(table: dict, label: str) -> list:
    lines = [f"| {label} | n | scored | rank-1 correct | accepted | confident-false |",
             "|---|---|---|---|---|---|"]
    for k, g in table.items():
        frac = f"{g['top1']}/{g['scored']}" + (f" ({g['top1'] / g['scored']:.0%})" if g["scored"] else "")
        lines.append(f"| {k} | {g['n']} | {g['scored']} | {frac} | {g['accepted']} | {g['conf_false']} |")
    return lines


# --------------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------------

def fig_anchor_map(index_rows: list, classification: dict, out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGSIZE_MAP)
    cls_by_session = {t["run_folder"]: t["class"] for t in classification["translation_runs"]}
    colors = {"lateral": "#1f77b4", "longitudinal": "#2ca02c", "vertical": "#9467bd"}
    seen = set()
    for r in index_rows:
        e, n = float(r["east_m"]), float(r["north_m"])
        klass = cls_by_session.get(r["session_id"])
        if klass:
            label = f"{klass} sweep" if klass not in seen else None
            seen.add(klass)
            ax.plot(e, n, ".", ms=5, color=colors.get(klass, "#888888"), label=label)
        else:
            ax.plot(e, n, "o", ms=4, color="#d62728", alpha=0.35)
    anchors = {}
    for r in index_rows:
        if r["anchor_id"] and r["session_id"] not in cls_by_session:
            anchors[r["anchor_id"]] = (float(r["east_m"]), float(r["north_m"]))
    for aid, (e, n) in sorted(anchors.items()):
        ax.plot(e, n, "k*", ms=14)
        ax.annotate(aid, (e, n), textcoords="offset points", xytext=(8, 6), fontsize=11)
    ax.set_xlabel("East (m, synthetic local frame)")
    ax.set_ylabel("North (m)")
    ax.set_title("Recovered anchors and translation sweeps (poses only)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_condition_montage(cfg: dict, base: Path, index_rows: list, anchor: str, out: Path) -> Path:
    plt = _plt()
    import cv2
    store = _resolve(base, cfg["store_root"])
    members = sorted((r for r in index_rows if r["anchor_id"] == anchor
                      and r["session_id"] == r["observation_id"].split("__")[0]
                      and r["condition_time_of_day"]),
                     key=lambda r: (r["condition_time_of_day"], r["condition_clouds"]))
    # keep the single-observation captures only, one per condition cell
    cells, seen = [], set()
    for r in members:
        key = (r["condition_time_of_day"], r["condition_clouds"])
        if key not in seen:
            seen.add(key)
            cells.append(r)
    n = len(cells)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, cells):
        img = cv2.imread(str(store / r["session_id"] / "images" / f"{r['observation_id']}.png"),
                         cv2.IMREAD_COLOR)
        if img is not None:
            ax.imshow(img[:, :, ::-1])
        ax.set_title(f"{r['condition_time_of_day']}+{r['condition_clouds']}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Anchor {anchor}: one frozen pose under every captured condition", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_rank1_vs_bin(all_rows: dict, out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, rows in all_rows.items():
        table = _rate_table([r for r in rows if r["stage"] == "stage1"], "translation_bin")
        xs, ys = [], []
        for b, g in sorted(table.items(), key=lambda kv: float(kv[0].split("-")[0])
                           if kv[0] not in ("", "None") else -1):
            if g["scored"]:
                xs.append(b)
                ys.append(g["top1"] / g["scored"])
        style = render.SOURCE_STYLE[sources.PROVENANCE_BY_KEY[key]]
        ax.plot(xs, ys, "o-", color=style["color"], label=style["label"])
    ax.axhline(1 / 6, color="#999999", lw=1.0, ls=":", label="chance (1/6)")
    ax.set_xlabel("horizontal translation bin (m)")
    ax.set_ylabel("rank-1 correct fraction (stage-1 queries)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Retrieval vs translation — 6-reference pilot database")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _pair_panel(cfg, base, built, obs_by_id, store, smap, row, refs_by_id, out_path, title):
    """One deformation/retrieval pair through render.render_pair, all three sources drawn."""
    qid, rid = row["query_id"], row["reference_id"]
    profile_config = ProfileConfig()

    def curves_of(oid):
        out = {}
        for key, src in built.items():
            try:
                out[sources.PROVENANCE_BY_KEY[key]] = np.asarray(src.get(oid).row_per_col)
            except CurveError:
                pass
        return out

    def image_of(oid):
        return store / smap[oid] / "images" / f"{oid}.png"

    matcher = FrozenNccMatcher()
    scores = {}
    try:
        qp = normalize(built["sim_exact"].get(qid), profile_config)
        rp = normalize(built["sim_exact"].get(rid), profile_config)
        res = matcher.match(qp, rp)
        scores["c0_frozen_ncc (GT curves)"] = {"score": res.score, "shift": res.shift,
                                               "scale": res.scale,
                                               "warp_magnitude": res.warp_magnitude,
                                               "accepted": res.accepted}
    except CurveError:
        pass
    offset = {k: row.get(k) for k in ("translation_m", "along_m", "lateral_m", "up_m",
                                      "yaw_diff_deg", "pitch_diff_deg", "roll_diff_deg")}
    return render.render_pair(
        out_path,
        {"observation_id": qid, "image_path": image_of(qid), "curves": curves_of(qid)},
        {"observation_id": rid, "image_path": image_of(rid), "curves": curves_of(rid)},
        offset, scores, correct=bool(row.get("correct", True)), title=title)


# --------------------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------------------

def build(cfg: dict, base: Path) -> dict:
    out_dir = _resolve(base, cfg["report_out"])
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    store = _resolve(base, cfg["store_root"])

    index = groups.load_index(_resolve(base, cfg["groups_out"]))
    classification = json.loads(_resolve(base, cfg["experiments"]["classification_json"])
                                .read_text(encoding="utf-8"))
    deform_rows = _read_csv(_resolve(base, cfg["experiments"]["deformation_out"])
                            / "deformation_gt.csv")
    for r in deform_rows:
        for k in ("translation_m", "along_m", "lateral_m", "up_m", "curve_median_abs_px",
                  "curve_offset_px", "curve_median_abs_offset_removed_px", "axis_displacement_m",
                  "yaw_diff_deg", "pitch_diff_deg", "roll_diff_deg"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None

    smap = sources.session_map(store)
    meta0 = json.loads((store / sorted(smap.values())[0] / "session.json")
                       .read_text(encoding="utf-8"))
    width, height = meta0["camera"]["resolution_px"]
    built = {key: sources.build_source(key, spec, smap, width, height, store_root=store,
                                       config_dir=base)
             for key, spec in cfg["sources"].items()}

    produced = {}
    produced["anchor_map"] = fig_anchor_map(index["rows"], classification,
                                            figs / "fig_anchor_map.jpg")
    produced["montage"] = fig_condition_montage(cfg, base, index["rows"], "a2",
                                                figs / "fig_condition_montage_a2.jpg")

    # deformation trend per direction
    for klass in ("lateral", "longitudinal", "vertical"):
        rows = [r for r in deform_rows if r["direction_class"] == klass]
        if rows:
            produced[f"deform_{klass}"] = render.plot_metric_vs_axis(
                figs / f"fig_deformation_{klass}.jpg", rows,
                "axis_displacement_bin", "curve_median_abs_px",
                f"GT skyline deformation vs {klass} displacement",
                "median |Δrow| (px, raw)")
    produced["lat_vs_lon"] = render.plot_lateral_vs_longitudinal(
        figs / "fig_lateral_vs_longitudinal.jpg",
        [r for r in deform_rows if r["direction_class"] != "vertical"],
        "curve_median_abs_px", "GT median |Δrow| (px)")

    # extraction error by condition, per source
    acc_rows = []
    for key in ("segformer", "dp"):
        for r in _read_csv(_resolve(base, cfg["experiments"]["extraction_out"])
                           / f"accuracy_{key}.csv"):
            acc_rows.append({"source": key, "time_of_day": r["time_of_day"],
                             "clouds": r["clouds"],
                             "median_abs_px": float(r["median_abs_px"])})
    for axis in ("time_of_day", "clouds"):
        produced[f"acc_{axis}"] = render.plot_metric_vs_axis(
            figs / f"fig_extraction_error_by_{axis}.jpg", acc_rows, axis, "median_abs_px",
            f"Extraction error vs GT by {axis} (answered frames only)",
            "per-frame median |err| (px)", series_field="source")

    # rank-1 retrieval
    all_rows = {key: rank1_rows(cfg, base, key) for key in cfg["sources"]}
    produced["rank1"] = fig_rank1_vs_bin(all_rows, figs / "fig_rank1_vs_translation.jpg")

    # pair panels — deformation examples (median and max displacement per direction)
    obs_by_id = None
    pair_files = []
    for klass in ("lateral", "longitudinal", "vertical"):
        rows = sorted((r for r in deform_rows if r["direction_class"] == klass
                       and r["axis_displacement_m"]),
                      key=lambda r: r["axis_displacement_m"])
        if not rows:
            continue
        for tag, row in (("median", rows[len(rows) // 2]), ("max", rows[-1])):
            out = figs / "pairs" / f"deform_{klass}_{tag}_{row['query_id']}.jpg"
            _pair_panel(cfg, base, built, obs_by_id, store, smap, row, None, out,
                        f"{klass} {tag}: {row['axis_displacement_m']:.1f} m")
            pair_files.append(out.relative_to(out_dir).as_posix())

    # retrieval example panels (rule in the module docstring)
    retr_files = []
    deform_by_qid = {r["query_id"]: r for r in deform_rows}
    for key, rows in all_rows.items():
        misses = [r for r in rows if r["scored"] and not r["top1_correct"]]
        if key == "dp":
            first_per_tod = {}
            for r in misses:
                first_per_tod.setdefault(r["time_of_day"], r)
            misses = list(first_per_tod.values())
        first_correct = next((r for r in rows if r["scored"] and r["top1_correct"]), None)
        for tag, sel in [("miss", m) for m in misses] + ([("correct", first_correct)]
                                                         if first_correct else []):
            base_row = deform_by_qid.get(sel["observation_id"])
            row = base_row or {"query_id": sel["observation_id"],
                               "reference_id": sel["correct_reference"],
                               "translation_m": sel["translation_m"],
                               "along_m": sel["along_m"], "lateral_m": sel["lateral_m"],
                               "up_m": sel["up_diff_m"],
                               "yaw_diff_deg": 0.0, "pitch_diff_deg": 0.0, "roll_diff_deg": 0.0,
                               "direction_class": ""}
            row = dict(row, correct=bool(sel["top1_correct"]))
            out = figs / "retrieval" / f"{key}_{tag}_{sel['observation_id']}.jpg"
            try:
                _pair_panel(cfg, base, built, obs_by_id, store, smap, row, None, out,
                            f"{key} {tag}: {sel['observation_id']} "
                            f"({sel['time_of_day']}+{sel['clouds']}, "
                            f"{sel['translation_m']:.0f} m)")
                retr_files.append(out.relative_to(out_dir).as_posix())
            except (CurveError, render.RenderError):
                continue

    # ---------------- report.md ----------------
    ext = {key: json.loads((_resolve(base, cfg["experiments"]["extraction_out"])
                            / f"accuracy_{key}_summary.json").read_text(encoding="utf-8"))
           for key in ("segformer", "dp")}
    cons = json.loads((_resolve(base, cfg["experiments"]["consistency_out"])
                       / "consistency_summary.json").read_text(encoding="utf-8"))
    evals = {}
    for key in cfg["sources"]:
        p = (_resolve(base, cfg["evaluations_root"])
             / f"sim-retrieval-sim-{cfg['task_name']}-s{float(cfg['tasks']['spacings_m'][0]):g}-ncc-{key}"
             / "metrics.json")
        evals[key] = json.loads(p.read_text(encoding="utf-8"))

    lines = [
        "# Simulator pilot — skyline evaluation and viewpoint sensitivity (EXP-SKY-007)",
        "",
        f"> **Evidence**: {LIMITATIONS}",
        "",
        "Generated by `hsreloc.simret.final_report` from committed experiment outputs; every",
        "example below is selected by the rule stated beside it. Chance baseline for retrieval:",
        "**1/6 references**.",
        "",
        "## 1. The collection",
        "",
        "![anchor map](figures/fig_anchor_map.jpg)",
        "",
        "Six anchors recovered from pose geometry; seven translation sweeps (62–120 m) from a0/a1.",
        "",
        "![condition montage](figures/fig_condition_montage_a2.jpg)",
        "",
        "## 2. Experiment A — extraction accuracy vs simulator GT",
        "",
        "| source | answered | refused | refusal rate |",
        "|---|---|---|---|",
    ]
    for key in ("segformer", "dp"):
        s = ext[key]
        lines.append(f"| {key} | {s['n_evaluated']} | {s['n_refused']} | {s['refusal_rate']:.1%} |")
    lines += [
        "",
        "Per-frame median error by slice (answered frames only; px of 512):",
        "",
        "| slice | SegFormer med / p90 / cat | DP med / p90 / cat |",
        "|---|---|---|",
    ]
    for axis in ("by_time_of_day", "by_clouds"):
        keys = sorted(set(ext["segformer"][axis]) | set(ext["dp"][axis]))
        for k in keys:
            cell = []
            for src in ("segformer", "dp"):
                v = ext[src][axis].get(k)
                if v and "median_abs_px" in v:
                    cell.append(f"{v['median_abs_px']['median']:.1f} / "
                                f"{v['p90_abs_px']['median']:.1f} / "
                                f"{v['catastrophic_frac']['median']:.2f} (n={v['n']})")
                else:
                    cell.append("—")
            lines.append(f"| {k} | {cell[0]} | {cell[1]} |")
    lines += [
        "",
        "![error by time of day](figures/fig_extraction_error_by_time_of_day.jpg)",
        "![error by clouds](figures/fig_extraction_error_by_clouds.jpg)",
        "",
        "DP's 18 refusals are all `inverted_polarity` at DAWN/DUSK (sky darker than terrain);",
        "its answered DAY frames carry a 32 % catastrophic-column median — it reads hazy distant",
        "terrain as sky. SegFormer, out of ADE20K's photographic domain, stays at 2–3 px median",
        "with zero catastrophic columns (DEV inspection: `../sim-exp-extraction/inspection/`).",
        "",
        "## 3. Experiment B — same-pose cross-condition consistency",
        "",
        f"GT control: **0.000 px** raw disagreement on all {cons['n_pairs_per_source'].get('sim_exact')} pairs",
        "(including same-condition rerenders) — the renderer's sky mask is purely geometric, so",
        "geometry invariance holds exactly and extractor rows are interpretable as-is.",
        "SegFormer: 1–5 px median across every condition pair. DP: 0 px within DAY, 6–35 px across",
        f"time-of-day, and it refuses most DAWN/DUSK members ({len(cons['refusals'])} refused pairs).",
        f"Catastrophic switches: {cons['catastrophic_counts']} (threshold "
        f"{cons['catastrophic_frac_height']:.0%} of height).",
        "",
        "## 4. Experiment C — GT skyline deformation under translation",
        "",
        "![lateral](figures/fig_deformation_lateral.jpg)",
        "![longitudinal](figures/fig_deformation_longitudinal.jpg)",
        "![vertical](figures/fig_deformation_vertical.jpg)",
        "![lateral vs longitudinal](figures/fig_lateral_vs_longitudinal.jpg)",
        "",
        "Raw GT-vs-GT disagreement stays ≤ 3 px median out to ~120 m horizontally; the C1",
        "diagnostic's winning shift is **0.0 samples at every displacement bin** and the C3 scale",
        "gain is ≤ +0.022 — the scene's skyline is far-field at these displacements. Vertical",
        "motion is a near-pure offset (+13 px at 120 m up) that mean removal absorbs to ≤ 1 px —",
        "mean-removed NCC is altitude-invariant here, and the discarded offset is real altitude",
        "information (a possible future altimetry signal, not a matching defect).",
        "",
        "**Matcher decision (pre-declared R9 rule): no ladder rung is justified — C0 stands.**",
        "No variant was run on real data; the measured deformation contains no recoverable shift",
        "or scale. The two genuine GT misses (below) are local scene changes, not alignment",
        "failures a rigid transform could absorb.",
        "",
        "### Deformation pair examples (rule: median + max displacement per direction)",
        "",
    ]
    lines += [f"![pair]({f})" for f in pair_files]
    lines += [
        "",
        "## 5. Experiment D — frozen C0 NCC retrieval (pre-registered)",
        "",
        "Primary statistic: **rank-1 correctness of the shortlist** (the frozen acceptance rule",
        "rejects even NCC ≈ 1.0 identity matches on a 6-reference database — the known EXP-SKY-006",
        "caveat; evaluator acceptance-gated numbers are reported beside it).",
        "",
        "| source | rank-1 correct | evaluator success (accepted) | recall@5-of-6 | confident-false |",
        "|---|---|---|---|---|",
    ]
    for key, rows in all_rows.items():
        scored = [r for r in rows if r["scored"]]
        top1 = sum(1 for r in scored if r["top1_correct"])
        cf = sum(1 for r in scored if r["confident_false"])
        ts = evals[key]["retrieval"]["topological_success"]
        lines.append(f"| {key} | {top1}/{len(scored)} ({top1 / len(scored):.1%}) "
                     f"| {ts['success_rate']:.1%} | {ts['recall_at_k_rate']:.1%} | {cf} |")
    lines += ["", "![rank1 vs translation](figures/fig_rank1_vs_translation.jpg)", ""]
    for slice_key, label in (("stage", "stage"), ("time_of_day", "time of day"),
                             ("clouds", "cloud state"), ("translation_bin", "translation bin")):
        for key, rows in all_rows.items():
            lines += [f"### {key} by {label}", ""]
            lines += _md_rate(_rate_table(rows, slice_key), label)
            lines.append("")
    lines += [
        "### Retrieval examples",
        "",
        "Rule: every GT and SegFormer rank-1 miss; DP's first miss per time of day; the first",
        "rank-1-correct query per source.",
        "",
    ]
    lines += [f"![retrieval]({f})" for f in retr_files]
    lines += [
        "",
        "## 6. Interpretation (pre-registered four-case matrix)",
        "",
        "- **Displacement axis** (DAY+CLEAR translated queries): GT 38/40 and SegFormer ~equal —",
        "  **Case 3 (viable)** on the shortlist statistic; the two GT misses at ~60 m east of a1",
        "  are a genuine local skyline transformation (**Case 4 locally**), worth targeted",
        "  captures.",
        "- **Condition axis** (same-pose cross-condition queries): GT and SegFormer near-perfect →",
        "  **Case 3**; DP → **Case 1 (extraction problem)** — its curve is condition-unstable, it",
        "  refuses most DAWN/DUSK frames, and it is the only source with confident-false answers",
        "  (5 of its 6 accepted matches are wrong).",
        "- The frozen acceptance rule, not ranking, is the operational bottleneck on a tiny",
        "  database — same conclusion as EXP-SKY-006, now shown against ground truth.",
        "",
        "## 7. Limitations",
        "",
        f"{LIMITATIONS}",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[simret] report -> {out_dir} ({len(pair_files)} pair panels, "
          f"{len(retr_files)} retrieval panels)")
    return {"out_dir": str(out_dir), "figures": sorted(p.name for p in figs.glob('fig_*.jpg')),
            "pair_panels": pair_files, "retrieval_panels": retr_files}
