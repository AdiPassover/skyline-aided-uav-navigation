"""The final SKY visual report (Experiment element list of spec FR-014; research R12).

Reads only committed experiment outputs (`evaluations/sim-ext-*`) plus the observation store for
example imagery, and writes `evaluations/sim-final-report/report.md` + JPEG figures. Every example
panel names its **declared selection rule** (median / best / threshold-adjacent / worst /
mechanism-specific) — nothing is cherry-picked, and refusals/failures appear beside successes.

Split from ``extreport.py`` deliberately: that module measures, this one depicts; the figures can
be rebuilt without re-measuring anything.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from hsreloc.simret import ext, extindex, render
from hsreloc.simret.report import ReportError

REPORT_VERSION = "1.0.0"

LEVEL_ORDER = ("village", "mountains", "city")
SOURCE_LABEL = {"sim_exact": "GT (oracle:sim_exact)", "segformer": "SegFormer-B0 (silver)",
                "dp": "poc_robust_dp (frozen classical)"}


def _resolve(base: Path, value) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fmt(v, digits=3):
    if v is None or v == "":
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _pct(v):
    return "—" if v is None else f"{100.0 * float(v):.1f}%"


# --------------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------------

def fig_level_map(out_path: Path, level: str, idx_rows: list) -> Path:
    plt = render._plt()
    anchors = [r for r in idx_rows if r["level"] == level and r["kind"] == "anchor_capture"
               and f"{r['time_of_day']}+{r['clouds']}" == "DAY+CLEAR"]
    swipes: dict = {}
    for r in idx_rows:
        if r["level"] == level and r["kind"] == "swipe":
            swipes.setdefault(r["session_id"], []).append(r)
    fig, ax = plt.subplots(figsize=(8, 8))
    for sid, rows in sorted(swipes.items()):
        rows = sorted(rows, key=lambda r: r["observation_id"])
        hard = rows[0]["hard_tag"]
        ax.plot([r["east_m"] for r in rows], [r["north_m"] for r in rows],
                lw=0.9, alpha=0.8, color="#cc4444" if hard else "#7799cc",
                label=None, zorder=1)
    ax.scatter([r["east_m"] for r in anchors], [r["north_m"] for r in anchors],
               s=45, color="#222222", marker="^", zorder=3, label="anchor (DAY+CLEAR reference)")
    for r in anchors:
        ax.annotate(r["anchor_id"], (r["east_m"], r["north_m"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.plot([], [], color="#7799cc", lw=1.2, label="star swipe")
    ax.plot([], [], color="#cc4444", lw=1.2, label="hard-tagged swipe")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title(f"{level}: {len(anchors)} anchors, {len(swipes)} swipes")
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    return render._save(fig, out_path)


def fig_condition_heatmap(out_path: Path, title: str, cell_values: dict, fmt="{:.1f}") -> Path:
    """3x3 TIME x CLOUD heatmap from ``{'DAY+CLEAR': value, ...}`` (None -> hatched)."""
    plt = render._plt()
    times = ("DAY", "DAWN", "DUSK")
    clouds = ("CLEAR", "CLOUDY", "VERY_CLOUDY")
    grid = np.full((3, 3), np.nan)
    for i, t in enumerate(times):
        for j, c in enumerate(clouds):
            v = cell_values.get(f"{t}+{c}")
            if v is not None:
                grid[i, j] = float(v)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    im = ax.imshow(grid, cmap="viridis")
    for i in range(3):
        for j in range(3):
            v = grid[i, j]
            ax.text(j, i, "—" if np.isnan(v) else fmt.format(v), ha="center", va="center",
                    color="white", fontsize=10,
                    path_effects=None)
    ax.set_xticks(range(3), clouds, fontsize=8)
    ax.set_yticks(range(3), times, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return render._save(fig, out_path)


def fig_deformation_response(out_path: Path, level_key: str, rows: list, axis: str,
                             y_field: str = "curve_median_abs_offset_removed_px") -> Path:
    """Per-swipe response curves vs SIGNED axis displacement; the pooled median on top."""
    plt = render._plt()
    members = [r for r in rows if r["leg_axis"] == axis and r["clouds"] == "CLEAR"
               and r.get("signed_axis_displacement_m") not in (None, "")]
    if not members:
        raise ReportError(f"{level_key}/{axis}: no response rows")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    by_swipe: dict = {}
    for r in members:
        by_swipe.setdefault(r["session_id"], []).append(r)
    for sid, srows in sorted(by_swipe.items()):
        srows = sorted(srows, key=lambda r: float(r["signed_axis_displacement_m"]))
        ax.plot([float(r["signed_axis_displacement_m"]) for r in srows],
                [float(r[y_field]) for r in srows], lw=0.8, alpha=0.55, color="#88aacc")
    xs = sorted({round(float(r["signed_axis_displacement_m"]) / 10.0) * 10.0 for r in members})
    med = []
    for x in xs:
        vals = [float(r[y_field]) for r in members
                if abs(float(r["signed_axis_displacement_m"]) - x) <= 5.0]
        med.append(np.median(vals) if vals else np.nan)
    ax.plot(xs, med, lw=2.2, color="#223355", label="pooled median (10 m bins)")
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlabel(f"signed {axis} displacement from swipe center (m)")
    ax.set_ylabel(y_field.replace("curve_", "").replace("_", " "))
    ax.set_title(f"{level_key}: GT skyline response — {axis} (per swipe + pooled median; "
                 f"DAY+CLEAR swipes only)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    return render._save(fig, out_path)


def fig_recall_vs_distance(out_path: Path, per_level: dict, threshold_lines=(0.8, 0.9)) -> Path:
    """R@1 vs nearest-reference-distance bin, one line per level, chance dashed per level."""
    plt = render._plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"village": "#3366aa", "mountains": "#117744", "city": "#aa5511"}
    for key, payload in per_level.items():
        slices = payload["by_nearest_ref_bin"]
        keys = sorted(slices, key=render._bin_sort_key)
        xs = [k for k in keys if slices[k]["recall_at_1"] is not None]
        ax.plot(range(len(xs)), [slices[k]["recall_at_1"] for k in xs], marker="o",
                color=colors.get(key, "#555555"),
                label=f"{key} (chance {payload['chance']:.3f})")
        for i, k in enumerate(xs):
            ax.annotate(f"n={slices[k]['n_attainable']}", (i, slices[k]["recall_at_1"]),
                        fontsize=6, xytext=(0, 5), textcoords="offset points", ha="center")
        ax.axhline(payload["chance"], color=colors.get(key, "#555555"), lw=0.7, ls=":")
        ax.set_xticks(range(len(xs)), xs, fontsize=8)
    for t in threshold_lines:
        ax.axhline(t, color="#bb3333", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("nearest-reference distance bin (m)")
    ax.set_ylabel("Recall@1 (attainable queries)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    return render._save(fig, out_path)


def fig_margin_distributions(out_path: Path, per_level_margins: dict) -> Path:
    """Correct-minus-best-incorrect margins as database size grows (one box per level/source)."""
    plt = render._plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, series = [], []
    for label, values in per_level_margins.items():
        if values:
            labels.append(label)
            series.append(values)
    if not series:
        raise ReportError("no margins to plot")
    ax.boxplot(series, tick_labels=labels, showfliers=False)
    ax.axhline(0.0, color="#bb3333", lw=0.9, ls="--")
    ax.set_ylabel("correct − best incorrect (NCC)")
    ax.set_title("score separation vs database size (attainable exact-pose queries)", fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax.grid(alpha=0.25)
    return render._save(fig, out_path)


def fig_extraction_example(out_path: Path, store: Path, session_id: str, observation_id: str,
                           curves: dict, title: str, rule: str) -> Path:
    plt = render._plt()
    img = render._image(store / session_id / "images" / f"{observation_id}.png")
    fig, ax = plt.subplots(figsize=(7, 7))
    if img is not None:
        ax.imshow(img)
        ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
    for provenance, rows in curves.items():
        style = render.SOURCE_STYLE.get(provenance, {"color": "#888888", "lw": 1.2,
                                                     "label": provenance, "ls": ":"})
        ax.plot(np.arange(len(rows)), rows, **style)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nselection rule: {rule}", fontsize=9)
    return render._save(fig, out_path)


# --------------------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------------------

def build(cfg: dict, base: Path) -> dict:
    out_dir = _resolve(base, cfg["report_out"])
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    store = _resolve(base, cfg["store_root"])
    idx = extindex.load_index(_resolve(base, cfg["index_out"]))
    idx_rows = idx["rows"]
    exp = {k: _resolve(base, v) for k, v in cfg["experiments"].items() if str(v).startswith("../")}

    md = ["# Final Simulator Skyline Relocalization Validation — visual report",
          "",
          f"Feature `20260903-225416-sky-final-sim-validation` · EXP-SKY-008 · evidence tier T2 "
          f"(`ue5_simulator`) · report version {REPORT_VERSION}",
          "",
          "Every example panel names its declared selection rule; refusals and failures are "
          "reported beside successes; unattainable queries are counted, never dropped.",
          ""]
    written = []

    # -- 1+2: dataset overview + anchor maps ------------------------------------------------------
    md += ["## 1–2. Dataset overview and anchor maps", ""]
    per_level = idx["manifest"]["per_level"]
    md += ["| level | split | rows | anchors | swipes |", "|---|---|---|---|---|"]
    for lvl, block in sorted(cfg["levels"].items()):
        n_swipes = len(block["swipe_sessions"])
        md.append(f"| {lvl} | {block['split']} | {per_level[lvl]['rows']} | "
                  f"{per_level[lvl]['anchors']} | {n_swipes} |")
    md += ["", "Every anchor carries the complete 9-condition TIME×CLOUD matrix at cm-identical "
               "poses (index validation). Raw digest: see the freeze prereg.", ""]
    for lvl, block in sorted(cfg["levels"].items()):
        p = fig_level_map(fig_dir / f"map_{block['task_key']}.jpg", lvl, idx_rows)
        written.append(p)
        md.append(f"![map {block['task_key']}](figures/{p.name})")
    md.append("")
    return _build_rest(cfg, base, md, written, out_dir, fig_dir, store, idx_rows, exp)


def _build_rest(cfg, base, md, written, out_dir, fig_dir, store, idx_rows, exp):
    # -- 3: extraction examples -------------------------------------------------------------------
    md += ["## 3. GT / SegFormer / DP extraction examples", "",
           "Selection rules: per level — the frame whose SegFormer error is the level median; the "
           "worst SegFormer frame; one DP refusal (first by observation id, if any).", ""]
    ctx = ext.context(cfg, base)
    srcs = ext.build_sources(cfg, base, ctx)
    smap = ctx["smap"]
    for lvl, block in sorted(cfg["levels"].items()):
        key = block["task_key"]
        acc_path = exp["exp1_out"] / _pop_of(cfg, lvl) / key / "accuracy_segformer.csv"
        if not acc_path.exists():
            md.append(f"*{key}: extraction rows not present ({acc_path.name} missing).*")
            continue
        rows = sorted(_load_csv(acc_path), key=lambda r: float(r["median_abs_px"]))
        for label, pick in (("median", rows[len(rows) // 2]), ("worst", rows[-1])):
            oid = pick["observation_id"]
            curves = {}
            for skey, s in srcs.items():
                try:
                    curves[s.provenance] = s.get(oid).row_per_col
                except Exception:
                    pass
            p = fig_extraction_example(
                fig_dir / f"extraction_{key}_{label}.jpg", store, smap[oid], oid, curves,
                f"{key} {label}: {oid} (SegFormer med |err| {float(pick['median_abs_px']):.2f} px)",
                f"{label} SegFormer median-abs-error frame of the level")
            written.append(p)
            md.append(f"![extraction {key} {label}](figures/{p.name})")
    md.append("")

    # -- 4: extraction accuracy tables ------------------------------------------------------------
    md += ["## 4. Extraction accuracy by level and condition (median abs px vs exact GT)", ""]
    for skey in ("segformer", "dp"):
        md += [f"### {SOURCE_LABEL[skey]}", "",
               "| level | overall med | refusal rate | worst TIME×CLOUD cell (med) | "
               "catastrophic frac (worst cell) |", "|---|---|---|---|---|"]
        for lvl, block in sorted(cfg["levels"].items()):
            key = block["task_key"]
            t = exp["exp1_out"] / _pop_of(cfg, lvl) / key / f"accuracy_{skey}_summary.json"
            if not t.exists():
                md.append(f"| {key} | — | — | — | — |")
                continue
            acc = _load_json(t)
            by_cond = acc.get("by_condition", {})
            worst = max(((c, v) for c, v in by_cond.items() if "median_abs_px" in v),
                        key=lambda cv: cv[1]["median_abs_px"]["median"], default=(None, None))
            meds = [float(r["median_abs_px"]) for r in acc["rows"]] if acc.get("rows") else []
            md.append(f"| {key} | {_fmt(float(np.median(meds)) if meds else None, 2)} | "
                      f"{_pct(acc.get('refusal_rate'))} | "
                      f"{worst[0]}: {_fmt(worst[1] and worst[1]['median_abs_px']['median'], 2)} | "
                      f"{_fmt(worst[1] and worst[1].get('catastrophic_frac', {}).get('median'), 3)} |")
        md.append("")
        for lvl, block in sorted(cfg["levels"].items()):
            key = block["task_key"]
            t = exp["exp1_out"] / _pop_of(cfg, lvl) / key / f"accuracy_{skey}_summary.json"
            if not t.exists():
                continue
            acc = _load_json(t)
            cells = {c: v["median_abs_px"]["median"] for c, v in acc.get("by_condition", {}).items()
                     if "median_abs_px" in v}
            p = fig_condition_heatmap(fig_dir / f"exp1_{key}_{skey}.jpg",
                                      f"{key} / {skey}: median abs px by TIME×CLOUD", cells)
            written.append(p)
            md.append(f"![exp1 {key} {skey}](figures/{p.name})")
        md.append("")

    return _build_retrieval(cfg, base, md, written, out_dir, fig_dir, store, idx_rows, exp)


def _pop_of(cfg: dict, level: str) -> str:
    return cfg["levels"][level]["split"]


def _build_retrieval(cfg, base, md, written, out_dir, fig_dir, store, idx_rows, exp):
    # -- 5+6: exact-pose retrieval matrix + score separation --------------------------------------
    md += ["## 5–6. Exact-pose 9-condition retrieval (databases 10 / 30 / 27) and score "
           "separation", ""]
    margins = {}
    for skey in ("sim_exact", "segformer", "dp"):
        md += [f"### {SOURCE_LABEL[skey]}", "",
               "| level | refs | chance | R@1 | R@3 | R@5 | accepted (frozen rule) | "
               "confident-false (strict) |", "|---|---|---|---|---|---|---|---|"]
        for lvl, block in sorted(cfg["levels"].items()):
            key = block["task_key"]
            sp = exp["exp3_out"] / _pop_of(cfg, lvl) / f"{key}_{skey}_summary.json"
            if not sp.exists():
                md.append(f"| {key} | — | — | — | — | — | — | — |")
                continue
            s = _load_json(sp)
            md.append(f"| {key} | {s['n_references']} | {_fmt(s['chance_recall_at_1'], 3)} | "
                      f"{_pct(s['recall_at_1'])} | {_pct(s['recall_at_3'])} | "
                      f"{_pct(s['recall_at_5'])} | {s['n_accepted_of_attainable']}/"
                      f"{s['n_attainable']} | {s['n_confident_false_strict']} |")
            rows_p = exp["exp3_out"] / _pop_of(cfg, lvl) / f"{key}_{skey}_rank_rows.csv"
            if rows_p.exists():
                vals = [float(r["margin_correct_minus_best_incorrect"])
                        for r in _load_csv(rows_p)
                        if r["attainable"] == "True"
                        and r["margin_correct_minus_best_incorrect"] not in ("", None)]
                margins[f"{key}\n{skey}\n(n={s['n_references']})"] = vals
        md.append("")
        for lvl, block in sorted(cfg["levels"].items()):
            key = block["task_key"]
            sp = exp["exp3_out"] / _pop_of(cfg, lvl) / f"{key}_{skey}_summary.json"
            if not sp.exists():
                continue
            s = _load_json(sp)
            cells = {c: (None if v["recall_at_1"] is None else 100.0 * v["recall_at_1"])
                     for c, v in s.get("by_condition", {}).items()}
            p = fig_condition_heatmap(fig_dir / f"exp3_{key}_{skey}.jpg",
                                      f"{key} / {skey}: exact-pose R@1 (%) by query condition",
                                      cells, fmt="{:.0f}")
            written.append(p)
            md.append(f"![exp3 {key} {skey}](figures/{p.name})")
        md.append("")
    try:
        p = fig_margin_distributions(fig_dir / "score_separation.jpg", margins)
        written.append(p)
        md += [f"![score separation](figures/{p.name})", ""]
    except ReportError:
        md += ["*score-separation figure skipped (no margins yet)*", ""]

    # -- 7–9: deformation curves ------------------------------------------------------------------
    md += ["## 7–9. GT viewpoint response (lateral / longitudinal / vertical)", "",
           "Offset-removed disagreement to the swipe's own center curve; thin lines are single "
           "swipes (DAY+CLEAR only), the heavy line the pooled 10 m-bin median. The final leg of "
           "every swipe is outbound-only **by trajectory design** (owner correction 2026-09-03).",
           ""]
    for lvl, block in sorted(cfg["levels"].items()):
        key = block["task_key"]
        d = exp["exp4_out"] / _pop_of(cfg, lvl) / key
        if not d.exists():
            md.append(f"*{key}: deformation rows not present.*")
            continue
        rows = []
        for f in sorted(d.glob("Run_*_rows.csv")):
            rows.extend(_load_csv(f))
        for r in rows:
            for k in ("signed_axis_displacement_m", "curve_median_abs_offset_removed_px",
                      "curve_median_abs_px", "curve_offset_px"):
                r[k] = None if r.get(k) in ("", None) else float(r[k])
        for axis in ("lateral", "longitudinal", "vertical"):
            try:
                p = fig_deformation_response(fig_dir / f"exp4_{key}_{axis}.jpg", key, rows, axis)
                written.append(p)
                md.append(f"![deformation {key} {axis}](figures/{p.name})")
            except ReportError as exc:
                md.append(f"*{key}/{axis}: {exc}*")
        vd = d / "deformation_summary.json"
        if vd.exists():
            v = _load_json(vd)["vertical_diagnostic"]
            md += ["", f"Vertical diagnostic ({key}): {v['note']}", ""]
    md.append("")
    return _build_tail(cfg, base, md, written, out_dir, fig_dir, store, idx_rows, exp)


def _build_tail(cfg, base, md, written, out_dir, fig_dir, store, idx_rows, exp):
    # -- 10: outbound/return repeatability --------------------------------------------------------
    md += ["## 10. Outbound vs return repeatability", "",
           "| level | source | axis | n pairs | med |Δ| px | med C0 NCC |", "|---|---|---|---|---|---|"]
    for lvl, block in sorted(cfg["levels"].items()):
        key = block["task_key"]
        sp = exp["exp4_out"] / _pop_of(cfg, lvl) / key / "deformation_summary.json"
        if not sp.exists():
            continue
        rep = _load_json(sp).get("repeatability_by_source", {})
        for skey, axes in sorted(rep.items()):
            for axis, stats in sorted(axes.items()):
                md.append(f"| {key} | {skey} | {axis} | {stats['n']} | "
                          f"{_fmt(stats.get('curve_median_abs_px', {}).get('median'), 2)} | "
                          f"{_fmt(stats.get('c0_ncc', {}).get('median'), 4)} |")
    md.append("")

    # -- 11+12+16: swipe retrieval vs distance, per level, GT vs SegFormer ------------------------
    md += ["## 11–12, 16. Swipe retrieval vs distance — per level, GT vs SegFormer", ""]
    for skey in ("sim_exact", "segformer"):
        per_level = {}
        md += [f"### {SOURCE_LABEL[skey]}", "",
               "| level | attainable | R@1 | R80 | R90 | unattainable (reported, not scored) |",
               "|---|---|---|---|---|---|"]
        for lvl, block in sorted(cfg["levels"].items()):
            key = block["task_key"]
            sp = exp["exp5_out"] / _pop_of(cfg, lvl) / f"{key}_{skey}_summary.json"
            if not sp.exists():
                md.append(f"| {key} | — | — | — | — | — |")
                continue
            s = _load_json(sp)
            prim = s["primary_excluding_hard"]
            md.append(f"| {key} | {prim['n_attainable']} | {_pct(prim['recall_at_1'])} | "
                      f"{_fmt(s['R80'].get('radius_m'))} m | {_fmt(s['R90'].get('radius_m'))} m | "
                      f"{prim['n_unattainable']} |")
            per_level[key] = {"by_nearest_ref_bin": s["by_nearest_ref_bin"],
                              "chance": prim["chance_recall_at_1"] or 0.0}
        md.append("")
        if per_level:
            p = fig_recall_vs_distance(fig_dir / f"exp5_recall_vs_distance_{skey}.jpg", per_level)
            written.append(p)
            md += [f"![recall vs distance {skey}](figures/{p.name})", ""]

    # -- 13+14+15: hard swipes, FOV/parallax mechanisms, far-field successes ----------------------
    md += ["## 13–15. Hard swipes, FOV/parallax mechanisms, far-field successes", ""]
    hard_summary = exp["exp6_out"] / "hard_summary.json"
    if hard_summary.exists():
        h = _load_json(hard_summary)
        md += [h.get("headline", ""), ""]
        for panel in h.get("panels", []):
            md.append(f"![{panel['title']}](../sim-ext-exp6-hard/{panel['file']})")
            md.append(f"*{panel['title']} — rule: {panel['rule']}*")
        md.append("")
    else:
        md += ["*Hard-swipe mechanism analysis not present yet (runs post-primary).*", ""]

    # -- 17: acceptance ---------------------------------------------------------------------------
    md += ["## 17. Acceptance: frozen rule vs accept-v2-margin", ""]
    acc_final = exp["acceptance_out"] / "final" / "side_by_side.json"
    acc_dev = exp["acceptance_out"] / "dev" / "calibration.json"
    if acc_dev.exists():
        c = _load_json(acc_dev)["chosen"]
        md += [f"DEV calibration chose θ_s = {c['theta_s']}, θ_m = {c['theta_m']} "
               f"(precision {_pct(c['accepted_precision'])}, coverage "
               f"{_pct(c['coverage_of_attainable'])}, {c['n_confident_false_strict']} strict "
               f"confident-false on {c['n_rows']} DEV rows).", ""]
    if acc_final.exists():
        payload = _load_json(acc_final)
        md += ["| cell | rule | accepted | precision | coverage | confident-false (strict) |",
               "|---|---|---|---|---|---|"]
        for cell, res in sorted(payload["results"].items()):
            for rk in ("frozen_rule", "variant"):
                r = res[rk]
                md.append(f"| {cell} | {r['rule'][:44]} | {r['n_accepted']} | "
                          f"{_pct(r['accepted_precision'])} | "
                          f"{_pct(r['coverage_of_attainable'])} | "
                          f"{r['n_confident_false_strict']} |")
        md.append("")
    else:
        md += ["*Held-out acceptance comparison not present yet.*", ""]

    # -- 18: matcher decision ---------------------------------------------------------------------
    md += ["## 18. Matcher decision", ""]
    dec = Path(base).resolve().parents[1] / "specs" / \
        "20260903-225416-sky-final-sim-validation" / "matcher-decision.md"
    if dec.exists():
        md += [dec.read_text(encoding="utf-8"), ""]
    else:
        md += ["*matcher-decision.md not written yet.*", ""]

    # -- 19: operating regime ---------------------------------------------------------------------
    md += ["## 19. Operating-regime summary", "",
           "*Filled at close-out from the pre-registered criteria (freeze prereg "
           "`interpretation` block); see EXP-SKY-008 for the verdict and its numeric citations.*",
           ""]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[simret] ext report -> {report_path} ({len(written)} figures)")
    return {"report": str(report_path), "figures": [str(p) for p in written]}
