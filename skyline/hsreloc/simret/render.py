"""Visual diagnostics for the simulator experiments (Part I).

Two kinds of picture, and one rule that governs both.

**The pair panel** — everything about one query/reference comparison on a page: both RGB frames, all
three sources' curves over each, the pose displacement split into lateral and longitudinal, the
frozen C0 score beside any matcher variant's score with its recovered shift/scale/warp, and whether
the pair is geographically correct. It exists because a table of scores never explains *why* a match
failed, and the ``EXP-SKY-006`` render set was what made its Outcome C legible.

**The trend plots** — extraction error by condition, GT deformation against translation distance,
Recall@1 against translation, lateral against longitudinal, the three sources side by side, and C0
against a variant.

The rule: **nothing here invents data.** Every function takes real records and raises ``RenderError``
if they are absent or empty. There is no demo mode, no placeholder series and no synthetic fallback,
because a plausible-looking plot with nothing behind it is indistinguishable from a result and would
be the worst artifact this repository could emit.

Matplotlib only (a declared project dependency); no seaborn, no pandas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RENDER_VERSION = "1.0.0"

#: Colour per curve source. Fixed, so the same source is the same colour in every figure.
SOURCE_STYLE = {
    "oracle:sim_exact": {"color": "#111111", "lw": 2.0, "label": "simulator GT", "ls": "-"},
    "automatic:segformer_b0_ade20k": {"color": "#1f77b4", "lw": 1.6, "label": "SegFormer (silver)",
                                      "ls": "-"},
    "automatic:poc_robust_dp": {"color": "#d62728", "lw": 1.6, "label": "DP (frozen)", "ls": "--"},
}


class RenderError(Exception):
    """A figure was requested without the data it depicts."""


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:                                  # pragma: no cover
        raise RenderError("matplotlib is required for the simulator renders") from exc
    return plt


def _require(values, what: str):
    if values is None or len(values) == 0:
        raise RenderError(
            f"no data for {what}. This figure is drawn from real records only — there is no demo "
            f"mode, and an empty plot would be indistinguishable from a result.")
    return values


def _image(path):
    p = Path(path)
    if not p.exists():
        return None
    import cv2
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if img is None else img[:, :, ::-1]


# --------------------------------------------------------------------------------------------------
# the pair panel
# --------------------------------------------------------------------------------------------------

def render_pair(out_path: Path, query: dict, reference: dict, offset: dict,
                scores: dict, correct: bool, title: str = "") -> Path:
    """One query/reference pair, in full.

    ``query`` / ``reference``: ``{"observation_id", "image_path", "curves": {provenance: rows}}``.
    ``offset``: the ``ViewpointOffset`` fields. ``scores``: ``{variant: MatchResult-like dict}`` —
    C0 first by convention, each with score/shift/scale/warp.
    """
    plt = _plt()
    _require(query.get("curves"), f"pair panel {query.get('observation_id')}")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8),
                             gridspec_kw={"height_ratios": [3, 2]})
    for ax, side in zip(axes[0], (query, reference)):
        img = _image(side.get("image_path"))
        if img is not None:
            ax.imshow(img)
            ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
        for provenance, rows in side["curves"].items():
            style = SOURCE_STYLE.get(provenance, {"color": "#888888", "lw": 1.2,
                                                  "label": provenance, "ls": ":"})
            ax.plot(np.arange(len(rows)), rows, **style)
        ax.set_title(f"{'query' if side is query else 'reference'}: {side['observation_id']}",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].legend(fontsize=7, loc="lower left", framealpha=0.8)

    ax = axes[1][0]
    for provenance in sorted(set(query["curves"]) | set(reference.get("curves", {}))):
        style = SOURCE_STYLE.get(provenance, {"color": "#888888", "lw": 1.2, "label": provenance,
                                              "ls": ":"})
        if provenance in query["curves"] and provenance in reference.get("curves", {}):
            d = np.asarray(query["curves"][provenance]) - np.asarray(reference["curves"][provenance])
            ax.plot(d, color=style["color"], lw=style["lw"], ls=style["ls"],
                    label=f"{style['label']}  (med |Δ| {np.median(np.abs(d)):.1f} px)")
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.set_xlabel("image column"); ax.set_ylabel("query − reference row (px)")
    ax.set_title("per-column disagreement", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1][1]
    ax.axis("off")
    lines = [
        f"displacement       {offset.get('translation_m', float('nan')):.2f} m",
        f"  longitudinal     {offset.get('along_m', float('nan')):+.2f} m  (along reference heading)",
        f"  lateral          {offset.get('lateral_m', float('nan')):+.2f} m  (to reference right)",
        f"  altitude         {_opt(offset.get('up_m'))}",
        f"Δyaw               {offset.get('yaw_diff_deg', float('nan')):+.2f}°",
        f"Δpitch / Δroll     {_opt(offset.get('pitch_diff_deg'), '°')} / {_opt(offset.get('roll_diff_deg'), '°')}",
        "",
        f"geographically correct: {'YES' if correct else 'NO'}",
        "",
    ]
    for variant, res in scores.items():
        lines.append(f"{variant:<22} score {res.get('score', float('nan')):+.4f}"
                     f"  shift {res.get('shift', 0.0):+.2f}  scale {res.get('scale', 1.0):.4f}"
                     f"  warp {res.get('warp_magnitude', 0.0):.2f}"
                     f"{'' if res.get('accepted', True) else '  [REFUSED]'}")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)

    fig.suptitle(title or f"{query['observation_id']} vs {reference['observation_id']}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _opt(value, unit: str = " m") -> str:
    return "n/a" if value is None else f"{value:+.2f}{unit}"


# --------------------------------------------------------------------------------------------------
# trend plots
# --------------------------------------------------------------------------------------------------

def _binned(rows, x_field, y_field):
    groups: dict = {}
    for r in rows:
        x, y = r.get(x_field), r.get(y_field)
        if x in (None, "") or y is None or not np.isfinite(float(y)):
            continue
        groups.setdefault(str(x), []).append(float(y))
    if not groups:
        raise RenderError(f"no finite ({x_field}, {y_field}) pairs to plot")
    keys = sorted(groups, key=_bin_sort_key)
    return keys, [groups[k] for k in keys]


def _bin_sort_key(label: str):
    try:
        return (0, float(str(label).split("-")[0]))
    except ValueError:
        return (1, str(label))


def plot_deformation_vs_translation(out_path: Path, rows: list,
                                    y_field: str = "curve_median_abs_px") -> Path:
    """GT skyline deformation against translation — the measurement that decides which matcher rung
    is even appropriate (a uniform offset means C1; offset + spread means C3; neither means neither)."""
    plt = _plt()
    _require(rows, "deformation vs translation")
    keys, groups = _binned(rows, "translation_bin", y_field)
    _, offsets = _binned(rows, "translation_bin", "curve_offset_px")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, tick_labels=keys, showfliers=False)
    ax.plot(range(1, len(keys) + 1), [np.median(np.abs(o)) for o in offsets], "o--",
            color="#d62728", label="median |constant offset| (px)")
    ax.set_xlabel("translation bin (m)")
    ax.set_ylabel(f"{y_field} (px)")
    ax.set_title("Ground-truth skyline deformation vs translation")
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def plot_metric_vs_axis(out_path: Path, rows: list, x_field: str, y_field: str,
                        title: str, ylabel: str, series_field: str | None = None) -> Path:
    """Any per-bin metric against any condition axis, optionally split into series (e.g. by source)."""
    plt = _plt()
    _require(rows, f"{y_field} vs {x_field}")
    fig, ax = plt.subplots(figsize=(8, 5))
    series = {None: rows} if series_field is None else _split(rows, series_field)
    for name, members in sorted(series.items(), key=lambda kv: str(kv[0])):
        keys, groups = _binned(members, x_field, y_field)
        ax.plot(keys, [float(np.median(g)) for g in groups], "o-",
                label=None if name is None else str(name))
    ax.set_xlabel(x_field); ax.set_ylabel(ylabel); ax.set_title(title)
    if series_field is not None:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_lateral_vs_longitudinal(out_path: Path, rows: list, value_field: str,
                                 label: str = "") -> Path:
    """The E3 split as a map: lateral against longitudinal displacement, coloured by a metric."""
    plt = _plt()
    _require(rows, "lateral vs longitudinal")
    lat = [float(r["lateral_m"]) for r in rows if r.get(value_field) is not None]
    lon = [float(r["along_m"]) for r in rows if r.get(value_field) is not None]
    val = [float(r[value_field]) for r in rows if r.get(value_field) is not None]
    _require(val, f"lateral vs longitudinal ({value_field})")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(lat, lon, c=val, cmap="viridis", s=28, edgecolor="none")
    fig.colorbar(sc, ax=ax, label=label or value_field)
    ax.axhline(0, color="#999999", lw=0.8); ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlabel("lateral displacement (m, + = reference's right)")
    ax.set_ylabel("longitudinal displacement (m, + = along reference heading)")
    ax.set_title("Lateral vs longitudinal displacement")
    ax.set_aspect("equal", adjustable="datalim")
    return _save(fig, out_path)


def plot_variant_comparison(out_path: Path, rows: list, variants: list,
                            x_field: str = "translation_bin") -> Path:
    """C0 against every later rung on identical pairs. C0 must be present or the figure is refused."""
    plt = _plt()
    _require(rows, "variant comparison")
    if not any(v.startswith("c0") for v in variants):
        raise RenderError("the frozen C0 baseline must appear in a variant comparison — without it "
                          "the figure shows a number, not a difference (the brief's Part H)")
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in variants:
        keys, groups = _binned(rows, x_field, f"{v}_score")
        ax.plot(keys, [float(np.median(g)) for g in groups], "o-", label=v)
    ax.set_xlabel(x_field); ax.set_ylabel("median score of the correct reference")
    ax.set_title("Matcher variants on identical pairs")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, out_path)


def _split(rows: list, field: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r.get(field), []).append(r)
    return out


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_path
