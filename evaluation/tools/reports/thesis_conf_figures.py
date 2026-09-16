"""Thesis figures for the confidence / estimator-instability component.

Every figure is produced from **committed artifacts only** — the held-out validation summary
`evaluations/exp-conf-005/analysis.json` and the corrected-convention render cards under
`evaluations/exp-conf-002/hypothesis_audit/corrected/cards/`. No estimator is run, no dataset
imagery is read (the audit cards are themselves committed renders), so this regenerates in a
fresh worktree in seconds.

    python evaluation/tools/reports/thesis_conf_figures.py \
        --out figures/conf

Outputs:
  conf_decile_enrichment.png  — held-out signal-decile enrichment against the independent
                                image-evidence reference (left) and against the fused-attitude
                                per-increment rotation error (right), both flights.
  conf_audit_examples.png     — two human-reviewed pairs rendered under the corrected
                                (full-resolution-conjugated) transform convention: a healthy
                                pair and the estimator-instability pair preceding a restart.
                                Source cases h1.1 (hkairport01-b frame 4946) and h2.1
                                (hkairport01-b frame 921).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
from PIL import Image                                                    # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ANALYSIS = REPO / "evaluations" / "exp-conf-005" / "analysis.json"
CARDS = REPO / "evaluations" / "exp-conf-002" / "hypothesis_audit" / "corrected" / "cards"

"""Bottom-row panels of an overview card are located by content rather than by fixed pixel
boxes: the cards are matplotlib renders whose panel extents depend on each pair's image aspect.
The bottom row holds the stitch under the transform the estimator shipped (left) and the stitch
under the independent best-fit reference transform (right)."""

SEQ_LABEL = {"hkairport03": "HKairport03", "amtown03": "AMtown03"}
SEQ_STYLE = {"hkairport03": ("#1f6fb4", "o"), "amtown03": ("#c0392b", "s")}


def decile_figure(out: Path) -> None:
    data = json.loads(ANALYSIS.read_text())["sequences"]
    x = np.arange(1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    panels = [
        (axes[0], "primary", "rot_dis_by_drot_decile",
         "disagreement with the independent\nregistration reference (deg)",
         "(a) independent image evidence"),
        (axes[1], "secondary_attitude", "err_att_by_drot_decile",
         "per-increment rotation error against\nthe fused attitude reference (deg)",
         "(b) physical rotation error"),
    ]
    for ax, block, key, ylabel, title in panels:
        for seq in ("hkairport03", "amtown03"):
            colour, marker = SEQ_STYLE[seq]
            deciles = data[seq][block][key]
            med = [d["median"] for d in deciles]
            p90 = [d["p90"] for d in deciles]
            ax.plot(x, med, color=colour, marker=marker, markersize=4, linewidth=1.6,
                    label=f"{SEQ_LABEL[seq]}, median")
            ax.plot(x, p90, color=colour, linewidth=1.0, linestyle="--", alpha=0.65,
                    label=f"{SEQ_LABEL[seq]}, 90th pct")
        ax.set_xticks(x)
        ax.set_xlabel("decile of the estimator-instability diagnostic", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8.5)
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout(pad=0.6)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _runs(flags: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Index ranges of consecutive True values at least `min_len` long."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    if start is not None and len(flags) - start >= min_len:
        out.append((start, len(flags) - 1))
    return out


def _panel_boxes(case: str) -> tuple[Image.Image, tuple, tuple]:
    """Locate the two bottom-row image panels of a card, excluding the card's own titles."""
    card = Image.open(CARDS / f"{case}-corrected_overview.png").convert("RGB")
    grey = np.asarray(card.convert("L"))
    ink = grey < 250
    # Horizontal bands of content; the last one is the bottom row of images (titles form their
    # own thin band above it and are separated by a white gutter).
    bands = _runs(ink.any(axis=1), min_len=40)
    top, bottom = bands[-1]
    band = ink[top:bottom + 1]
    cols = _runs(band.any(axis=0), min_len=40)
    left, right = cols[0], cols[-1]
    return card, (left[0], top, left[1] + 1, bottom + 1), (right[0], top, right[1] + 1, bottom + 1)


def audit_figure(out: Path) -> None:
    rows = [
        ("h1.1", "healthy pair"),
        ("h2.1", "estimator-instability pair"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.9))
    for r, (case, row_label) in enumerate(rows):
        card, box_shipped, box_reference = _panel_boxes(case)
        for c, box in enumerate((box_shipped, box_reference)):
            ax = axes[r][c]
            ax.imshow(np.asarray(card.crop(box)))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#999999")
                spine.set_linewidth(0.6)
        axes[r][0].set_ylabel(row_label, fontsize=10, labelpad=5)
    axes[0][0].set_title("estimator's own transform", fontsize=10, pad=4)
    axes[0][1].set_title("independent reference transform", fontsize=10, pad=4)
    fig.tight_layout(pad=0.3, h_pad=0.7, w_pad=0.5)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory for the figures")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)

    decile_figure(out / "conf_decile_enrichment.png")
    audit_figure(out / "conf_audit_examples.png")
    print(f"wrote conf_decile_enrichment.png, conf_audit_examples.png -> {out}")


if __name__ == "__main__":
    main()
