"""Figures (contracts/evaluation-outputs.md, FR-052).

Every figure is stamped with the evaluation_id and config_digest that produced it, so a
plot dropped into the thesis can always be traced back to its run (FR-051).

Phase 3 added trajectory_xy. Phase 6 adds event_timeline (error trace with detected
events marked); Phase 7 adds scale_series (per-segment fitted scale vs. distance, with
recenters marked -- the direct visual answer to `COMP-001` §4's recentering question).
Phase 10 (T084) adds error_vs_time, error_vs_distance, yaw_error and rpe_by_length,
completing FR-052's figure set with the same stamping convention and the same rule:
a metric that is unsupported or omitted still produces a file, an explicitly labelled
empty (or partially labelled) panel, never a missing one -- SC-014 applies to figures
exactly as it does to `metrics.json` entries.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this module must not require a display (CI, tests)

import matplotlib.pyplot as plt
import numpy as np


def _stamp(fig, evaluation_id: str, config_digest: str) -> None:
    fig.text(
        0.99, 0.01,
        f"{evaluation_id} | {config_digest[:12]}",
        ha="right", va="bottom", fontsize=6, color="gray",
    )


def plot_trajectory_xy(
    aligned_est_points: np.ndarray,
    gt_points: np.ndarray,
    evaluation_id: str,
    config_digest: str,
    out_path: Path | str,
    title: str = "Aligned trajectory vs ground truth",
) -> Path:
    """Plan-view comparison of the aligned estimated trajectory against ground truth."""
    aligned_est_points = np.asarray(aligned_est_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gt_points[:, 0], gt_points[:, 1], "-", color="black", linewidth=2, label="Ground truth")
    ax.plot(
        aligned_est_points[:, 0], aligned_est_points[:, 1],
        "--", color="tab:blue", linewidth=1.5, label="Estimate (aligned)",
    )
    ax.scatter(gt_points[0, 0], gt_points[0, 1], color="green", marker="o", s=60, zorder=5, label="Start")
    ax.scatter(gt_points[-1, 0], gt_points[-1, 1], color="red", marker="s", s=60, zorder=5, label="End")

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    _stamp(fig, evaluation_id, config_digest)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


_EVENT_COLORS = {
    "restart": "tab:red",
    "recenter": "tab:orange",
    "pose_discontinuity": "tab:purple",
    "large_error_excursion": "tab:brown",
}


def plot_event_timeline(
    timestamps_s: np.ndarray,
    per_frame_error_m: np.ndarray,
    events,
    evaluation_id: str,
    config_digest: str,
    out_path: Path | str,
    title: str = "Error trace with detected events",
) -> Path:
    """Per-frame position error over time, with every detected event (both
    from_run_record and derived -- events.py) marked as a vertical line, colour-coded
    by type. A visual complement to events.csv's error_before_m/error_after_m: this is
    where a reader judges the TEMPORAL association for themselves (module docstring in
    events.py: association, not proven causation).

    Renders a labelled empty panel, never a missing file, when no error trace is
    available (e.g. no ground truth at all -- FR-045 says events/runtime still work
    without it, so the plot must not simply fail to exist).
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps_s.size == 0:
        ax.text(0.5, 0.5, "No synchronised ground truth -- no error trace to plot",
                 ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        per_frame_error_m = np.asarray(per_frame_error_m, dtype=np.float64)
        ax.plot(timestamps_s, per_frame_error_m, "-", color="black", linewidth=1.2, label="Position error")
        seen_types = set()
        for e in events:
            color = _EVENT_COLORS.get(e.type, "gray")
            label = e.type if e.type not in seen_types else None
            seen_types.add(e.type)
            ax.axvline(e.timestamp_s, color=color, linestyle="--", linewidth=1.0, alpha=0.8, label=label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position error (m)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    ax.set_title(title)
    _stamp(fig, evaluation_id, config_digest)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _empty_panel(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
             fontsize=11, color="gray", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_error_vs_time(
    timestamps_s: np.ndarray,
    per_frame_error_m: np.ndarray,
    evaluation_id: str,
    config_digest: str,
    out_path: Path | str,
    title: str = "Position error vs time",
    omission_reason: str = None,
) -> Path:
    """ATE per-frame error against elapsed time -- the plain time-domain view that
    `event_timeline` overlays events onto. Kept as its own figure because a reader
    comparing error growth against distance (below) rather than time, or wanting a
    version uncluttered by event markers, needs it separately (T084)."""
    fig, ax = plt.subplots(figsize=(9, 4))

    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps_s.size == 0:
        _empty_panel(ax, omission_reason or "No synchronised ground truth -- no error trace to plot")
    else:
        per_frame_error_m = np.asarray(per_frame_error_m, dtype=np.float64)
        ax.plot(timestamps_s, per_frame_error_m, "-", color="tab:blue", linewidth=1.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position error (m)")
        ax.grid(True, alpha=0.3)

    ax.set_title(title)
    _stamp(fig, evaluation_id, config_digest)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_error_vs_distance(
    distance_m: np.ndarray,
    per_frame_error_m: np.ndarray,
    evaluation_id: str,
    config_digest: str,
    out_path: Path | str,
    title: str = "Position error vs distance travelled",
    omission_reason: str = None,
) -> Path:
    """ATE per-frame error against cumulative ground-truth distance travelled
    (`naveval.metrics._cumulative_distance`), the distance-domain complement to
    `plot_error_vs_time`. `drift_per_distance` is this curve's slope, in effect."""
    fig, ax = plt.subplots(figsize=(9, 4))

    distance_m = np.asarray(distance_m, dtype=np.float64)
    if distance_m.size == 0:
        _empty_panel(ax, omission_reason or "No synchronised ground truth -- no error trace to plot")
    else:
        per_frame_error_m = np.asarray(per_frame_error_m, dtype=np.float64)
        ax.plot(distance_m, per_frame_error_m, "-", color="tab:blue", linewidth=1.2)
        ax.set_xlabel("Distance travelled (m)")
        ax.set_ylabel("Position error (m)")
        ax.grid(True, alpha=0.3)

    ax.set_title(title)
    _stamp(fig, evaluation_id, config_digest)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_yaw_error(
    timestamps_s: np.ndarray,
    yaw_error_per_frame_deg: np.ndarray,
    evaluation_id: str,
    config_digest: str,
    out_path: Path | str,
    title: str = "Yaw error vs time (shortest-angle)",
    omission_reason: str = None,
) -> Path:
    """Per-frame shortest-angle yaw error (`naveval.metrics.yaw_error_metric`'s
    `per_frame` array) against time. Independent of every position figure above --
    FR-031's "yaw MUST be reported independently of position error" made visual.

    `omission_reason` covers the FR-061 case this figure exists to never hide: a
    dataset with no heading ground truth at all. The panel then says exactly why,
    rather than a plot silently not being generated -- SC-014 for figures.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    timestamps_s = np.asarray(timestamps_s, dtype=np.float64) if timestamps_s is not None else np.zeros(0)
    if timestamps_s.size == 0:
        _empty_panel(ax, omission_reason or "No heading ground truth -- yaw error unsupported")
    else:
        yaw_error_per_frame_deg = np.asarray(yaw_error_per_frame_deg, dtype=np.float64)
        ax.plot(timestamps_s, yaw_error_per_frame_deg, "-", color="tab:green", linewidth=1.2)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Yaw error (deg, shortest-angle)")
        ax.grid(True, alpha=0.3)

    ax.set_title(title)
    _stamp(fig, evaluation_id, config_digest)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_rpe_by_length(
    rpe_by_length_m: dict,
    evaluation_id: str,
    config_digest: str,
    out_path: Path | str,
    title: str = "RPE by sub-trajectory length (global scale, FR-067)",
    omitted_reasons: dict = None,
) -> Path:
    """RPE at each configured sub-trajectory length, under the single global fitted
    scale (never the per-segment-rescaled diagnostic -- plotting that beside the
    primary figure would blur exactly the FR-067 distinction the diagnostic exists to
    keep visible, so it is deliberately left out of this figure).

    `rpe_by_length_m` maps `length_m -> {"rmse": ...}` or `None` where
    `naveval.metrics.relative_pose_error` returned `None` (trajectory shorter than that
    length). A length with no bar is not a missing bar -- it is annotated in place with
    why, and the whole panel falls back to the labelled-empty case only when nothing at
    all could be computed (SC-014 for figures, not just `metrics.json`).
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    omitted_reasons = omitted_reasons or {}
    lengths = sorted(rpe_by_length_m.keys())
    available = [(l, rpe_by_length_m[l]) for l in lengths if rpe_by_length_m[l] is not None]

    if not lengths:
        _empty_panel(ax, "No RPE lengths configured")
    elif not available:
        _empty_panel(ax, "No RPE length is computable for this trajectory")
    else:
        avail_lengths = [l for l, _ in available]
        avail_values = [v["rmse"] for _, v in available]
        x = np.arange(len(lengths))
        bar_map = {l: i for i, l in enumerate(lengths)}
        ax.bar([bar_map[l] for l in avail_lengths], avail_values, color="tab:blue", width=0.6)

        for l in lengths:
            if rpe_by_length_m[l] is None:
                reason = omitted_reasons.get(l, "omitted")
                ax.text(bar_map[l], 0, reason, rotation=90, ha="center", va="bottom",
                         fontsize=7, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{l:g} m" for l in lengths])
        ax.set_xlabel("Sub-trajectory length")
        ax.set_ylabel("RPE RMSE (m)")
        ax.grid(True, axis="y", alpha=0.3)

    ax.set_title(title)
    _stamp(fig, evaluation_id, config_digest)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scale_series(
    scale_points,
    evaluation_id: str,
    config_digest: str,
    out_path: Path | str,
    title: str = "Per-segment fitted scale (diagnostic)",
) -> Path:
    """`scale_ratio_to_global` against distance travelled, per segment -- the direct
    visual form of SC-012 and the `COMP-001` §4 recentering question. Segments whose
    range spans a recenter are marked distinctly, so a reader can see at a glance
    whether scale steps coincide with recentering.

    ALWAYS a diagnostic (FR-067, DEC-003) -- the title says so, and this function
    never touches or overrides a primary metric. Segments the local fit refused (no
    scale) are still plotted as marked, unfilled points at ratio=NaN (matplotlib skips
    NaN silently) rather than being dropped from the x-axis entirely.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    if not scale_points:
        ax.text(0.5, 0.5, "No segments to plot", ha="center", va="center",
                 transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        distances = [p.distance_start_m for p in scale_points]
        ratios = [p.scale_ratio_to_global if p.scale_ratio_to_global is not None else float("nan") for p in scale_points]
        spans = [p.spans_recenter for p in scale_points]

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, label="Global scale (ratio = 1.0)")
        ax.plot(distances, ratios, "-o", color="tab:blue", markersize=5, linewidth=1.2, label="Segment scale ratio")

        recenter_d = [d for d, s in zip(distances, spans) if s]
        recenter_r = [r for r, s in zip(ratios, spans) if s]
        if recenter_d:
            ax.scatter(recenter_d, recenter_r, color="tab:red", marker="^", s=80, zorder=5, label="Spans a recenter")

        ax.set_xlabel("Distance travelled (m)")
        ax.set_ylabel("scale_ratio_to_global")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    ax.set_title(title)
    _stamp(fig, evaluation_id, config_digest)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
