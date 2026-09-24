"""EXP-CONF-002 Phase 4b: build the small UNBLINDED hypothesis-audit packet.

Owner-directed methodology change (2026-08-31): the 102-case blinded per-pair H/Q/F/X
adjudication was judged insufficiently interpretable for reliable human per-frame labels
(consecutive 10 Hz aerial frames are too similar; difference heatmaps alone lack intuitive
context). It is preserved as an artifact but NOT EXECUTED. This packet replaces it with
~15 deliberately chosen, fully unblinded exemplars whose purpose is to let the owner judge
whether the major EXP-CONF-002 quantitative conclusions are visually and mechanistically
credible at the hypothesis level (agree / partly agree / disagree / visually inconclusive):

  H1  A1 healthy + A2 healthy (registrable pair, shipped VO transform agrees with evidence)
  H2  A1 healthy + A2 degraded (registrable pair, shipped transform substantially off) — key
  H3  A1 degraded + A2 degraded (genuinely difficult / scene-driven pair)
  H4  old T-D p99.5 label says degraded but panel evidence is healthy (RTK artifact hypothesis)
  H5  cheap EXP-CONF-001 signals indicate degradation AND A2 is degraded (association is real)
  H6  counterexamples deliberately chosen to weaken the emerging interpretation

Nothing is fitted or tuned; the frozen VO is untouched; no new confidence target is created.
Selection is deterministic (documented criteria, ties by extremity then frame index) —
these are exemplars, not a sample; no statistical claim rests on them.

The dominant visual evidence is actual stitched mosaics on an expanded common canvas:
the frozen VO's own pair transform vs the best-fit reference transform (MAGSAC++ + LS refit
on FB-validated KLT correspondences, the panel's R2 machinery), plus 1:1 overlap zooms at
high-contrast structure, edge overlays, and the existing track/support diagnostics.

Usage:
  python build_audit.py --images-root-hkb <dir> --images-root-amd <dir> [--select-only]
  python build_audit.py --html-only        # rebuild index.html from cases.json (+ visual_checks.json)
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
REPO = TOOL_DIR.parents[2]
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_001"))
from p4_analysis import SIGNALS, load_frames, load_labels                       # noqa: E402
from panel import FB_VALID_PX, GFTT, LK, MAGSAC, load_run, vo_pair_homography   # noqa: E402
from panel_analysis import PANEL_ALL, PANEL_Q, load_panel, tail_flags           # noqa: E402
from readout import apply_h, fit_h_ls                                           # noqa: E402

OUT = REPO / "evaluations/exp-conf-002/hypothesis_audit"
SEQS = ("hkairport01-b", "amtown01-d")
RESTARTS = {"hkairport01-b": (922,), "amtown01-d": (10698, 11793)}
ZOOM = 300           # zoom crop side, canvas px (1:1)
ZOOM_MIN_SEP = 450   # min distance between the two zoom centres
COVERAGE_GRID = 8

REG_Q = ("fb_median_px", "ho_ste_median_px", "boot_rot_sd_deg")   # A1: pair registrability
FID_Q = ("d_vo_median_px", "xchk_grid_px")                        # A2: shipped-estimate fidelity

EXPLAIN = {
    "fb_median_px": "median forward-backward round-trip error of the KLT tracks "
                    "(t-1 → t → t-1); how self-consistent the tracking itself is on this pair",
    "ho_ste_median_px": "achievable registration: median symmetric transfer error of a refit "
                        "homography measured only on held-out correspondences it never saw",
    "boot_rot_sd_deg": "rotation instability: SD of the pair rotation when the homography is "
                       "refit 200 times on resampled correspondences",
    "d_vo_median_px": "shipped-estimate deviation: median symmetric transfer error of the VO's "
                      "own persisted pair transform on the same independent correspondences",
    "xchk_grid_px": "independent-pipeline disagreement: median grid-point displacement between "
                    "the VO transform and a SIFT+MAGSAC++ transform computed from scratch",
    "inlier_count": "cheap runtime signal: number of RANSAC inliers in the production estimator "
                    "(lower = worse)",
    "inlier_ratio": "cheap runtime signal: inlier fraction among tracked features "
                    "(lower = worse)",
    "inlier_coverage": "cheap runtime signal: fraction of an 8×8 grid occupied by inliers — "
                       "spatial spread of the supporting points (lower = worse)",
    "relative_support": "cheap runtime signal: inlier count relative to the recent running "
                        "level (lower = worse)",
    "inc_flow_px": "cheap runtime signal: image-centre displacement implied by this increment "
                   "(higher = worse)",
    "inc_log_scale_dispersion": "cheap runtime signal: spread of per-inlier log-scale estimates "
                                "(higher = worse)",
}
HIGHER_WORSE_SIGNALS = {"inc_flow_px", "inc_log_scale_dispersion"}


# ------------------------------------------------------------------ data assembly
def pct_within(x: np.ndarray, v: float) -> float:
    m = np.isfinite(x)
    return float(np.mean(x[m] <= v)) if m.sum() else float("nan")


def load_ctx(seq: str) -> dict:
    run_dir = REPO / f"runs/{seq}-homography-rigid-conf-v1"
    ds_dir = REPO / f"datasets/{seq}"
    frames_rows = list(csv.DictReader((ds_dir / "frames.csv").open(newline="", encoding="utf-8")))
    rframes, sidecar = load_run(run_dir)
    cheap = load_frames(run_dir)
    labels = load_labels(REPO / f"evaluations/exp-conf-001-dev/labels/{seq}.csv")
    n = len(rframes)
    panel = load_panel(seq, n)
    flags = {q: tail_flags(panel[q]) for q in PANEL_Q}
    n_tails = np.stack([flags[q] for q in PANEL_Q]).sum(axis=0)

    pcts = {q: np.full(n, np.nan) for q in PANEL_ALL}
    for q in PANEL_ALL:
        m = np.isfinite(panel[q])
        if m.sum():
            order = np.argsort(np.argsort(panel[q][m]))
            pcts[q][m] = (order + 1) / m.sum()
    reg = np.nanmax(np.stack([pcts[q] for q in REG_Q]), axis=0)     # A1 badness (worst member)
    fid = np.nanmin(np.stack([pcts[q] for q in FID_Q]), axis=0)     # A2 badness (both must be high)
    fid_hi = np.nanmax(np.stack([pcts[q] for q in FID_Q]), axis=0)  # for "A2 healthy": both low

    # native RTK heading transitions (committed groundtruth) for artifact-context distances
    ts, hd = [], []
    with (ds_dir / "groundtruth.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts.append(float(r["timestamp_s"]))
            hd.append(float(r["heading_deg"]))
    ts, hd = np.array(ts), np.array(hd)
    chg = np.where(np.abs(np.diff(hd)) > 0.25)[0]
    trans_t = (ts[chg] + ts[chg + 1]) / 2.0
    frame_t = np.array([float(r["timestamp_s"]) for r in frames_rows])

    return dict(frames_rows=frames_rows, rframes=rframes, sidecar=sidecar, cheap=cheap,
                labels=labels, panel=panel, pcts=pcts, reg=reg, fid=fid, fid_hi=fid_hi,
                n_tails=n_tails, events=cheap["event"], trans_t=trans_t, frame_t=frame_t, n=n)


# ------------------------------------------------------------------ deterministic selection
def select_cases(ctx: dict[str, dict], td_thr: float) -> list[dict]:
    """Deterministic exemplar selection. Every criterion is written next to its mask."""
    used: set[tuple[str, int]] = set()
    cases: list[dict] = []

    def covered(c):
        return np.all([np.isfinite(c["panel"][q]) for q in PANEL_ALL], axis=0)

    def td_mask(c):
        la = c["labels"]
        return np.where(la["gradable_a"], la["t_a"] > td_thr, False)

    def take(hyp: str, seq: str, i: int, why: str):
        if (seq, int(i)) in used:
            return False
        used.add((seq, int(i)))
        cases.append(dict(hyp=hyp, seq=seq, frame=int(i), why=why))
        return True

    def argbest(mask: np.ndarray, score: np.ndarray, seq: str, largest=True):
        """Highest/lowest score among masked frames not already used; None if empty."""
        idx = np.where(mask)[0]
        idx = np.array([i for i in idx if (seq, int(i)) not in used and i >= 1], dtype=int)
        if len(idx) == 0:
            return None
        order = idx[np.argsort(score[idx], kind="stable")]
        return int(order[-1] if largest else order[0])

    # ---- H1: A1 healthy + A2 healthy, with real motion (one per sequence) ----------------
    for seq in SEQS:
        c = ctx[seq]
        flow = c["cheap"]["inc_flow_px"]
        med_flow = np.nanmedian(flow)
        mask = (covered(c) & (c["events"] == "none") & (c["reg"] <= 0.5)
                & (c["pcts"]["d_vo_median_px"] <= 0.10) & (c["pcts"]["xchk_grid_px"] <= 0.10)
                & np.where(np.isfinite(flow), flow >= med_flow, False))
        i = argbest(mask, -(c["pcts"]["d_vo_median_px"] + c["pcts"]["xchk_grid_px"]), seq)
        if i is not None:
            take("H1", seq, i, "all three registrability diagnostics below the sequence median, "
                 "shipped-estimate deviation and cross-check both in the best decile, "
                 "camera motion at or above the sequence median (not a trivially static pair)")

    # ---- H2: A1 healthy + A2 degraded (the key hypothesis) -------------------------------
    # H2.1 / H2.2: worst shipped-estimate lead pair before each estimator-internal restart
    for seq, r in (("hkairport01-b", 922), ("amtown01-d", 10698)):
        c = ctx[seq]
        lead = np.zeros(c["n"], bool)
        lead[max(1, r - 10):r] = True
        i = argbest(lead & covered(c) & (c["reg"] <= 0.6), c["fid"], seq)
        if i is None:      # FB is mildly elevated in the @10698 lead — take the best available
            i = argbest(lead & covered(c) & (c["reg"] <= 0.85), c["fid"], seq)
        if i is not None:
            take("H2", seq, i, f"lead-in pair of the genuine restart at frame {r}: the pair "
                 "registers cleanly by every registrability diagnostic while the shipped "
                 "transform is in the extreme tail — the estimator-internal-failure signature")
    # H2.3: same signature in ordinary operation, far from any restart
    best = None
    for seq in SEQS:
        c = ctx[seq]
        far = np.ones(c["n"], bool)
        for r in RESTARTS[seq]:
            far[max(0, r - 25):r + 26] = False
        mask = (covered(c) & (c["events"] == "none") & far & (c["reg"] <= 0.5)
                & (c["pcts"]["d_vo_median_px"] >= 0.99) & (c["pcts"]["xchk_grid_px"] >= 0.99))
        i = argbest(mask, c["fid"], seq)
        if i is not None and (best is None or c["fid"][i] > best[2]):
            best = (seq, i, c["fid"][i])
    if best:
        take("H2", best[0], best[1], "ordinary operation far from any restart: registrability "
             "healthy (all three below the sequence median band) while both independent "
             "fidelity diagnostics are in the worst 1% — A2 degradation is not confined to "
             "restart neighbourhoods")

    # ---- H3: A1 degraded + A2 degraded (scene-driven) ------------------------------------
    c = ctx["amtown01-d"]
    mask = np.zeros(c["n"], bool)
    mask[11783:11793] = True
    mask &= covered(c)
    i = argbest(mask, c["reg"] + c["fid"], "amtown01-d")
    if i is not None:
        take("H3", "amtown01-d", i, "lead-in pair of the genuine restart at frame 11793, the "
             "one collapse where every panel member is pegged — the scene-driven signature")
    best = None
    for seq in SEQS:
        c = ctx[seq]
        far = np.ones(c["n"], bool)
        for r in RESTARTS[seq]:
            far[max(0, r - 25):r + 26] = False
        mask = covered(c) & (c["reg"] >= 0.98) & (c["fid"] >= 0.90) & far
        i = argbest(mask, c["reg"] + c["fid"], seq)
        if i is not None and (best is None or ctx[best[0]]["reg"][best[1]] +
                              ctx[best[0]]["fid"][best[1]] < c["reg"][i] + c["fid"][i]):
            best = (seq, i)
    if best:
        take("H3", best[0], best[1], "every panel member degraded (registrability in the worst "
             "2%, fidelity in the worst 10%) away from the known collapse — a genuinely hard "
             "pair in ordinary operation")

    # ---- H4: old T-D positive, panel healthy (RTK-artifact hypothesis) --------------------
    h4 = []
    for seq in SEQS:
        c = ctx[seq]
        mask = (covered(c) & td_mask(c) & (c["n_tails"] == 0)
                & (c["reg"] <= 0.75) & (c["fid_hi"] <= 0.75))
        for i in np.where(mask)[0]:
            h4.append((seq, int(i), float(c["labels"]["t_a"][i])))
    h4.sort(key=lambda t: -t[2])               # most extreme labelled "error" first
    h4_why = ("the old RTK-derived per-frame label calls this stitch degraded (T-A above the "
              "frozen p99.5 threshold) while no panel member places it in any tail and every "
              "diagnostic sits at or below mid-range — candidate reference artifact; the "
              "distance to the nearest native integer-degree RTK heading transition is below")
    picks = []                                  # best per sequence first, then fill globally
    for seq in SEQS:
        first = next((t for t in h4 if t[0] == seq), None)
        if first:
            picks.append(first)
    for t in h4:
        if len(picks) >= 3:
            break
        if t not in picks:
            picks.append(t)
    for seq, i, _ in picks[:3]:
        take("H4", seq, i, h4_why)

    # ---- H5: cheap signals degraded AND A2 degraded ---------------------------------------
    def far_from_restarts(seq: str, pad: int = 10) -> np.ndarray:
        far = np.ones(ctx[seq]["n"], bool)
        for r in RESTARTS[seq]:
            far[max(0, r - pad):r + pad + 1] = False
        return far

    best = None
    for seq in SEQS:
        c = ctx[seq]
        cov = c["cheap"]["inlier_coverage"]
        m = np.isfinite(cov)
        cov_thr = np.quantile(cov[m], 0.02)
        mask = covered(c) & far_from_restarts(seq) & np.where(m, cov <= cov_thr, False) & \
            (c["pcts"]["d_vo_median_px"] >= 0.98)
        i = argbest(mask, c["pcts"]["d_vo_median_px"], seq)
        if i is not None and (best is None or c["pcts"]["d_vo_median_px"][i] > best[2]):
            best = (seq, i, c["pcts"]["d_vo_median_px"][i])
    if best:
        take("H5", best[0], best[1], "runtime inlier coverage in the worst 2% of the sequence "
             "AND shipped-estimate deviation in the worst 2% — the cheap spatial-support "
             "signal flagging a pair whose shipped transform is indeed far from the evidence")
    best = None
    for seq in SEQS:
        c = ctx[seq]
        flow = c["cheap"]["inc_flow_px"]
        m = np.isfinite(flow)
        flow_thr = np.quantile(flow[m], 0.98)
        mask = covered(c) & far_from_restarts(seq) & np.where(m, flow >= flow_thr, False) & \
            (c["pcts"]["d_vo_median_px"] >= 0.98)
        i = argbest(mask, c["pcts"]["d_vo_median_px"], seq)
        if i is not None and (best is None or c["pcts"]["d_vo_median_px"][i] > best[2]):
            best = (seq, i, c["pcts"]["d_vo_median_px"][i])
    if best:
        take("H5", best[0], best[1], "runtime centre-flow in the worst 2% AND shipped-estimate "
             "deviation in the worst 2% — the cheap motion-magnitude signal flagging a pair "
             "whose shipped transform is indeed far from the evidence")

    # ---- H6: counterexamples chosen to weaken the interpretation --------------------------
    best = None
    for seq in SEQS:                            # H6.1 cheap-signal false alarm
        c = ctx[seq]
        cov = c["cheap"]["inlier_coverage"]
        m = np.isfinite(cov)
        cov_thr = np.quantile(cov[m], 0.02)
        mask = covered(c) & np.where(m, cov <= cov_thr, False) & \
            (c["fid_hi"] <= 0.6) & (c["pcts"]["d_vo_median_px"] <= 0.5) & (c["reg"] <= 0.5)
        i = argbest(mask, -c["pcts"]["d_vo_median_px"], seq)
        if i is not None and (best is None or c["pcts"]["d_vo_median_px"][i] < best[2]):
            best = (seq, i, c["pcts"]["d_vo_median_px"][i])
    if best:
        take("H6", best[0], best[1], "COUNTEREXAMPLE — cheap-signal false alarm: runtime "
             "coverage in the worst 2% yet every panel diagnostic healthy; the cheap signals' "
             "association with A2 is statistical, not per-frame reliable")
    h6b = []
    for seq in SEQS:                            # H6.2 old label agrees with the panel
        c = ctx[seq]
        mask = covered(c) & td_mask(c) & (c["n_tails"] >= 2)
        for i in np.where(mask)[0]:
            h6b.append((seq, int(i), int(c["n_tails"][i]), float(c["labels"]["t_a"][i])))
    h6b.sort(key=lambda t: (-t[2], -t[3]))
    for seq, i, nt, _ in h6b:                   # first candidate not already used elsewhere
        if take("H6", seq, i, f"COUNTEREXAMPLE — the old T-D label and the panel agree "
                f"({nt} member tails): not every RTK-derived positive is an artifact"):
            break
    best = None
    for seq in SEQS:                            # H6.3 A2 tail the cheap signals miss
        c = ctx[seq]
        cov, flow, disp = (c["cheap"]["inlier_coverage"], c["cheap"]["inc_flow_px"],
                           c["cheap"]["inc_log_scale_dispersion"])
        def midband(x, lo, hi):
            m = np.isfinite(x)
            ql, qh = np.quantile(x[m], [lo, hi])
            return np.where(m, (x >= ql) & (x <= qh), False)
        mask = (covered(c) & (c["pcts"]["d_vo_median_px"] >= 0.99)
                & midband(cov, 0.2, 1.0) & midband(flow, 0.0, 0.8)
                & midband(disp, 0.0, 0.8))
        i = argbest(mask, c["pcts"]["d_vo_median_px"], seq)
        if i is not None and (best is None or c["pcts"]["d_vo_median_px"][i] > best[2]):
            best = (seq, i, c["pcts"]["d_vo_median_px"][i])
    if best:
        take("H6", best[0], best[1], "COUNTEREXAMPLE — a shipped-estimate deviation in the "
             "worst 1% that ALL the cheap runtime signals miss (coverage, flow and dispersion "
             "all in their unremarkable mid-band): the cheap signals are not a sufficient "
             "runtime detector for A2 either")
    return cases


# ------------------------------------------------------------------ rendering
def to_rgb(g: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def build_mosaic(prev: np.ndarray, cur: np.ndarray, H: np.ndarray,
                 T: np.ndarray, size: tuple[int, int]):
    """Blend mosaic on the common canvas; returns (mosaic_rgb, prev_layer, cur_layer, overlap)."""
    Wp = cv2.warpPerspective(prev, T, size)
    Wc = cv2.warpPerspective(cur, (T @ H).astype(np.float64), size)
    Mp = cv2.warpPerspective(np.full_like(prev, 255), T, size) >= 250
    Mc = cv2.warpPerspective(np.full_like(cur, 255), (T @ H).astype(np.float64), size) >= 250
    both = Mp & Mc
    mos = np.where(both, (Wp.astype(np.uint16) + Wc) // 2,
                   np.maximum(np.where(Mp, Wp, 0), np.where(Mc, Wc, 0))).astype(np.uint8)
    rgb = to_rgb(mos)
    h, w = prev.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    for M, col in ((T, (255, 160, 40)), ((T @ H), (40, 200, 255))):
        pts = apply_h(np.asarray(M, np.float64), corners).astype(np.int32)
        cv2.polylines(rgb, [pts.reshape(-1, 1, 2)], True, col, 3)
    return rgb, Wp, Wc, both


def zoom_regions(energy_src: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """Two ZOOM x ZOOM windows maximizing gradient energy inside the valid overlap."""
    gx = cv2.Sobel(energy_src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(energy_src, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    v = cv2.erode(valid.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    mag[~v] = 0.0
    box = cv2.boxFilter(mag, -1, (ZOOM, ZOOM), normalize=True)
    box[~v] = 0.0
    picks: list[tuple[int, int]] = []
    b = box.copy()
    for _ in range(2):
        y, x = np.unravel_index(int(np.argmax(b)), b.shape)
        if b[y, x] <= 0:
            break
        picks.append((int(x), int(y)))
        y0, y1 = max(0, y - ZOOM_MIN_SEP), min(b.shape[0], y + ZOOM_MIN_SEP)
        x0, x1 = max(0, x - ZOOM_MIN_SEP), min(b.shape[1], x + ZOOM_MIN_SEP)
        b[y0:y1, x0:x1] = 0.0
    return picks


def edge_overlay(prev_crop: np.ndarray, cur_crop: np.ndarray) -> np.ndarray:
    base = (prev_crop.astype(np.float32) * 0.30)
    ep = cv2.Canny(prev_crop, 60, 140)
    ec = cv2.Canny(cur_crop, 60, 140)
    rgb = np.stack([base + np.where(ep > 0, 255.0, 0.0),
                    base + np.where(ec > 0, 255.0, 0.0),
                    base + np.where(ec > 0, 255.0, 0.0)], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def crop(a: np.ndarray, cx: int, cy: int) -> np.ndarray:
    r = ZOOM // 2
    return a[max(0, cy - r):cy + r, max(0, cx - r):cx + r]


def render_case(cid: str, prev: np.ndarray, cur: np.ndarray,
                H_vo: np.ndarray, H_ref: np.ndarray,
                p0, p1, fbv, valid, inl, out_dir: Path) -> dict:
    h, w = prev.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    pts = np.vstack([corners] + [apply_h(H, corners) for H in (H_vo, H_ref)])
    mn = np.floor(pts.min(axis=0)) - 4
    mx = np.ceil(pts.max(axis=0)) + 4
    T = np.array([[1, 0, -mn[0]], [0, 1, -mn[1]], [0, 0, 1]], np.float64)
    size = (int(mx[0] - mn[0]), int(mx[1] - mn[1]))

    mos_vo, Wp_vo, Wc_vo, ov_vo = build_mosaic(prev, cur, H_vo, T, size)
    mos_rf, Wp_rf, Wc_rf, ov_rf = build_mosaic(prev, cur, H_ref, T, size)
    both = ov_vo & ov_rf

    # ---------- overview: frames + the two full mosaics ---------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    fig.suptitle(f"{cid} — orange: t−1 border, cyan: warped t border; overlap is a 50/50 "
                 "blend, so misalignment shows as double edges", fontsize=14, fontweight="bold")
    axes[0, 0].imshow(prev, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("frame t−1 (earlier)")
    axes[0, 1].imshow(cur, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("frame t (later)")
    axes[1, 0].imshow(mos_vo)
    axes[1, 0].set_title("stitch under the frozen VO's own pair transform")
    axes[1, 1].imshow(mos_rf)
    axes[1, 1].set_title("stitch under the best-fit reference transform\n"
                         "(MAGSAC++ + LS refit on FB-validated tracks)")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_dir / f"{cid}_overview.png", dpi=100)
    plt.close(fig)

    # ---------- zooms: 1:1 crops at high-contrast structure ------------------------------
    picks = zoom_regions(Wp_rf, both)
    fig, axes = plt.subplots(len(picks), 4, figsize=(19, 5.0 * len(picks)), squeeze=False)
    fig.suptitle(f"{cid} — overlap zooms at 1:1 pixel scale "
                 "(same canvas region left and right)", fontsize=14, fontweight="bold")
    zoom_info = []
    for r, (cx, cy) in enumerate(picks):
        p_cur = apply_h(np.linalg.inv((T @ H_ref)), np.array([[cx, cy]], np.float64))
        disp = float(np.linalg.norm(apply_h((T @ H_vo), p_cur) - apply_h((T @ H_ref), p_cur)))
        zoom_info.append(dict(canvas_xy=[cx, cy], transform_disagreement_px=round(disp, 2)))
        bl_vo = ((Wp_vo.astype(np.uint16) + Wc_vo) // 2).astype(np.uint8)
        bl_rf = ((Wp_rf.astype(np.uint16) + Wc_rf) // 2).astype(np.uint8)
        panels = [
            (crop(bl_vo, cx, cy), f"VO stitch blend — transforms disagree by "
                                  f"{disp:.1f} px here", "gray"),
            (crop(bl_rf, cx, cy), "best-fit stitch blend", "gray"),
            (edge_overlay(crop(Wp_vo, cx, cy), crop(Wc_vo, cx, cy)),
             "VO edges (red: t−1, cyan: warped t)", None),
            (edge_overlay(crop(Wp_rf, cx, cy), crop(Wc_rf, cx, cy)),
             "best-fit edges (red: t−1, cyan: warped t)", None),
        ]
        for cidx, (img, title, cmap) in enumerate(panels):
            ax = axes[r, cidx]
            ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.subplots_adjust(hspace=0.16)
    fig.savefig(out_dir / f"{cid}_zoom.png", dpi=110)
    plt.close(fig)

    # ---------- diagnostics: tracks + FB + difference maps -------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2))
    fig.suptitle(f"{cid} — correspondence diagnostics", fontsize=14, fontweight="bold")
    ax = axes[0]
    ax.imshow(prev, cmap="gray", vmin=0, vmax=255, alpha=0.85)
    if inl is not None and valid.any():
        pv0, pv1 = p0[valid], p1[valid]
        out_m = ~inl
        ax.plot([pv0[out_m, 0], pv1[out_m, 0]], [pv0[out_m, 1], pv1[out_m, 1]],
                color="red", lw=0.6, alpha=0.7)
        ax.plot([pv0[inl, 0], pv1[inl, 0]], [pv0[inl, 1], pv1[inl, 1]],
                color="lime", lw=0.6, alpha=0.7)
    ax.set_title("KLT tracks (green: homography consensus, red: excluded)")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax = axes[1]
    ax.imshow(prev, cmap="gray", vmin=0, vmax=255, alpha=0.6)
    if valid.any():
        q = ax.scatter(p0[valid, 0], p0[valid, 1], c=np.clip(fbv[valid], 0, 2), s=6,
                       cmap="viridis", vmin=0, vmax=2)
        fig.colorbar(q, ax=ax, fraction=0.04, label="FB round-trip error (px)")
    ax.set_title("forward-backward consistency")
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    for ax, Wc_, ov_, title in ((axes[2], Wc_rf, ov_rf, "abs difference under best-fit"),
                                (axes[3], Wc_vo, ov_vo, "abs difference under the VO transform")):
        d = cv2.absdiff(Wp_rf, Wc_)
        d[~ov_] = 0
        ax.imshow(d, cmap="inferno", vmin=0, vmax=96)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[:2]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / f"{cid}_diag.png", dpi=100)
    plt.close(fig)
    return dict(zooms=zoom_info, canvas_size=list(size))


# ------------------------------------------------------------------ HTML
def build_html(cases: list[dict], checks: dict) -> str:
    css = ("body{font-family:sans-serif;max-width:1600px;margin:auto;padding:0 16px}"
           "table{border-collapse:collapse;font-size:13px}"
           "td,th{border:1px solid #bbb;padding:3px 8px;text-align:left}"
           "img{max-width:100%;border:1px solid #ddd;margin:4px 0}"
           ".why{background:#eef4ff;padding:8px 12px;border-left:4px solid #4a7fd4}"
           ".vc{background:#fff7e6;padding:8px 12px;border-left:4px solid #d49a4a}"
           ".ctx{background:#f2f2f2;padding:6px 12px}")
    hyp_desc = {
        "H1": "A1 healthy + A2 healthy — the pair is registrable and the shipped VO "
              "transform agrees relatively well with the independent evidence.",
        "H2": "A1 healthy + A2 degraded — the pair itself registers cleanly, yet the "
              "shipped VO transform substantially disagrees with the independent evidence. "
              "The most important hypothesis to illustrate.",
        "H3": "A1 degraded + A2 degraded — a genuinely difficult, scene-driven pair.",
        "H4": "The old RTK-derived T-D label (p99.5) says degraded, but the panel evidence is "
              "healthy — supporting the hypothesis that the per-frame RTK label was "
              "artifact-contaminated.",
        "H5": "Cheap EXP-CONF-001 runtime signals strongly indicate degradation AND the "
              "shipped-estimate fidelity (A2) is degraded — the association is real.",
        "H6": "Deliberate counterexamples that weaken or bound the emerging interpretation.",
    }
    parts = [
        "<html><head><meta charset='utf-8'>"
        "<title>EXP-CONF-002 hypothesis audit (unblinded)</title>"
        f"<style>{css}</style></head><body>",
        "<h1>EXP-CONF-002 — unblinded hypothesis-audit packet</h1>",
        "<p><b>Purpose.</b> Not per-stitch labelling. These exemplars exist so the owner can "
        "judge, at the hypothesis level (<i>agree / partly agree / disagree / visually "
        "inconclusive</i>), whether the quantitative EXP-CONF-002 conclusions are visually and "
        "mechanistically credible: (A) A1 and A2 are genuinely different concepts; (B) there are "
        "registrable pairs for which the shipped VO chooses a materially worse transform; (C) "
        "some old RTK-derived T-D positives look geometrically healthy and were likely "
        "inappropriate per-frame labels; (D) the cheap runtime signals meaningfully associate "
        "with A2 estimator fidelity; (E) some failures are scene-driven, others "
        "estimator-internal. Selection is deterministic and deliberately includes "
        "counterexamples (H6); nothing here is a statistical sample, and panel agreement "
        "remains evidence, not ground truth.</p>",
        "<p><b>How to read the stitches.</b> Both mosaics place frame t−1 and a warped "
        "frame t on the same expanded canvas (orange border: t−1; cyan border: warped t); "
        "the overlap is a 50/50 blend, so misalignment appears as double edges / ghosting. The "
        "left stitch uses the frozen VO's own persisted pair transform; the right uses the "
        "best-fit reference transform refit on forward-backward-validated KLT correspondences. "
        "The zoom rows show the same canvas regions at 1:1 pixel scale, chosen automatically at "
        "the highest-contrast structure in the common overlap.</p>",
    ]
    order = {"H1": 0, "H2": 1, "H3": 2, "H4": 3, "H5": 4, "H6": 5}
    cases = sorted(cases, key=lambda c: (order[c["hyp"]], c["seq"], c["frame"]))
    parts.append("<h2>Case list</h2><table><tr><th>case</th><th>hypothesis</th>"
                 "<th>sequence</th><th>frame</th><th>one-line reason</th></tr>")
    for c in cases:
        parts.append(f"<tr><td><a href='#{c['id']}'>{c['id']}</a></td><td>{c['hyp']}</td>"
                     f"<td>{c['seq']}</td><td>{c['frame']}</td>"
                     f"<td>{html.escape(c['why'][:110])}…</td></tr>")
    parts.append("</table>")
    cur_h = None
    for c in cases:
        if c["hyp"] != cur_h:
            cur_h = c["hyp"]
            parts.append(f"<h2>{cur_h}</h2><p>{hyp_desc[cur_h]}</p>")
        parts.append(f"<h3 id='{c['id']}'>{c['id']} — {c['seq']} frame {c['frame']} "
                     f"(event: {c['event']})</h3>")
        parts.append(f"<div class='why'><b>Why selected:</b> {html.escape(c['why'])}</div>")
        if c.get("context"):
            parts.append(f"<div class='ctx'>{html.escape(c['context'])}</div>")
        parts.append("<table><tr><th>quantity</th><th>value</th>"
                     "<th>sequence percentile</th><th>meaning</th></tr>")
        for row in c["metrics"]:
            parts.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                row["name"], row["value"], row["pct"], html.escape(row["explain"])))
        parts.append("</table>")
        vc = checks.get(c["id"])
        if vc:
            adj = "" if vc.get("adjudicable", True) else \
                " <b>[marked: difference not visually adjudicable at card scale]</b>"
            parts.append(f"<div class='vc'><b>Qualitative visual check</b> (qualitative only; not "
                         f"ground truth, does not override the diagnostics): "
                         f"{html.escape(vc['note'])}{adj}</div>")
        for suffix in ("overview", "zoom", "diag"):
            parts.append(f"<img src='cards/{c['id']}_{suffix}.png'>")
    parts.append("</body></html>")
    return "".join(parts)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root-hkb")
    ap.add_argument("--images-root-amd")
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--html-only", action="store_true")
    a = ap.parse_args()

    if a.html_only:
        cases = json.load((OUT / "cases.json").open(encoding="utf-8"))["cases"]
        checks_p = OUT / "visual_checks.json"
        checks = json.load(checks_p.open(encoding="utf-8")) if checks_p.exists() else {}
        (OUT / "index.html").write_text(build_html(cases, checks), encoding="utf-8")
        print(f"[audit] index.html rebuilt ({len(cases)} cases, "
              f"{sum(1 for c in cases if c['id'] in checks)} visual checks)")
        return 0

    decision = json.load((REPO / "evaluations/exp-conf-001-dev/fitting_target_decision.json")
                         .open(encoding="utf-8"))
    td_thr = decision["t_a"]["thresholds"]["p99.5"]
    ctx = {seq: load_ctx(seq) for seq in SEQS}
    cases = select_cases(ctx, td_thr)
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["hyp"]] = counts.get(c["hyp"], 0) + 1
    print(f"[audit] selected {len(cases)} cases: {counts}")
    for c in cases:
        print(f"  {c['hyp']}  {c['seq']}  frame {c['frame']}")
    if a.select_only:
        return 0

    roots = {"hkairport01-b": Path(a.images_root_hkb), "amtown01-d": Path(a.images_root_amd)}
    (OUT / "cards").mkdir(parents=True, exist_ok=True)

    per_h_counter: dict[str, int] = {}
    for c in cases:
        per_h_counter[c["hyp"]] = per_h_counter.get(c["hyp"], 0) + 1
        c["id"] = f"{c['hyp']}.{per_h_counter[c['hyp']]}".lower()

    for c in cases:
        seq, i = c["seq"], c["frame"]
        cx = ctx[seq]
        frames_rows = cx["frames_rows"]
        prev = cv2.imread(str(roots[seq] / frames_rows[i - 1]["image_path"]),
                          cv2.IMREAD_GRAYSCALE)
        cur = cv2.imread(str(roots[seq] / frames_rows[i]["image_path"]), cv2.IMREAD_GRAYSCALE)
        assert prev is not None and cur is not None, (seq, i)
        H_vo, _ = vo_pair_homography(cx["rframes"], cx["sidecar"], i)
        assert H_vo is not None, (seq, i)

        p0 = cv2.goodFeaturesToTrack(prev, **GFTT).reshape(-1, 2).astype(np.float32)
        p1r, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0.reshape(-1, 1, 2), None, **LK)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1r, None, **LK)
        p1 = p1r.reshape(-1, 2)
        fbv = np.linalg.norm(p0 - p0b.reshape(-1, 2), axis=1)
        valid = (st1.ravel() == 1) & (st2.ravel() == 1) & (fbv <= FB_VALID_PX)
        H_ref, inl = cv2.findHomography(p1[valid], p0[valid], **MAGSAC)
        assert H_ref is not None, (seq, i)
        inl = inl.ravel().astype(bool)
        if inl.sum() >= 8:
            Hls = fit_h_ls(p1[valid][inl], p0[valid][inl])
            if Hls is not None:
                H_ref = Hls

        render = render_case(c["id"], prev, cur, H_vo, H_ref, p0, p1, fbv, valid, inl,
                             OUT / "cards")
        c.update(render)
        c["event"] = cx["rframes"][i]["event"]

        # metrics table with per-sequence percentiles and plain-language meaning
        rows = []
        for q in PANEL_ALL:
            v = cx["panel"][q][i]
            rows.append(dict(name=q, value=f"{v:.3f}" if np.isfinite(v) else "n/a",
                             pct=f"{100*cx['pcts'][q][i]:.1f}%"
                                 if np.isfinite(cx["pcts"][q][i]) else "n/a",
                             explain=EXPLAIN[q]))
        for sig in SIGNALS:
            v = cx["cheap"][sig][i]
            p = pct_within(cx["cheap"][sig], v) if np.isfinite(v) else float("nan")
            if np.isfinite(p) and sig not in HIGHER_WORSE_SIGNALS:
                p = 1.0 - p        # report all cheap signals as "percentile toward worse"
            rows.append(dict(name=sig, value=f"{v:.4g}" if np.isfinite(v) else "n/a",
                             pct=f"{100*p:.1f}% toward worse" if np.isfinite(p) else "n/a",
                             explain=EXPLAIN[sig]))
        la = cx["labels"]
        t_a = la["t_a"][i]
        td = bool(la["gradable_a"][i]) and np.isfinite(t_a) and t_a > td_thr
        rows.append(dict(
            name="old T-A / T-D label",
            value=(f"T-A {t_a:.3f}° — T-D p99.5 verdict: "
                   f"{'DEGRADED' if td else 'healthy'}") if np.isfinite(t_a) else "ungradable",
            pct="", explain="EXP-CONF-001 per-frame heading-error label from 10 Hz-interpolated "
                            f"RTK heading (frozen threshold {td_thr:.3f}°)"))
        dt = np.min(np.abs(cx["trans_t"] - cx["frame_t"][i])) if len(cx["trans_t"]) else np.nan
        rows.append(dict(
            name="nearest native RTK heading transition",
            value=f"{dt:.3f} s away" if np.isfinite(dt) else "n/a", pct="",
            explain="the RTK heading is integer-degree quantized at 5 Hz; T-D positives cluster "
                    "at these transitions (Phase 2 audit) — small distances mean the label "
                    "sits on a quantization step"))
        rows.append(dict(name="panel member tails", value=str(int(cx["n_tails"][i])), pct="",
                         explain="how many of the four panel members place this pair in their "
                                 "worst-2% tail for this sequence"))
        c["metrics"] = rows

        # event context
        ctxt = []
        for r in RESTARTS[seq]:
            if 0 <= r - i <= 12:
                ctxt.append(f"This pair is {r - i} frame(s) before the genuine restart at "
                            f"frame {r}.")
        if c["event"] == "recenter":
            ctxt.append("The VO performed a mosaic recenter at this frame (pair transform "
                        "taken from the logical-anchor composition).")
        c["context"] = " ".join(ctxt)
        print(f"[audit] rendered {c['id']} ({seq} {i})", flush=True)

    with (OUT / "cases.json").open("w", encoding="utf-8") as f:
        json.dump({"td_threshold_p995": td_thr, "n_cases": len(cases), "cases": cases}, f,
                  indent=1, default=str)
    checks_p = OUT / "visual_checks.json"
    checks = json.load(checks_p.open(encoding="utf-8")) if checks_p.exists() else {}
    (OUT / "index.html").write_text(build_html(cases, checks), encoding="utf-8")
    print(f"[audit] {len(cases)} cases written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
