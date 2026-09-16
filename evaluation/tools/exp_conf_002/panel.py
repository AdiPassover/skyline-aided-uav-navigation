"""EXP-CONF-002 Phase 3: the frozen four-member reference diagnostic panel, offline.

Implements exactly the panel frozen in the EXP-CONF-002 pre-registration (Phase 1):

- R1 forward-backward track consistency (Kalal et al. 2010), with Shi-Tomasi min-eigenvalue
  auxiliary covariates (no panel vote);
- R2 refit + held-out symmetric transfer error (5-fold, MAGSAC++ + LS refinement), plus d_VO:
  the frozen VO's own incremental pair homography evaluated on the same independent
  correspondences;
- R3 increment stability under bootstrap correspondence resampling (B = 200, seeded);
- R4 independent-pipeline cross-check (SIFT + ratio test + MAGSAC++ vs the VO increment).

The production estimator is untouched; R1-R3 use recomputed OpenCV correspondences (adjudicating
the pair, not BoofCV internals); d_VO/R4 evaluate the persisted VO transform. One CSV row per
consecutive pair. Terminology: this is a reference diagnostic panel / adjudication reference —
never "ground-truth confidence"; agreement between members is evidence, not ground truth.

Usage:
  python panel.py --dataset datasets/hkairport01-b --run runs/hkairport01-b-homography-rigid-conf-v1 \
      --images-root <dir containing images/> --out evaluations/exp-conf-002/panel/hkairport01-b.csv \
      [--limit N] [--start K]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))
from readout import (apply_h, circular_sd_deg, fit_h_ls, readout,          # noqa: E402
                     symmetric_transfer_px)

SEED = 20260830

# --- frozen member parameters (EXP-CONF-002 front half) --------------------------------------
GFTT = dict(maxCorners=1200, qualityLevel=0.01, minDistance=12, blockSize=7)
LK = dict(winSize=(21, 21), maxLevel=4,
          criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))
FB_VALID_PX = 1.0
K_FOLDS = 5
MAGSAC = dict(method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999)
B_BOOT = 200
SIFT_N = 2000
RATIO = 0.75
SIFT_MIN_MATCHES = 30
GRID_N = 16
GRID_MARGIN = 0.05

COLUMNS = [
    "frame_index", "event", "vo_h_source",
    # R1
    "n_corners", "fb_fail_frac", "fb_median_px", "fb_p90_px", "fb_frac_gt1", "n_fb_valid",
    "minEig_p10", "minEig_median",
    # R2
    "refit_inlier_frac", "ho_ste_median_px", "ho_ste_p90_px",
    "d_vo_median_px", "d_vo_p90_px",
    # R3
    "boot_rot_sd_deg", "boot_flow_sd_px", "boot_logscale_sd", "n_boot_inliers",
    # R4
    "sift_n_matches", "sift_inlier_frac", "xchk_rot_deg", "xchk_grid_px", "xchk_logscale",
    # readouts (reporting only)
    "ref_rot_deg", "ref_flow_px", "vo_rot_deg", "vo_flow_px", "sift_rot_deg",
    "wall_ms",
]


def load_run(run_dir: Path) -> tuple[list[dict], list[dict]]:
    with (run_dir / "frames.csv").open(newline="", encoding="utf-8") as f:
        frames = list(csv.DictReader(f))
    with (run_dir / "logical_transform.csv").open(newline="", encoding="utf-8") as f:
        sidecar = list(csv.DictReader(f))
    return frames, sidecar


def h_of(row: dict, keys: tuple[str, ...]) -> np.ndarray:
    v = [float(row[k]) for k in keys]
    return np.array(v).reshape(3, 3)


H_KEYS = tuple(f"h{i}{j}" for i in range(3) for j in range(3))
G_KEYS = tuple(f"g{i}{j}" for i in range(3) for j in range(3))


def vo_pair_homography(frames: list[dict], sidecar: list[dict], i: int) -> tuple[np.ndarray | None, str]:
    """H mapping frame i pixels -> frame i-1 pixels, from the persisted run record.

    event none:      inv(Hcw_{i-1}) @ Hcw_i        (currToWorld composition)
    event recenter:  g_{i-1} @ inv(g_i)             (logical-anchor composition)
    event init/restart: None (no VO increment exists).
    """
    ev = frames[i]["event"]
    if ev in ("init", "restart"):
        return None, "none"
    if ev == "none":
        H_prev = h_of(frames[i - 1], H_KEYS)
        H_cur = h_of(frames[i], H_KEYS)
        try:
            return np.linalg.solve(H_prev, H_cur), "world"
        except np.linalg.LinAlgError:
            return None, "none"
    if ev == "recenter":
        g_prev = h_of(sidecar[i - 1], G_KEYS)
        g_cur = h_of(sidecar[i], G_KEYS)
        try:
            return g_prev @ np.linalg.inv(g_cur), "logical"
        except np.linalg.LinAlgError:
            return None, "none"
    return None, "none"


def grid_points(width: int, height: int) -> np.ndarray:
    xs = np.linspace(width * GRID_MARGIN, width * (1 - GRID_MARGIN), GRID_N)
    ys = np.linspace(height * GRID_MARGIN, height * (1 - GRID_MARGIN), GRID_N)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def evaluate_pair(prev: np.ndarray, cur: np.ndarray, H_vo: np.ndarray | None,
                  sift, matcher, sift_cache: dict, i: int,
                  rng: np.random.Generator) -> dict:
    h, w = prev.shape[:2]
    out: dict = {}

    # ---- R1: forward-backward -----------------------------------------------------------
    p0 = cv2.goodFeaturesToTrack(prev, **GFTT)
    if p0 is None or len(p0) < 8:
        out.update(n_corners=0 if p0 is None else len(p0))
        return out
    p0 = p0.reshape(-1, 2).astype(np.float32)
    out["n_corners"] = len(p0)
    mineig = cv2.cornerMinEigenVal(prev, blockSize=GFTT["blockSize"])
    me = mineig[np.clip(p0[:, 1].astype(int), 0, h - 1), np.clip(p0[:, 0].astype(int), 0, w - 1)]
    out["minEig_p10"] = float(np.percentile(me, 10))
    out["minEig_median"] = float(np.median(me))

    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0.reshape(-1, 1, 2), None, **LK)
    p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **LK)
    p1 = p1.reshape(-1, 2)
    p0b = p0b.reshape(-1, 2)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1)
    out["fb_fail_frac"] = float(1.0 - np.mean(ok))
    e_fb = np.linalg.norm(p0 - p0b, axis=1)
    e_ok = e_fb[ok]
    if len(e_ok) == 0:
        return out
    out["fb_median_px"] = float(np.median(e_ok))
    out["fb_p90_px"] = float(np.percentile(e_ok, 90))
    out["fb_frac_gt1"] = float(np.mean(e_ok > FB_VALID_PX))
    valid = ok & (e_fb <= FB_VALID_PX)
    out["n_fb_valid"] = int(np.sum(valid))
    if out["n_fb_valid"] < 12:
        return out
    src = p1[valid]      # frame i   (cur)
    dst = p0[valid]      # frame i-1 (prev)  — pair H maps cur -> prev, matching the VO convention

    # ---- R2: reference fit + held-out STE + d_VO ---------------------------------------
    H_ref, inl = cv2.findHomography(src, dst, **MAGSAC)
    if H_ref is None:
        return out
    inl = inl.ravel().astype(bool)
    out["refit_inlier_frac"] = float(np.mean(inl))
    ro = readout(H_ref, w, h)
    out["ref_rot_deg"] = ro["rot_deg"]
    out["ref_flow_px"] = ro["flow_px"]

    n = len(src)
    fold = rng.permutation(n) % K_FOLDS
    ho = []
    for f in range(K_FOLDS):
        tr, te = fold != f, fold == f
        if np.sum(tr) < 8 or np.sum(te) < 1:
            continue
        Hf, inlf = cv2.findHomography(src[tr], dst[tr], **MAGSAC)
        if Hf is None:
            continue
        inlf = inlf.ravel().astype(bool)
        if np.sum(inlf) >= 8:
            Hls = fit_h_ls(src[tr][inlf], dst[tr][inlf])
            if Hls is not None:
                Hf = Hls
        ho.append(symmetric_transfer_px(Hf, src[te], dst[te]))
    if ho:
        ho = np.concatenate(ho)
        out["ho_ste_median_px"] = float(np.median(ho))
        out["ho_ste_p90_px"] = float(np.percentile(ho, 90))

    if H_vo is not None:
        d = symmetric_transfer_px(H_vo, src, dst)
        out["d_vo_median_px"] = float(np.median(d))
        out["d_vo_p90_px"] = float(np.percentile(d, 90))
        rv = readout(H_vo, w, h)
        out["vo_rot_deg"] = rv["rot_deg"]
        out["vo_flow_px"] = rv["flow_px"]

    # ---- R3: bootstrap stability --------------------------------------------------------
    bs_src, bs_dst = src[inl], dst[inl]
    out["n_boot_inliers"] = len(bs_src)
    if len(bs_src) >= 8:
        rots, flows, lss = [], [], []
        c = np.array([[w / 2.0, h / 2.0]])
        for _ in range(B_BOOT):
            idx = rng.integers(0, len(bs_src), size=len(bs_src))
            Hb = fit_h_ls(bs_src[idx], bs_dst[idx])
            if Hb is None:
                continue
            rb = readout(Hb, w, h)
            if not rb["proper"]:
                continue
            rots.append(rb["rot_deg"])
            lss.append(rb["log_scale"])
            flows.append(apply_h(Hb, c)[0])
        if len(rots) >= B_BOOT // 2:
            out["boot_rot_sd_deg"] = circular_sd_deg(np.array(rots))
            out["boot_logscale_sd"] = float(np.std(lss))
            fl = np.array(flows)
            out["boot_flow_sd_px"] = float(np.sqrt(np.mean(
                np.sum((fl - fl.mean(axis=0))**2, axis=1))))

    # ---- R4: independent pipeline -------------------------------------------------------
    def sift_feats(key: int, img: np.ndarray):
        if key not in sift_cache:
            for k in [k for k in sift_cache if k < key - 1]:
                del sift_cache[k]
            sift_cache[key] = sift.detectAndCompute(img, None)
        return sift_cache[key]

    kp0, de0 = sift_feats(i - 1, prev)
    kp1, de1 = sift_feats(i, cur)
    if de0 is not None and de1 is not None and len(kp0) >= 2 and len(kp1) >= 2:
        knn = matcher.knnMatch(de1, de0, k=2)     # query = cur, train = prev (cur -> prev)
        good = [m for m, s in (p for p in knn if len(p) == 2) if m.distance < RATIO * s.distance]
        out["sift_n_matches"] = len(good)
        if len(good) >= SIFT_MIN_MATCHES:
            s_src = np.float32([kp1[m.queryIdx].pt for m in good])
            s_dst = np.float32([kp0[m.trainIdx].pt for m in good])
            H_sift, s_inl = cv2.findHomography(s_src, s_dst, **MAGSAC)
            if H_sift is not None:
                out["sift_inlier_frac"] = float(np.mean(s_inl.ravel().astype(bool)))
                rs = readout(H_sift, w, h)
                out["sift_rot_deg"] = rs["rot_deg"]
                if H_vo is not None:
                    rv = readout(H_vo, w, h)
                    out["xchk_rot_deg"] = abs(rv["rot_deg"] - rs["rot_deg"])
                    out["xchk_logscale"] = abs(rv["log_scale"] - rs["log_scale"])
                    g = grid_points(w, h)
                    out["xchk_grid_px"] = float(np.median(
                        np.linalg.norm(apply_h(H_vo, g) - apply_h(H_sift, g), axis=1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--images-root", required=True,
                    help="directory containing the images/ tree referenced by frames.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=1)
    a = ap.parse_args()

    repo = TOOL_DIR.parents[2]
    ds_dir = repo / a.dataset
    with (ds_dir / "frames.csv").open(newline="", encoding="utf-8") as f:
        frame_rows = list(csv.DictReader(f))
    frames, sidecar = load_run(repo / a.run)
    assert len(frames) == len(frame_rows), "run/dataset frame count mismatch"

    # cross-validate the two VO pair-homography derivations on event=none rows (frozen check)
    g = None
    diffs = []
    W, Hh = None, None
    img0 = cv2.imread(str(Path(a.images_root) / frame_rows[1]["image_path"]), cv2.IMREAD_GRAYSCALE)
    assert img0 is not None, "cannot read imagery"
    Hh, W = img0.shape[:2]
    gpts = grid_points(W, Hh)
    for i in range(1, len(frames)):
        if frames[i]["event"] != "none":
            continue
        Ha, _ = vo_pair_homography(frames, sidecar, i)
        g_prev = h_of(sidecar[i - 1], G_KEYS)
        g_cur = h_of(sidecar[i], G_KEYS)
        try:
            Hb = g_prev @ np.linalg.inv(g_cur)
        except np.linalg.LinAlgError:
            continue
        diffs.append(float(np.max(np.linalg.norm(apply_h(Ha, gpts) - apply_h(Hb, gpts), axis=1))))
        if len(diffs) >= 500:
            break
    xval = {"n": len(diffs), "max_px": max(diffs) if diffs else None,
            "median_px": float(np.median(diffs)) if diffs else None}
    print(f"[panel] VO-H derivation cross-check (500-sample): {xval}", flush=True)

    sift = cv2.SIFT_create(nfeatures=SIFT_N)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rng = np.random.default_rng(SEED)
    sift_cache: dict = {}

    out_path = repo / a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_end = len(frames) if not a.limit else min(len(frames), a.start + a.limit)
    t0 = time.time()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        wtr.writeheader()
        prev_img = cv2.imread(str(Path(a.images_root) / frame_rows[a.start - 1]["image_path"]),
                              cv2.IMREAD_GRAYSCALE)
        for i in range(a.start, n_end):
            cur_img = cv2.imread(str(Path(a.images_root) / frame_rows[i]["image_path"]),
                                 cv2.IMREAD_GRAYSCALE)
            if cur_img is None or prev_img is None:
                prev_img = cur_img
                continue
            H_vo, source = vo_pair_homography(frames, sidecar, i)
            t1 = time.time()
            row = evaluate_pair(prev_img, cur_img, H_vo, sift, matcher, sift_cache, i, rng)
            row.update(frame_index=i, event=frames[i]["event"], vo_h_source=source,
                       wall_ms=round((time.time() - t1) * 1000.0, 1))
            wtr.writerow({k: row.get(k, "") for k in COLUMNS})
            prev_img = cur_img
            if (i - a.start) % 250 == 0:
                el = time.time() - t0
                done = i - a.start + 1
                print(f"[panel] {i}/{n_end}  {el:.0f}s elapsed, "
                      f"{el / max(done,1) * (n_end - i):.0f}s left", flush=True)

    meta = {"seed": SEED, "gftt": GFTT, "lk": {**LK, "winSize": list(LK["winSize"])},
            "fb_valid_px": FB_VALID_PX, "k_folds": K_FOLDS,
            "magsac": {k: v for k, v in MAGSAC.items() if k != "method"},
            "b_boot": B_BOOT, "sift_n": SIFT_N, "ratio": RATIO,
            "sift_min_matches": SIFT_MIN_MATCHES, "grid": [GRID_N, GRID_MARGIN],
            "vo_h_xval": xval, "dataset": a.dataset, "run": a.run,
            "start": a.start, "end": n_end}
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"[panel] written {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
