"""EXP-CONF-003 Phase A: per-pair navigation-relevant decomposition of A2.

For every valid consecutive pair of a development sequence: recompute the independent
reference registration H_ref (forward-backward-validated KLT correspondences -> MAGSAC++ ->
LS refit on the MAGSAC inlier set; the EXP-CONF-002 panel's frozen R2 machinery), derive the
frozen VO's persisted pair transform H_vo, and decompose their disagreement into the
navigation-consumed similarity part and the projective residual (motion_decomp.decompose).
H_ref matrices are persisted so later phases (the Phase C counterfactual scoring) never
recompute correspondences.

COORDINATE CONVENTION (established 2026-08-31 by this phase's readout gate): the persisted
run-record homographies are maps in PROCESSED (downsampled) pixel coordinates — the conf runs
use downsampleFactor 2, so 1224x1024, and frame-0 currToWorld = [0.5, 0, 306; 0, 0.5, 256]
is the canvas shrink for the PROCESSED frame. Read at the processed centre they reproduce the
production sidecar's inc_rotation_deg / inc_flow_px / inc_log_scale exactly. EXP-CONF-002's
panel evaluated these transforms as full-resolution maps against full-resolution
correspondences, which inflates its image-space A2 quantities (d_vo, xchk_*) by a
flow/perspective-dependent artifact. Here H_vo is conjugated to full resolution
(H_full = D H D^-1, D = diag(f, f, 1)) before any comparison, and the per-pair column
`d_tot_misread_px` records the EXP-CONF-002-convention disagreement so the artifact's share
can be quantified.

Startup gate (frozen in the record): rot(H_vo, full after conjugation) must agree with the
production sidecar's inc_rotation_deg up to a global sign, to < 0.01 deg median over the
first 500 event=none pairs — otherwise this analysis readout is not the navigator's and the
run aborts.

SIFT quantities are NOT recomputed here; they come from the committed panel CSVs (and carry
the same published-convention caveat).

Usage:
  python phase_a.py --sequence hkairport01-b --images-root <dir with images/> \
      [--limit N] [--start K] [--out <csv>]
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
REPO = TOOL_DIR.parents[2]
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_002"))
from panel import FB_VALID_PX, GFTT, LK, MAGSAC, load_run, vo_pair_homography   # noqa: E402
from readout import apply_h, fit_h_ls, readout                                  # noqa: E402
from motion_decomp import decompose, grid_points, wrap_deg                      # noqa: E402

GATE_N = 500
GATE_MEDIAN_DEG = 0.01

DECOMP_KEYS = [
    "rot_a_deg", "rot_b_deg", "rot_dis_deg",
    "flow_a_px", "flow_b_px", "flowvec_dis_px", "flowmag_dis_px", "dir_dis_deg",
    "ls_a", "ls_b", "logscale_dis",
    "nonrigid_a_px", "nonrigid_b_px", "d_tot_px", "d_sim_px", "d_proj_px",
]
H_KEYS = [f"r{i}{j}" for i in range(3) for j in range(3)]
COLUMNS = (["frame_index", "event", "n_fb_valid", "n_inl"]
           + DECOMP_KEYS + ["d_tot_misread_px"] + H_KEYS
           + ["t_klt_ms", "t_magsac_ms", "t_refit_ms"])


def downsample_factor(run_dir: Path) -> float:
    with (run_dir / "manifest.json").open(encoding="utf-8") as f:
        m = json.load(f)
    return float(m["estimator_config"]["downsampleFactor"])


def conjugate_to_full(H: np.ndarray, f: float) -> np.ndarray:
    D = np.diag([f, f, 1.0])
    return D @ H @ np.linalg.inv(D)


def sidecar_inc_rot(sidecar: list[dict]) -> np.ndarray:
    return np.array([float(r["inc_rotation_deg"]) for r in sidecar])


def run_gate(frames: list[dict], sidecar: list[dict], w: int, h: int, f: float) -> dict:
    """rot(conjugated H_vo) vs the production readout's inc_rotation_deg, up to a global sign."""
    inc = sidecar_inc_rot(sidecar)
    diffs = {1.0: [], -1.0: []}
    n = 0
    for i in range(1, len(frames)):
        if frames[i]["event"] != "none":
            continue
        H_vo, _ = vo_pair_homography(frames, sidecar, i)
        if H_vo is None:
            continue
        r = readout(conjugate_to_full(H_vo, f), w, h)["rot_deg"]
        for s in (1.0, -1.0):
            diffs[s].append(abs(wrap_deg(r - s * inc[i])))
        n += 1
        if n >= GATE_N:
            break
    med = {s: float(np.median(v)) for s, v in diffs.items()}
    sign = 1.0 if med[1.0] <= med[-1.0] else -1.0
    return {"n": n, "median_abs_diff_deg": {str(s): m for s, m in med.items()},
            "rot_readout_sign": sign, "passed": med[sign] < GATE_MEDIAN_DEG}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--run-id", default="",
                    help="run record to score (default: <sequence>-homography-rigid-conf-v1)")
    ap.add_argument("--stride", type=int, default=1,
                    help="score every Nth pair (EXP-CONF-005 reduced rule); extra indices "
                         "from --extra-frames are always scored")
    ap.add_argument("--extra-frames", default="",
                    help="comma-separated frame indices always scored regardless of stride")
    a = ap.parse_args()

    seq = a.sequence
    run_dir = REPO / f"runs/{a.run_id or f'{seq}-homography-rigid-conf-v1'}"
    ds_dir = REPO / f"datasets/{seq}"
    with (ds_dir / "frames.csv").open(newline="", encoding="utf-8") as f:
        frame_rows = list(csv.DictReader(f))
    frames, sidecar = load_run(run_dir)
    assert len(frames) == len(frame_rows)

    img0 = cv2.imread(str(Path(a.images_root) / frame_rows[0]["image_path"]),
                      cv2.IMREAD_GRAYSCALE)
    assert img0 is not None, "cannot read imagery"
    h, w = img0.shape[:2]
    dfac = downsample_factor(run_dir)

    gate = run_gate(frames, sidecar, w, h, dfac)
    print(f"[phase_a] {seq} readout gate: {gate}", flush=True)
    if not gate["passed"]:
        print("[phase_a] GATE FAILED — the analysis readout does not reproduce the "
              "navigator's inc_rotation_deg; aborting per the frozen front half.")
        return 2

    out_path = REPO / (a.out or f"evaluations/exp-conf-003/phase_a/{seq}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_pts = grid_points(w, h)
    n_end = len(frames) if not a.limit else min(len(frames), a.start + a.limit)
    extra = {int(x) for x in a.extra_frames.split(",") if x.strip()}
    selected = {i for i in range(a.start, n_end)
                if a.stride <= 1 or i % a.stride == 0 or i in extra}
    t0 = time.time()
    n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        wtr.writeheader()
        prev = cv2.imread(str(Path(a.images_root) / frame_rows[a.start - 1]["image_path"]),
                          cv2.IMREAD_GRAYSCALE)
        for i in range(a.start, n_end):
            cur = cv2.imread(str(Path(a.images_root) / frame_rows[i]["image_path"]),
                             cv2.IMREAD_GRAYSCALE)
            if cur is None or prev is None:
                prev = cur
                continue
            H_vo, _src = vo_pair_homography(frames, sidecar, i)
            row: dict = {"frame_index": i, "event": frames[i]["event"]}
            if i not in selected:
                wtr.writerow({k: row.get(k, "") for k in COLUMNS})
                prev = cur
                continue

            t1 = time.time()
            p0 = cv2.goodFeaturesToTrack(prev, **GFTT)
            if p0 is not None and len(p0) >= 8:
                p0 = p0.reshape(-1, 2).astype(np.float32)
                p1r, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0.reshape(-1, 1, 2),
                                                       None, **LK)
                p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1r, None, **LK)
                p1 = p1r.reshape(-1, 2)
                fb = np.linalg.norm(p0 - p0b.reshape(-1, 2), axis=1)
                valid = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb <= FB_VALID_PX)
                row["n_fb_valid"] = int(valid.sum())
                row["t_klt_ms"] = round((time.time() - t1) * 1000.0, 2)
                if valid.sum() >= 12:
                    t2 = time.time()
                    H_ref, inl = cv2.findHomography(p1[valid], p0[valid], **MAGSAC)
                    row["t_magsac_ms"] = round((time.time() - t2) * 1000.0, 2)
                    if H_ref is not None:
                        inl = inl.ravel().astype(bool)
                        row["n_inl"] = int(inl.sum())
                        if inl.sum() >= 8:
                            t3 = time.time()
                            Hls = fit_h_ls(p1[valid][inl], p0[valid][inl])
                            row["t_refit_ms"] = round((time.time() - t3) * 1000.0, 2)
                            if Hls is not None:
                                H_ref = Hls
                        Hn = H_ref / H_ref[2, 2]
                        for k, v in zip(H_KEYS, Hn.ravel()):
                            row[k] = repr(float(v))
                        if H_vo is not None:
                            H_vo_full = conjugate_to_full(H_vo, dfac)
                            d = decompose(H_vo_full, H_ref, w, h)
                            for k in DECOMP_KEYS:
                                row[k] = d[k]
                            # the EXP-CONF-002 convention (raw processed-coords H read as a
                            # full-resolution map): its grid disagreement quantifies the
                            # published d_vo/xchk artifact per pair
                            g = grid_pts
                            row["d_tot_misread_px"] = float(np.median(np.linalg.norm(
                                apply_h(H_vo, g) - apply_h(H_ref, g), axis=1)))
            wtr.writerow({k: row.get(k, "") for k in COLUMNS})
            n_rows += 1
            prev = cur
            if (i - a.start) % 500 == 0:
                el = time.time() - t0
                done = i - a.start + 1
                print(f"[phase_a] {seq} {i}/{n_end}  {el:.0f}s elapsed, "
                      f"{el / max(done, 1) * (n_end - i):.0f}s left", flush=True)

    meta = {"sequence": seq, "gate": gate, "downsample_factor": dfac,
            "coordinate_note": "persisted run-record homographies are processed-coordinate "
                               "maps; conjugated to full resolution before comparison; "
                               "d_tot_misread_px is the EXP-CONF-002-convention disagreement",
            "gftt": GFTT,
            "lk": {**LK, "winSize": list(LK["winSize"])}, "fb_valid_px": FB_VALID_PX,
            "magsac": {k: v for k, v in MAGSAC.items() if k != "method"},
            "start": a.start, "end": n_end, "n_rows": n_rows,
            "stride": a.stride, "n_extra_frames": len(extra),
            "images_root": a.images_root, "image_size": [w, h],
            "vo_run": str(run_dir.relative_to(REPO)).replace("\\", "/")}
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"[phase_a] written {out_path} ({n_rows} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
