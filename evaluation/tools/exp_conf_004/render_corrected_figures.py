"""EXP-CONF-004 Phase 8: re-render the nominated thesis-candidate cases under the CORRECTED
coordinate convention (EXP-CONF-003 R0).

The original hypothesis-audit cards warped full-resolution imagery with the raw persisted
(processed-coordinate) VO transform — halved translation, doubled perspective bend. Here the
VO transform is conjugated to full resolution before rendering; the best-fit reference and the
correspondence diagnostics were always full-resolution and are recomputed identically.
Originals are preserved untouched; these are candidate thesis figures, not final publication
graphics.

Usage: python render_corrected_figures.py --images-root-hkb <dir> --images-root-amd <dir>
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import cv2
import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
REPO = TOOL_DIR.parents[2]
sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_002"))
sys.path.insert(0, str(TOOL_DIR.parent / "exp_conf_003"))
from panel import FB_VALID_PX, GFTT, LK, MAGSAC, load_run, vo_pair_homography   # noqa: E402
from readout import fit_h_ls                                                    # noqa: E402
from build_audit import render_case                                             # noqa: E402
from phase_a import conjugate_to_full, downsample_factor                        # noqa: E402

OUT = REPO / "evaluations/exp-conf-002/hypothesis_audit/corrected"

# (case id, sequence, frame, scientific point)
CASES = [
    ("h1.1", "hkairport01-b", 4946, "healthy baseline — what a nominal stitch looks like "
     "(corrected shipped deviation 1.6 px)"),
    ("h2.1", "hkairport01-b", 921, "estimator-internal A1-healthy/A2-degraded failure the "
     "frame before the 922 restart (corrected 58 px, real)"),
    ("h5.1", "hkairport01-b", 884, "the genuine dissociation: large image-space disagreement "
     "(44 px) with healthy navigation rotation (0.012 deg)"),
    ("h6.2", "amtown01-d", 11794, "catastrophic fully real failure: label, panel, attitude "
     "and eye agree (300 px / 3.9 deg)"),
    ("h6.3", "hkairport01-b", 5586, "the convention-artifact lesson: published 37.9 px "
     "corrects to 1.96 px; the ghost was the rendering, not the transform"),
    ("h3.2", "amtown01-d", 11889, "high-motion regime: unstable registration statistics with "
     "a healthy shipped estimate (published 102 px corrects to 1.8 px)"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root-hkb", required=True)
    ap.add_argument("--images-root-amd", required=True)
    a = ap.parse_args()
    roots = {"hkairport01-b": Path(a.images_root_hkb), "amtown01-d": Path(a.images_root_amd)}

    (OUT / "cards").mkdir(parents=True, exist_ok=True)
    ctx = {}
    import csv as _csv
    for seq in ("hkairport01-b", "amtown01-d"):
        run_dir = REPO / f"runs/{seq}-homography-rigid-conf-v1"
        frames_rows = list(_csv.DictReader(
            (REPO / f"datasets/{seq}/frames.csv").open(newline="", encoding="utf-8")))
        rframes, sidecar = load_run(run_dir)
        ctx[seq] = (frames_rows, rframes, sidecar, downsample_factor(run_dir))

    items = []
    for cid, seq, i, point in CASES:
        frames_rows, rframes, sidecar, dfac = ctx[seq]
        prev = cv2.imread(str(roots[seq] / frames_rows[i - 1]["image_path"]),
                          cv2.IMREAD_GRAYSCALE)
        cur = cv2.imread(str(roots[seq] / frames_rows[i]["image_path"]), cv2.IMREAD_GRAYSCALE)
        assert prev is not None and cur is not None, (seq, i)
        H_raw, _ = vo_pair_homography(rframes, sidecar, i)
        H_vo = conjugate_to_full(H_raw, dfac)          # the corrected-convention transform

        p0 = cv2.goodFeaturesToTrack(prev, **GFTT).reshape(-1, 2).astype(np.float32)
        p1r, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0.reshape(-1, 1, 2), None, **LK)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1r, None, **LK)
        p1 = p1r.reshape(-1, 2)
        fbv = np.linalg.norm(p0 - p0b.reshape(-1, 2), axis=1)
        valid = (st1.ravel() == 1) & (st2.ravel() == 1) & (fbv <= FB_VALID_PX)
        H_ref, inl = cv2.findHomography(p1[valid], p0[valid], **MAGSAC)
        inl = inl.ravel().astype(bool)
        if inl.sum() >= 8:
            Hls = fit_h_ls(p1[valid][inl], p0[valid][inl])
            if Hls is not None:
                H_ref = Hls

        render_case(f"{cid}-corrected", prev, cur, H_vo, H_ref, p0, p1, fbv, valid, inl,
                    OUT / "cards")
        items.append((cid, seq, i, point))
        print(f"[corrected] {cid} rendered ({seq} {i})", flush=True)

    body = ["<html><head><meta charset='utf-8'><title>EXP-CONF-002 corrected thesis-candidate "
            "figures</title></head><body style='font-family:sans-serif;max-width:1500px;"
            "margin:auto'>",
            "<h1>Corrected-convention re-renders of the nominated thesis cases</h1>",
            "<p>VO-side stitches rendered with the persisted transform conjugated to full "
            "resolution (EXP-CONF-003 R0). Candidate thesis figures, not final publication "
            "graphics; the original packet is preserved unchanged with a STATUS caveat.</p>"]
    for cid, seq, i, point in items:
        body.append(f"<h3>{cid} — {seq} frame {i}</h3><p><b>Point:</b> "
                    f"{html.escape(point)}</p>")
        for suffix in ("overview", "zoom", "diag"):
            body.append(f"<img src='cards/{cid}-corrected_{suffix}.png' "
                        f"style='max-width:100%'>")
    body.append("</body></html>")
    (OUT / "index.html").write_text("".join(body), encoding="utf-8")
    print(f"[corrected] index written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
