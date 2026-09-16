"""Ground-truth-only figures and an integrity report, produced **before** any VO is run
(`EXP-VO-007` *Procedure* step 3, ingest gates G1-G10).

The point of running this first is that the dataset's geometry has to be understood without
reference to estimator performance. Every number here comes from `dataset.json`, `frames.csv`,
`groundtruth.csv` and `attitude.csv`; nothing here reads a run record, and the script fails if asked
to.

Usage::

    python evaluation/tools/exp_vo_007/plot_groundtruth.py \
        --dataset datasets/amtown01-c --out evaluations/exp-vo-007/ingest \
        [--height-profile <candidate_probe height_profile JSON>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

EVALUATION_DIR = Path(__file__).resolve().parents[2]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from height_profile import load_height_profile                           # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402

# EXP-VO-007 ingest gates.
GATE_RATE_HZ = (9.8, 10.2)
GATE_MAX_GAP_S = 0.3
GATE_FRAME_COUNT_TOL = 0.01
GATE_RTK_FIXED_FRAC = 0.99
GATE_AGL_TOL_M = 0.5
GATE_ATTITUDE_VS_COURSE_DEG = 10.0
EXPECTED_IMAGE_SIZE = (2448, 2048)


def _wrap180(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def gates(ds, root: Path, target_agl_m: float) -> dict:
    g, t = {}, ds.gt_timestamps
    ft = np.asarray(ds.frame_timestamps if hasattr(ds, "frame_timestamps") else [])
    if ft.size == 0:
        with (root / "frames.csv").open(newline="") as f:
            ft = np.array([float(r["timestamp_s"]) for r in csv.DictReader(f)])
    dt = np.diff(ft)
    span = float(ft[-1] - ft[0])
    rate = float((ft.size - 1) / span) if span > 0 else 0.0
    expected = span * 10.0 + 1
    g["G1_frame_count"] = {"n": int(ft.size), "expected_at_10hz": round(expected, 1),
                           "pass": abs(ft.size - expected) / expected <= GATE_FRAME_COUNT_TOL}
    g["G2_monotonic"] = {"strictly_increasing": bool(np.all(dt > 0)),
                         "max_gap_s": float(dt.max()), "min_gap_s": float(dt.min()),
                         "pass": bool(np.all(dt > 0) and dt.max() <= GATE_MAX_GAP_S)}
    g["G3_frame_rate"] = {"measured_hz": round(rate, 4),
                          "pass": GATE_RATE_HZ[0] <= rate <= GATE_RATE_HZ[1]}
    g["G4_gt_covers_frames"] = {
        "frame_range": [float(ft[0]), float(ft[-1])],
        "gt_range": [float(t[0]), float(t[-1])],
        "pass": bool(t[0] <= ft[0] and t[-1] >= ft[-1])}

    with (root / "groundtruth.csv").open(newline="") as f:
        gt_rows = list(csv.DictReader(f))
    valid = np.array([r["valid"].strip().lower() in ("true", "1") for r in gt_rows])
    g["G5_rtk_fix"] = {"n": int(valid.size), "valid_fraction": float(valid.mean()),
                       "pass": bool(valid.mean() >= GATE_RTK_FIXED_FRAC)}

    up = np.array([float(r["up_m"]) for r in gt_rows])
    g["G6_agl_plateau"] = {"mean_m": float(up.mean()), "sd_m": float(up.std()),
                           "min_m": float(up.min()), "max_m": float(up.max()),
                           "target_m": target_agl_m,
                           "pass": bool(np.all(np.abs(up - target_agl_m) <= GATE_AGL_TOL_M))}

    meta = json.loads((root / "dataset.json").read_text())["metadata"]
    conv = meta.get("heading_conversion", {})
    g["G7_yaw_convention"] = {"offset_deg": conv.get("offset_deg"),
                              "derivation": (conv.get("derivation") or "")[:200],
                              "pass": conv.get("offset_deg") is not None}

    att_p = root / "attitude.csv"
    if att_p.exists():
        with att_p.open(newline="") as f:
            arows = list(csv.DictReader(f))
        at = np.array([float(r["timestamp_s"]) for r in arows])
        ayaw = np.array([float(r["yaw_compass_deg"]) for r in arows])
        arate = float((at.size - 1) / (at[-1] - at[0]))
        east, north = np.asarray(ds.gt_east), np.asarray(ds.gt_north)
        de, dn = np.diff(east), np.diff(north)
        moving = np.hypot(de, dn) > 0.2
        course = np.degrees(np.arctan2(de, dn)) % 360.0
        ct = 0.5 * (t[1:] + t[:-1])
        ay_at = np.interp(ct[moving], at, np.unwrap(np.radians(ayaw)))
        d = _wrap180(np.degrees(ay_at) - course[moving])
        g["G8_attitude"] = {"n": int(at.size), "rate_hz": round(arate, 2),
                            "attitude_minus_course_mean_deg": round(float(d.mean()), 2),
                            "attitude_minus_course_sd_deg": round(float(d.std()), 2),
                            "pass": bool(abs(d.mean()) <= GATE_ATTITUDE_VS_COURSE_DEG
                                         and arate > 50)}
    else:
        g["G8_attitude"] = {"pass": False, "reason": "attitude.csv missing"}

    g["G9_images"] = {"declared": [meta.get("image_width"), meta.get("image_height")],
                      "expected": list(EXPECTED_IMAGE_SIZE),
                      "pass": [meta.get("image_width"), meta.get("image_height")]
                              == list(EXPECTED_IMAGE_SIZE)}
    g["G10_contract"] = {"has_heading": bool(ds.has_heading),
                         "position_quality": ds.position_quality.quality_class,
                         "heading_quality": ds.heading_quality.quality_class if ds.heading_quality else None,
                         "evidence_tier": ds.evidence_tier,
                         "pass": bool(ds.has_heading
                                      and ds.position_quality.quality_class == "rtk_gnss")}
    g["all_pass"] = all(v.get("pass") for v in g.values() if isinstance(v, dict))
    return g


def figures(ds, root: Path, out: Path, profile: dict | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t = np.asarray(ds.gt_timestamps)
    e, n = np.asarray(ds.gt_east), np.asarray(ds.gt_north)
    rel = t - t[0]
    seg = np.hypot(np.diff(e), np.diff(n))
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    speed = seg / np.maximum(np.diff(t), 1e-9)
    with (root / "groundtruth.csv").open(newline="") as f:
        up = np.array([float(r["up_m"]) for r in csv.DictReader(f)])

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))
    ax[0, 0].plot(e, n, lw=0.8)
    ax[0, 0].plot(e[0], n[0], "go", ms=6, label="start")
    ax[0, 0].plot(e[-1], n[-1], "rs", ms=6, label="end")
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_title(f"ground-truth XY ({dist[-1]:.0f} m)")
    ax[0, 0].set_xlabel("east (m)"); ax[0, 0].set_ylabel("north (m)"); ax[0, 0].legend()

    ax[0, 1].plot(rel, up, lw=0.8)
    ax[0, 1].set_title(f"height above takeoff: {up.mean():.3f} +/- {up.std():.3f} m")
    ax[0, 1].set_xlabel("t (s)"); ax[0, 1].set_ylabel("up (m)")

    ax[0, 2].plot(dist[1:], speed, lw=0.6)
    ax[0, 2].set_title(f"speed: {speed.mean():.2f} +/- {speed.std():.2f} m/s")
    ax[0, 2].set_xlabel("distance (m)"); ax[0, 2].set_ylabel("m/s")

    if ds.has_heading:
        ax[1, 0].plot(rel, np.asarray(ds.gt_heading), lw=0.6)
        ax[1, 0].set_title("ground-truth heading"); ax[1, 0].set_ylabel("compass deg")
    else:
        ax[1, 0].text(0.5, 0.5, "no heading ground truth", ha="center")
    ax[1, 0].set_xlabel("t (s)")

    att_p = root / "attitude.csv"
    if att_p.exists():
        with att_p.open(newline="") as f:
            ar = list(csv.DictReader(f))
        at = np.array([float(r["timestamp_s"]) for r in ar]) - t[0]
        roll = np.array([float(r["roll_deg"]) for r in ar])
        pitch = np.array([float(r["pitch_deg"]) for r in ar])
        tilt = np.array([float(r["tilt_deg"]) for r in ar])
        ax[1, 1].plot(at, roll, lw=0.3, label="roll")
        ax[1, 1].plot(at, pitch, lw=0.3, label="pitch")
        ax[1, 1].set_title(f"airframe attitude (tilt p50 {np.median(tilt):.2f}, "
                           f"max {tilt.max():.2f} deg)")
        ax[1, 1].set_xlabel("t (s)"); ax[1, 1].set_ylabel("deg"); ax[1, 1].legend()
    else:
        ax[1, 1].text(0.5, 0.5, "no attitude sidecar", ha="center")

    if profile:
        pt = np.array(profile["t"]) - t[0]
        ph = np.array(profile["height_m"])
        ax[1, 2].plot(pt, ph, lw=0.9, label="height above imaged surface (LiDAR)")
        ax[1, 2].axhline(up.mean(), color="k", ls="--", lw=0.8, label="height above takeoff")
        ax[1, 2].set_title(f"effective height: p5-p95 ratio "
                           f"{np.percentile(ph, 95) / np.percentile(ph, 5):.2f}")
        ax[1, 2].set_xlabel("t (s)"); ax[1, 2].set_ylabel("m"); ax[1, 2].legend(fontsize=8)
    else:
        ax[1, 2].text(0.5, 0.5, "no LiDAR height profile", ha="center")

    fig.suptitle(f"{ds.dataset_id}: ground truth only, before any VO run "
                 f"(EXP-VO-007 ingest gates)")
    fig.tight_layout()
    fig.savefig(out / "fig0_groundtruth_only.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height-profile", default="",
                    help="probe_candidates.py report, or a bare {t, height_m} object")
    ap.add_argument("--scene", default="",
                    help="which sequence to read from a multi-sequence probe report; "
                         "defaults to the dataset's own metadata.flight_id")
    ap.add_argument("--target-agl", type=float, default=80.0)
    a = ap.parse_args()
    root = REPO / a.dataset if not Path(a.dataset).is_absolute() else Path(a.dataset)
    ds = load_dataset(root)
    scene = a.scene or json.loads((root / "dataset.json").read_text())["metadata"].get("flight_id")
    profile = load_height_profile(a.height_profile, scene) if a.height_profile else None
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    g = gates(ds, root, a.target_agl)
    (out / "ingest_gates.json").write_text(json.dumps(g, indent=2, default=float))
    figures(ds, root, out, profile)
    for k, v in g.items():
        if isinstance(v, dict):
            print(f"{'PASS' if v.get('pass') else 'FAIL'}  {k}: "
                  f"{ {kk: vv for kk, vv in v.items() if kk != 'pass'} }")
    print(f"\nALL GATES PASS: {g['all_pass']}")
    raise SystemExit(0 if g["all_pass"] else 1)


if __name__ == "__main__":
    main()
