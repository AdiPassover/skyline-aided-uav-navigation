"""`EXP-VO-013`: score the three height arms on real imagery, without fitting scale.

One VO run, three arms. They differ **only** in the height array multiplied into the per-frame
increments — the same architecture `EXP-VO-012` used, which structurally guarantees that no arm can
perturb the estimator and makes the comparison free.

    A  fixed      h_used(t) = h0                        the null: what the readout implicitly assumes
    B  external   h_used(t) = h0 + dh_rtk(t)            the datum a barometer would give (class C)
    C  oracle     h_used(t) = h_lidar(t)                the height the geometry actually wants (class D)

Everything numeric here is scale-preserving by default. `EXP-VO-012` measured that a Sim(2)-aligned
metric is *exactly* blind to a constant metric-scale error, so the historical normalised ATE is
computed and reported as a **secondary continuity** number only, and is labelled as scale-fitted
everywhere it appears.

    PY=.venv/Scripts/python.exe
    $PY evaluation/tools/exp_vo_013/analyse.py \
        --dataset datasets/amtown01-d --run runs/amtown01-d-homography-rigid-v1 \
        --lidar <dense lidar profile json> --out evaluations/exp-vo-013
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVALUATION_DIR = HERE.parents[1]
REPO = EVALUATION_DIR.parent
sys.path.insert(0, str(EVALUATION_DIR))
sys.path.insert(0, str(HERE.parent / "exp_vo_012"))

import metric_metrics as mm                                              # noqa: E402
import metric_readout as mr                                              # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402
from naveval.sync import synchronize                                     # noqa: E402

# Declared before scoring (`EXP-VO-013` Metrics). `LIT-VO-003` §10.6: at step 1 the path-length
# ratio is inflated 76 % on real imagery by noise rectification, so the stride is not a free
# parameter and is not chosen after seeing a result.
PATH_STEP = 5
RPE_LENGTHS = (10, 25, 50, 100)


def load_lidar_profile(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """(t, height above the imaged surface) from a dense `altitude_sources.lidar_agl_series` report.

    This is height above **the imaged surface**, not the `up_m` column of `groundtruth.csv`, which is
    height above the **takeoff datum**. `LIT-VO-004` §4: they coincide only over flat ground and on
    no MARS-LVIG sequence do they coincide. Conflating them is the single error this experiment is
    built to measure the cost of.
    """
    obj = json.loads(Path(path).read_text())
    s = obj["_series"] if "_series" in obj else obj
    t = np.asarray(s["t"], float)
    h = np.asarray(s["agl"] if "agl" in s else s["height_m"], float)
    o = np.argsort(t)
    return t[o], h[o]


def arm_heights(frame_t: np.ndarray, gt_t: np.ndarray, gt_up: np.ndarray,
                lidar_t: np.ndarray, lidar_h: np.ndarray) -> tuple[dict, dict]:
    """The three height arrays, plus the provenance of each. `h0` is a LiDAR reading, not a fit.

    **`h0` is the LiDAR median in-cone range at the reference frame** — an independently measured
    initial AGL, i.e. an oracle/known-reference initialisation. It is NOT obtained by fitting the
    estimated trajectory to ground truth, and it is not a function of the trajectory at all.
    """
    h_lidar = np.interp(frame_t, lidar_t, lidar_h)
    h0 = float(h_lidar[0])

    up = np.interp(frame_t, gt_t, gt_up)
    dh_ext = up - up[0]

    heights = {
        "fixed": np.full(frame_t.shape, h0),
        "external": h0 + dh_ext,
        "oracle": h_lidar,
    }
    provenance = {
        "h0_m": h0,
        "h0_source": ("LiDAR median in-camera-cone range at the reference frame "
                      "(class D, oracle/known-reference initialisation). NOT fitted to any "
                      "trajectory or to ground-truth position."),
        "fixed": "h0 held constant. The null arm: what the readout implicitly assumes today.",
        "external": ("h0 + (RTK takeoff-relative altitude - its value at the reference frame). "
                     "Class C, RTK/GNSS-derived. This is NOT a barometer; it stands in for the "
                     "DATUM a barometer would report, because MARS-LVIG's actual FC "
                     "height-above-takeoff channel is populated with nothing (EXP-VO-013 B1)."),
        "oracle": ("LiDAR-measured height above the imaged surface, linearly interpolated onto "
                   "frame timestamps. Class D. An ORACLE: no deployed monocular system has it."),
        "dh_external_ptp_m": float(np.ptp(dh_ext)),
        "lidar_sample_interval_s": float(np.median(np.diff(lidar_t))) if len(lidar_t) > 1 else None,
        "lidar_interpolation": "linear (offline analysis; a causal system would hold, see DEC-VO-007 D5)",
    }
    return heights, provenance


# Declared here, not tuned: the window over which the camera's yaw mounting angle is derived.
# 60 s is 1.3 % of this flight's path and the first straight leg fits inside it.
FRAME_ROTATION_WINDOW_S = 60.0
# Courses are taken over CHORDS, not consecutive frames. At 4 m/s and 10 Hz a per-frame step is
# 0.4 m, so feature-localisation noise of order 0.1 m puts +/-14 deg on each course and the
# circular spread swamps the quantity being measured (measured: 38.7 deg). A 50-frame chord is
# ~20 m, where the same noise is worth ~0.3 deg, while 60 s of heading drift stays small.
FRAME_ROTATION_STRIDE = 50
FRAME_ROTATION_MIN_STEP_M = 1.0


def derive_frame_rotation(est_xy: np.ndarray, gt_xy: np.ndarray, frame_t: np.ndarray,
                          window_s: float = FRAME_ROTATION_WINDOW_S) -> dict:
    """The CCW rotation taking the estimator's own frame onto ENU — the camera's yaw MOUNTING angle.

    **Why this is needed, and why it is not cheating.** `reference_initialised` claims zero fitted
    parameters by rotating the estimate by the ground-truth compass heading at the reference frame.
    That is exact for a renderer, whose camera axes are the world axes. It is *wrong* for a real
    aircraft, because the heading of the **airframe** is not the heading of the **camera**: the two
    differ by a fixed mounting yaw. MARS-LVIG does not publish one. `LIT-006` had to derive the
    RTK **antenna** baseline offset empirically for exactly the same reason (+269.16°, a quarter
    turn, because the antennas are mounted across the airframe); the camera's offset has never been
    derived here, because until this experiment every VO metric used a Sim(2) or SE(2) alignment
    that fits rotation and so never needed it.

    Derived by `LIT-006` Stage 2's method, adapted: the circular mean of (ground-truth course over
    ground − estimated course over ground) over the **first `window_s` seconds only**, gated on
    both courses being well defined. It is therefore **one extrinsic taken from a declared 60 s
    window**, not a parameter fitted to the 1,189 s trajectory under test — and it is computed from
    displacement DIRECTIONS, which are identical across the three height arms, so it cannot move the
    arm comparison at all.

    Reported alongside the whole-trajectory SE(2) fit, so the difference between "the mounting
    angle" and "the mounting angle plus accumulated heading drift" is visible rather than assumed.
    """
    est = np.asarray(est_xy, float)
    gt = np.asarray(gt_xy, float)
    t = np.asarray(frame_t, float)
    m = (t - t[0]) <= window_s
    if m.sum() < 10:
        raise ValueError("frame-rotation window holds fewer than 10 frames")
    s = FRAME_ROTATION_STRIDE
    de = est[m][s:] - est[m][:-s]
    dg = gt[m][s:] - gt[m][:-s]
    ok = ((np.linalg.norm(de, axis=1) > FRAME_ROTATION_MIN_STEP_M)
          & (np.linalg.norm(dg, axis=1) > FRAME_ROTATION_MIN_STEP_M))
    if ok.sum() < 10:
        raise ValueError("frame-rotation window holds fewer than 10 moving frames")
    d = np.arctan2(dg[ok, 1], dg[ok, 0]) - np.arctan2(de[ok, 1], de[ok, 0])
    ang = float(np.degrees(np.arctan2(np.sin(d).mean(), np.cos(d).mean())) % 360.0)
    # Circular SD of the per-frame differences, as a quality figure.
    r = float(np.hypot(np.sin(d).mean(), np.cos(d).mean()))
    sd = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(r, 1e-12)), 0.0))))
    return {"camera_frame_rotation_deg": ang, "circular_sd_deg": sd,
            "standard_error_deg": float(sd / max(np.sqrt(ok.sum()), 1.0)),
            "n_chords_used": int(ok.sum()), "chord_stride_frames": s, "window_s": window_s,
            "method": ("circular mean of (GT course over ground - estimated course over ground) "
                       "over %d-frame chords within the first %.0f s, LIT-006 Stage 2's method. "
                       "One extrinsic from a "
                       "declared window; NOT fitted to the trajectory under test, and identical "
                       "across arms because course direction is arm-independent."
                       % (FRAME_ROTATION_STRIDE, window_s))}


def closed_form_prediction(h_used: np.ndarray, h_true: np.ndarray,
                           gt_xy: np.ndarray) -> dict:
    """`EXP-VO-012` R7/R8's closed form: a wrong height enters as a PATH-WEIGHTED mean of the ratio.

    Computed from the LiDAR profile and the ground-truth path alone — no VO output is involved — so
    it is a genuine prediction of what the arm's path-length ratio will be, not a fit to it.
    """
    step = np.hypot(np.diff(gt_xy[:, 0]), np.diff(gt_xy[:, 1]))
    ratio = h_used / np.maximum(h_true, 1e-9)
    mid = 0.5 * (ratio[:-1] + ratio[1:])
    w = step.sum()
    return {
        "path_weighted_mean_h_used_over_h_true": float((mid * step).sum() / w) if w > 0 else float("nan"),
        "unweighted_mean_ratio": float(ratio.mean()),
        "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
    }


def run(dataset_dir: Path, run_dir: Path, lidar_path: Path, out_dir: Path) -> dict:
    ds = load_dataset(dataset_dir)
    # `camera_intrinsics` is carried in dataset.json's metadata block but is not a field of
    # naveval's Metadata dataclass, so it is read from the file rather than from the loader. Without
    # it there is no f_working and no metre is derivable at all -- hence a hard failure, never a
    # default.
    raw_meta = json.loads((Path(dataset_dir) / "dataset.json").read_text()).get("metadata", {})
    fx = (raw_meta.get("camera_intrinsics") or {}).get("fx")
    if fx is None:
        raise SystemExit("dataset.json carries no camera intrinsics; f_working is not derivable")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["estimator_config"]
    down = int(cfg.get("downsampleFactor", 1))
    f_work = mr.f_working(float(fx), down)
    width = ds.metadata.image_width // down
    height = ds.metadata.image_height // down

    inc = mr.load_increments(run_dir, width, height)
    frame_t = inc.timestamps_s
    if frame_t.size != len(inc):
        raise SystemExit("frames.csv did not supply a timestamp for every sidecar row")

    sync = synchronize(frame_t, ds.gt_timestamps, ds.gt_east, ds.gt_north,
                       gt_heading=ds.gt_heading, gt_valid=ds.gt_valid,
                       clock_offset_s=ds.clock_offset_s,
                       clock_drift_s_per_s=ds.clock_drift_s_per_s)
    keep = sync.frame_indices
    gt_xy = np.stack([sync.east, sync.north], axis=1)
    heading0 = float(sync.heading[0]) if sync.has_heading else 0.0

    lidar_t, lidar_h = load_lidar_profile(lidar_path)
    heights, prov = arm_heights(frame_t, ds.gt_timestamps, ds.gt_up, lidar_t, lidar_h)

    # The camera's yaw mounting angle, derived once from the oracle arm's early motion. Heading is
    # arm-independent, so one value serves all three and no arm can be advantaged by it.
    probe = mr.integrate_metric(inc, heights["oracle"], f_work, arm="probe")
    rot = derive_frame_rotation(probe.xy()[keep], gt_xy, frame_t[keep])
    psi_cam = rot["camera_frame_rotation_deg"]

    results = {}
    tracks = {}
    for arm, h in heights.items():
        track = mr.integrate_metric(inc, h, f_work, arm=arm)
        est_xy = track.xy()[keep]
        s = mm.score(est_xy, gt_xy, psi_cam, path_step=PATH_STEP, rpe_lengths=RPE_LENGTHS)
        # ...and the same score with the naive rotation (GT airframe heading at frame 0), which is
        # what `reference_initialised` assumes for a renderer. Kept so the cost of the assumption
        # is visible rather than argued about.
        s["ref_init_airframe_heading"] = mm.score(
            est_xy, gt_xy, heading0, path_step=PATH_STEP, rpe_lengths=RPE_LENGTHS)["ref_init"]
        s["h_used_m"] = {"min": float(h.min()), "max": float(h.max()), "mean": float(h.mean())}
        s["closed_form"] = closed_form_prediction(h[keep], heights["oracle"][keep], gt_xy)
        results[arm] = s
        tracks[arm] = track

    # The estimator's own diagnostics, for H6 and the stability report.
    n = len(inc)
    events = inc.events
    stability = {
        "frames_in_sidecar": n,
        "frames_synced": int(keep.size),
        "excluded_out_of_span": int(sync.excluded_out_of_span_count),
        "restarts": sum(1 for e in events if e == "restart"),
        "recenters": sum(1 for e in events if e == "recenter"),
        "frame_success_pct": 100.0 * float(manifest.get("processed_count", n)) / max(1, manifest.get("frame_count", n)),
    }

    # The terrain decomposition: h_AGL = h0 + dh_aircraft - dh_terrain (LIT-VO-006 eq. 4).
    up_f = np.interp(frame_t, ds.gt_timestamps, ds.gt_up)
    dh_aircraft = up_f - up_f[0]
    h_agl = heights["oracle"]
    dh_terrain = prov["h0_m"] + dh_aircraft - h_agl
    decomposition = {
        "dh_aircraft_m": {"min": float(dh_aircraft.min()), "max": float(dh_aircraft.max()),
                          "ptp": float(np.ptp(dh_aircraft)), "sd": float(dh_aircraft.std())},
        "dh_terrain_m": {"min": float(dh_terrain.min()), "max": float(dh_terrain.max()),
                         "ptp": float(np.ptp(dh_terrain)), "sd": float(dh_terrain.std())},
        "h_agl_m": {"min": float(h_agl.min()), "max": float(h_agl.max()),
                    "max_over_min": float(h_agl.max() / h_agl.min())},
        "share_of_agl_variation_from_terrain": float(
            np.ptp(dh_terrain) / max(np.ptp(dh_terrain) + np.ptp(dh_aircraft), 1e-9)),
    }

    # Sim(2) blindness (H5), stated as a ratio so it cannot be read as an accuracy claim.
    a, c = results["fixed"], results["oracle"]
    sim2_a = 100.0 * a["sim2"]["ate_rmse_normalised"]
    sim2_c = 100.0 * c["sim2"]["ate_rmse_normalised"]
    h5 = {
        "sim2_nate_pct_fixed": sim2_a,
        "sim2_nate_pct_oracle": sim2_c,
        "sim2_relative_difference_pct": 100.0 * abs(sim2_c - sim2_a) / max(sim2_a, 1e-12),
        "refinit_endpoint_pct_fixed": a["ref_init"]["endpoint_error_pct_of_path"],
        "refinit_endpoint_pct_oracle": c["ref_init"]["endpoint_error_pct_of_path"],
        "refinit_relative_difference_pct": 100.0 * abs(
            c["ref_init"]["endpoint_error_pct_of_path"] - a["ref_init"]["endpoint_error_pct_of_path"])
        / max(a["ref_init"]["endpoint_error_pct_of_path"], 1e-12),
    }

    gt_path = float(np.hypot(np.diff(gt_xy[:, 0]), np.diff(gt_xy[:, 1])).sum())
    out = {
        "dataset": ds.dataset_id,
        "run": run_dir.name,
        "estimator_config": cfg,
        "f_working_px": f_work,
        "fx_native_px": float(fx),
        "downsample_factor": down,
        "path_step": PATH_STEP,
        "gt_path_length_m": gt_path,
        "gt_endpoint_to_start_m": float(np.hypot(*(gt_xy[-1] - gt_xy[0]))),
        "gt_closure_pct_of_path": 100.0 * float(np.hypot(*(gt_xy[-1] - gt_xy[0]))) / max(gt_path, 1e-9),
        "camera_frame_rotation": rot,
        "gt_heading0_deg": heading0,
        "height_provenance": prov,
        "terrain_decomposition": decomposition,
        "stability": stability,
        "sim2_blindness": h5,
        "arms": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))

    # Per-frame series for the figures, kept out of results.json so it stays readable.
    np.savez_compressed(out_dir / "series.npz",
                        frame_t=frame_t, keep=keep, gt_east=sync.east, gt_north=sync.north,
                        h_fixed=heights["fixed"], h_external=heights["external"],
                        h_oracle=heights["oracle"],
                        **{f"{a}_{k}": getattr(tracks[a], k)
                           for a in tracks for k in ("east_m", "north_m", "yaw_deg")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--lidar", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = run(Path(a.dataset), Path(a.run), Path(a.lidar), Path(a.out))

    p = out["height_provenance"]
    print(f"dataset {out['dataset']}  run {out['run']}")
    print(f"  f_working = {out['fx_native_px']:.4f} / {out['downsample_factor']} "
          f"= {out['f_working_px']:.4f} px")
    print(f"  h0 = {p['h0_m']:.2f} m (LiDAR at the reference frame)")
    r = out["camera_frame_rotation"]
    print(f"  camera frame rotation = {r['camera_frame_rotation_deg']:.3f} deg "
          f"(+/- {r['circular_sd_deg']:.2f} circ sd, {r['n_chords_used']} chords in the first "
          f"{r['window_s']:.0f} s); GT airframe heading at frame 0 = {out['gt_heading0_deg']:.2f} deg")
    print(f"  GT path {out['gt_path_length_m']:.0f} m, closure "
          f"{out['gt_closure_pct_of_path']:.2f} % of path")
    d = out["terrain_decomposition"]
    print(f"  AGL {d['h_agl_m']['min']:.1f}..{d['h_agl_m']['max']:.1f} m "
          f"(x{d['h_agl_m']['max_over_min']:.2f});  aircraft ptp "
          f"{d['dh_aircraft_m']['ptp']:.2f} m, terrain ptp {d['dh_terrain_m']['ptp']:.2f} m "
          f"({100 * d['share_of_agl_variation_from_terrain']:.1f} % of the variation is terrain)")
    print(f"  stability: {out['stability']}")
    print()
    print(f"  {'arm':10s} {'refinit end (m)':>16s} {'% of path':>10s} {'RMSE (m)':>10s} "
          f"{'path ratio':>11s} {'SE2 ATE':>9s} {'Sim2 nATE %':>12s}")
    for arm in ("fixed", "external", "oracle"):
        r = out["arms"][arm]
        print(f"  {arm:10s} {r['ref_init']['endpoint_error_m']:16.2f} "
              f"{r['ref_init']['endpoint_error_pct_of_path']:10.3f} "
              f"{r['ref_init']['ate_rmse_m']:10.2f} {r['path_length_ratio']:11.4f} "
              f"{r['se2']['ate_rmse_m']:9.2f} "
              f"{100.0 * r['sim2']['ate_rmse_normalised']:12.4f}")
    print()
    b = out["sim2_blindness"]
    print(f"  H5: metric family separates fixed vs oracle by "
          f"{b['refinit_relative_difference_pct']:.1f} % relative; "
          f"Sim(2)-aligned by {b['sim2_relative_difference_pct']:.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
