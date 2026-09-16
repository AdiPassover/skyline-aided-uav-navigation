"""EXP-INT-001 DEV calibration sweep: replay every candidate policy over the recorded DEV tracks and
apply the pre-registered selection rule.

The grid, the DEV recordings, the classification thresholds and the selection rule are those
written into the EXP-INT-001 record §Calibration **before** this script was first
run; the script only executes them. For every (matcher arm, θ_s, θ_m, R_ambiguity) it writes a
`RelocalizationConfig`, replays it with `RelocalizationReplayApp` over each DEV recording's VO_ONLY
run (the VO is never re-run), tabulates the accepted re-anchors against ground truth with
`evaluate_int_arms.events_only`, and pools the counts. Selection is lexicographic and mechanical:

1. discard any policy with an accepted **wrong-place** re-anchor (reference > 50 m from the query);
2. discard any policy with a **severely harmful** drift correction (``delta_e > +15 m``);
3. among the rest, take the policy with the most accepted **genuine-revisit** re-anchors
   (reference within 20 m);
4. ties: the most conservative — higher θ_s, then higher θ_m, then smaller R_ambiguity.

R_dual and R_temporal are not in the acceptance path of the milestone policy (``weakest_view``,
temporal ``fallback``); they are fixed at their semantic values and a sensitivity block shows the
acceptance set is invariant to them. Under ``strict_agreement`` R_dual *does* gate, and the sweep
also reports that arm so R_dual can be chosen by its own meaning.

Run from the repository root (needs ``gradlew installDist`` first)::

    python evaluation/tools/int/dev_calibration_sweep.py --out evaluations/exp-int-001-dev/sweep [--workers 6]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import evaluate_int_arms as ev  # noqa: E402

REPO = HERE.parents[2]
CLASSPATH = str(REPO / "build" / "install" / "skyline-aided-uav-navigation" / "lib" / "*")

DEV = ["fig8-flat-const-v1", "fig8-flat-vary-v1", "mtn-r1-const-v1", "mtn-r2-vary-v1", "mtn-r3-vary-v1"]
VO_RUN = "runs/exp-int-001-{ds}-vo-only"

# ---- the pre-registered grid (EXP-INT-001 §Calibration) ----
THETA_S = [0.90, 0.95, 0.97, 0.98, 0.99]
THETA_M = [0.10, 0.20, 0.30]
R_AMBIGUITY = [20.0, 30.0, 50.0]
R_DUAL_FIXED = 30.0
R_TEMPORAL_FIXED = 30.0
SENSITIVITY_RADII = [15.0, 30.0, 50.0]
ARMS = {"c0": {"matcher": "c0_frozen_ncc", "max_lag_samples": 0},
        "c1-4": {"matcher": "c1_bounded_lag_ncc", "max_lag_samples": 4}}
WRONG_PLACE_M, SEVERE_HARM_M, GENUINE_M = 50.0, 15.0, 20.0

FIXED = {
    "descriptor_length": 256, "degenerate_std_floor": 0.001, "min_overlap_frac": 0.6,
    "novelty_min_distance": 0.05, "max_spacing_frames": 100, "search_exposure_bound": 300,
    "recent_exclusion_mode": "time", "recent_exclusion_seconds": 30.0,
    "region_rule": "position_radius", "top_k": 5,
    "west_match_threshold": None, "west_margin_threshold": None,
    "temporal_confirmation": "fallback", "skyline_sync_tolerance_s": 0.0,
    "min_retry_gap_frames": 30, "confirm_queries": 2, "region_track_max_gap": 2,
}


def config(arm: str, ts: float, tm: float, r_amb: float, r_dual: float, r_temp: float, fusion: str) -> dict:
    c = dict(FIXED)
    c.update(ARMS[arm])
    c.update({"fusion_rule": fusion, "match_threshold": ts, "margin_threshold": tm,
              "ambiguity_region_radius_m": r_amb, "dual_agreement_radius_m": r_dual,
              "temporal_region_radius_m": r_temp})
    return c


def cid(arm, ts, tm, r_amb, r_dual, r_temp, fusion) -> str:
    f = {"weakest_view": "w", "strict_agreement": "s"}[fusion]
    return f"{arm}_{f}_s{ts:.2f}_m{tm:.2f}_a{r_amb:g}_d{r_dual:g}_t{r_temp:g}"


def replay_one(job) -> dict:
    cfg_path, ds, out_dir = job
    vo = REPO / VO_RUN.format(ds=ds)
    run_dir = out_dir / ds
    if not (run_dir / "alignment_events.csv").exists():
        cmd = ["java", "-cp", CLASSPATH, "org.boofcv.evaluation.RelocalizationReplayApp",
               "--vo-run", str(vo), "--relocalization-config", str(cfg_path),
               "--skyline-profiles", str(REPO / "datasets" / ds / "skyline_profiles.csv"), "--out", str(run_dir)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        if r.returncode != 0:
            raise RuntimeError(f"{cfg_path.name} / {ds}: {r.stderr[-800:]}")
    res = ev.events_only(REPO / "datasets" / ds, run_dir, run_dir, WRONG_PLACE_M, SEVERE_HARM_M, GENUINE_M)
    return {"config": cfg_path.stem, "dataset": ds, **{k: v for k, v in res["reanchors"].items()
                                                     if k not in ("gt_query_to_reference_m", "delta_e_m")},
            "retrievals": res["recency_density"].get("retrievals", 0),
            "no_competing_hypothesis": res["recency_density"].get("no_competing_hypothesis", 0),
            "ambiguous_region": res["recency_density"].get("ambiguous_region", 0),
            "weak_match": res["recency_density"].get("weak_match", 0),
            "gt_query_to_reference_m": ";".join(f"{d:.1f}" for d in res["reanchors"]["gt_query_to_reference_m"]),
            "delta_e_m": ";".join(f"{d:.1f}" for d in res["reanchors"]["delta_e_m"])}


def select(rows: list[dict], arm: str, fusion_code: str) -> dict:
    pooled = {}
    for r in rows:
        if not r["config"].startswith(f"{arm}_{fusion_code}_"):
            continue
        p = pooled.setdefault(r["config"], {"config": r["config"], "total": 0, "genuine_revisit": 0, "wrong_place": 0,
                                            "severely_harmful": 0, "harmful": 0, "beneficial": 0})
        for k in ("total", "genuine_revisit", "wrong_place", "severely_harmful", "harmful", "beneficial"):
            p[k] += r[k]
    cands = [p for p in pooled.values() if p["wrong_place"] == 0 and p["severely_harmful"] == 0]

    def key(p):
        parts = p["config"].split("_")
        ts = float(parts[2][1:]); tm = float(parts[3][1:]); ra = float(parts[4][1:])
        return (p["genuine_revisit"], ts, tm, -ra)

    cands.sort(key=key, reverse=True)
    return {"arm": arm, "fusion": fusion_code, "eliminated_wrong_place_or_severe": len(pooled) - len(cands),
            "selected": cands[0] if cands else None, "ranked": cands[:10], "pooled": sorted(pooled.values(), key=lambda p: p["config"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--skip-sensitivity", action="store_true")
    a = ap.parse_args(argv)
    out = a.out if a.out.is_absolute() else REPO / a.out
    (out / "configs").mkdir(parents=True, exist_ok=True)

    jobs = []
    specs = []
    for arm in a.arms:
        for fusion in ("weakest_view", "strict_agreement"):
            for ts, tm, ra in itertools.product(THETA_S, THETA_M, R_AMBIGUITY):
                specs.append((arm, ts, tm, ra, R_DUAL_FIXED, R_TEMPORAL_FIXED, fusion))
    for spec in specs:
        name = cid(*spec)
        cfg_path = out / "configs" / f"{name}.json"
        cfg_path.write_text(json.dumps(config(*spec), indent=2) + "\n")
        for ds in DEV:
            jobs.append((cfg_path, ds, out / "replays" / name))
    print(f"{len(specs)} policies x {len(DEV)} DEV recordings = {len(jobs)} replays")
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(replay_one, jobs), 1):
            rows.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)}")
    fields = list(rows[0].keys())
    with (out / "sweep_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    selection = {"grid": {"theta_s": THETA_S, "theta_m": THETA_M, "r_ambiguity_m": R_AMBIGUITY,
                          "r_dual_fixed_m": R_DUAL_FIXED, "r_temporal_fixed_m": R_TEMPORAL_FIXED},
                 "thresholds": {"wrong_place_m": WRONG_PLACE_M, "severe_harm_m": SEVERE_HARM_M, "genuine_m": GENUINE_M},
                 "dev_recordings": DEV, "rule": __doc__.split("Selection is")[1].split("R_dual")[0].strip(),
                 "arms": {}}
    for arm in a.arms:
        for fc in ("w", "s"):
            selection["arms"][f"{arm}_{fc}"] = select(rows, arm, fc)

    # Sensitivity: at each arm's selected weakest-view policy, vary R_dual and R_temporal.
    if not a.skip_sensitivity:
        sens = {}
        for arm in a.arms:
            sel = selection["arms"][f"{arm}_w"]["selected"]
            if sel is None:
                continue
            parts = sel["config"].split("_")
            ts = float(parts[2][1:]); tm = float(parts[3][1:]); ra = float(parts[4][1:])
            sjobs = []
            for rd, rt in itertools.product(SENSITIVITY_RADII, SENSITIVITY_RADII):
                spec = (arm, ts, tm, ra, rd, rt, "weakest_view")
                name = cid(*spec)
                cfg_path = out / "configs" / f"{name}.json"
                cfg_path.write_text(json.dumps(config(*spec), indent=2) + "\n")
                for ds in DEV:
                    sjobs.append((cfg_path, ds, out / "replays" / name))
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
                srows = list(ex.map(replay_one, sjobs))
            by_cfg = {}
            for r in srows:
                by_cfg.setdefault(r["config"], []).append(
                    f"{r['dataset']}:{r['total']}/{r['genuine_revisit']}/{r['wrong_place']}")
            sens[arm] = {k: sorted(v) for k, v in by_cfg.items()}
            rows.extend(srows)
        selection["sensitivity_r_dual_r_temporal_weakest_view"] = sens
        with (out / "sweep_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (out / "selection.json").write_text(json.dumps(selection, indent=2, default=ev._json_default) + "\n")
    for k, v in selection["arms"].items():
        s = v["selected"]
        print(f"{k}: eliminated {v['eliminated_wrong_place_or_severe']}; selected "
              f"{s['config'] if s else None} -> {s if s else 'none survives'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
