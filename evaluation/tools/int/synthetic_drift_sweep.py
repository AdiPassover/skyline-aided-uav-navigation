"""Controlled mechanism study (`EXP-INT-002` §18/§19): real skyline retrieval and memory, declared
synthetic VO drift — where is the crossover at which a discrete-reference snap starts to help?

For every drift setting the script derives a drifted VO_ONLY run (`inject_synthetic_drift.py`),
replays one policy over it (`RelocalizationReplayApp`; the matcher, memory, scheduler and gate are
the real ones), evaluates drifted-VO versus INT-on-drifted-VO (`evaluate_int_arms.py`) and follows
every accepted correction forward (`snap_longitudinal.py`). The table it writes has one row per
(setting, accepted snap): the drift accumulated since the reference `D_gq`, the reference floor
`|u + r|`, the pre/post errors, the online-visible correction size `|c|` (`position_jump_m`), the
horizon benefits and the class — plus one row per setting with the trajectory verdict.

Settings are given as a JSON list, e.g.::

    [{"model": "linear", "rate": 0.05, "bearing_deg": 45}, {"model": "linear", "rate": 0.2, "bearing_deg": 45}]

Run from the repository root (needs `gradlew installDist`)::

    python evaluation/tools/int/synthetic_drift_sweep.py --dataset datasets/<id> --vo-run runs/<vo-only> \\
        --policy <RelocalizationConfig.json> --settings settings.json --out evaluations/<dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import evaluate_int_arms as ev  # noqa: E402
import snap_longitudinal as sl  # noqa: E402

REPO = HERE.parents[2]
CLASSPATH = str(REPO / "build" / "install" / "skyline-aided-uav-navigation" / "lib" / "*")


def setting_id(s: dict) -> str:
    parts = [s["model"]]
    for k in ("rate", "frac", "scale", "step", "step_frame", "sigma", "seed", "bearing_deg"):
        if k in s:
            parts.append(f"{k}{s[k]:g}")
    return "_".join(parts)


def run(cmd, cwd=REPO):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    if r.returncode != 0:
        raise RuntimeError(" ".join(map(str, cmd)) + "\n" + r.stderr[-2000:])
    return r.stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--vo-run", required=True, type=Path)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default="c0")
    a = ap.parse_args(argv)
    settings = json.loads(a.settings.read_text())
    profiles = a.dataset / "skyline_profiles.csv"
    a.out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    snap_rows, setting_rows = [], []
    for s in settings:
        sid = setting_id(s)
        drifted = a.out / "drifted_runs" / sid
        cmd = [py, str(HERE / "inject_synthetic_drift.py"), "--vo-run", str(a.vo_run), "--out", str(drifted), "--model", s["model"]]
        for k in ("rate", "frac", "scale", "step", "step_frame", "sigma", "seed", "bearing_deg"):
            if k in s:
                cmd += [f"--{k.replace('_', '-')}", str(s[k])]
        if s.get("allow_segments"):
            cmd.append("--allow-segments")
        run(cmd)
        replay = a.out / "replays" / sid / a.label
        run(["java", "-cp", CLASSPATH, "org.boofcv.evaluation.RelocalizationReplayApp", "--vo-run", str(drifted),
             "--relocalization-config", str(a.policy), "--skyline-profiles", str(profiles), "--out", str(replay)])
        res = ev.compare(a.dataset, drifted, {a.label: replay}, replay / "eval", 50.0, 15.0, 20.0, ev.MATERIAL_IMPROVEMENT_PCT)
        arm = res["arms"][a.label]
        lon = sl.analyse(a.dataset, drifted, replay, replay / "longitudinal", [5.0, 10.0, 20.0, 30.0], 0.5)
        man = json.loads((drifted / "manifest.json").read_text())["synthetic_drift"]
        setting_rows.append({
            "setting": sid, **{k: s.get(k) for k in ("model", "rate", "frac", "scale", "step", "step_frame", "sigma", "bearing_deg")},
            "final_drift_m": man["final_drift_m"]["magnitude"], "max_drift_m": man["max_drift_m"],
            "vo_ate_m": arm["vo_only_on_common_frames"]["ate_rmse_m"], "int_ate_m": arm["int_on_common_frames"]["ate_rmse_m"],
            "ate_change_pct": arm["ate_change_pct_negative_is_better"],
            "vo_final_m": arm["vo_only_on_common_frames"]["final_m"], "int_final_m": arm["int_on_common_frames"]["final_m"],
            "reanchors": arm["reanchors"]["total"], "genuine": arm["reanchors"]["genuine_revisit"], "wrong_place": arm["reanchors"]["wrong_place"],
            "beneficial": arm["reanchors"]["beneficial"], "harmful": arm["reanchors"]["harmful"], "severely_harmful": arm["reanchors"]["severely_harmful"],
            "classes": json.dumps(lon["classification_counts"]), "verdict": arm["verdict_mechanical"],
            "retrievals": arm["recency_density"].get("retrievals", 0), "weak_match": arm["recency_density"].get("weak_match", 0),
            "ambiguous_region": arm["recency_density"].get("ambiguous_region", 0),
        })
        for sn in lon["snaps"]:
            h = sn["horizons"]
            tail = h["to_next_snap"] or h["to_end_isolated"]
            snap_rows.append({
                "setting": sid, "final_drift_m": man["final_drift_m"]["magnitude"], "frame": sn["frame"], "time_s": sn["time_s"],
                "recovery_case": sn["recovery_case"], "reference_id": sn["reference_id"], "reference_age_s": sn["reference_age_s"],
                "u_m": sn["gt_query_to_reference_m"], "r_m": sn["stored_reference_pose_error_m"],
                "floor_u_plus_r_m": sn["e_after_m"], "D_gq_m": sn["vo_drift_since_reference_insertion_m"],
                "e_before_m": sn["e_before_m"], "e_after_m": sn["e_after_m"], "delta_e_0_m": sn["delta_e_0_m"],
                "correction_c_m": sn["correction_m"], "position_jump_online_m": sn["position_jump_m"],
                "delta_e_10s_m": (h.get("10s") or {}).get("delta_e_iso_m"), "delta_e_30s_m": (h.get("30s") or {}).get("delta_e_iso_m"),
                "delta_e_tail_m": (tail or {}).get("delta_e_iso_m"), "dA_tail_m_s": (tail or {}).get("dA_iso_vs_nok_m_s"),
                "tail_s": (tail or {}).get("reached_s"), "cos_c_d_tail": (tail or {}).get("cos_c_d"), "class": sn["classification"],
            })
        print(f"{sid}: drift {man['final_drift_m']['magnitude']:.1f} m | VO ATE {arm['vo_only_on_common_frames']['ate_rmse_m']:.2f} -> INT "
              f"{arm['int_on_common_frames']['ate_rmse_m']:.2f} ({arm['ate_change_pct_negative_is_better']:+.1f} %) final "
              f"{arm['vo_only_on_common_frames']['final_m']:.2f} -> {arm['int_on_common_frames']['final_m']:.2f} | snaps {arm['reanchors']['total']} "
              f"({lon['classification_counts']}) | {arm['verdict_mechanical']}")
    for name, rows in (("settings.csv", setting_rows), ("snaps.csv", snap_rows)):
        with (a.out / name).open("w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            else:
                f.write("setting\n")
    (a.out / "summary.json").write_text(json.dumps({"dataset": str(a.dataset), "vo_run": str(a.vo_run), "policy": str(a.policy),
                                                    "settings": setting_rows, "snaps": snap_rows}, indent=2, default=ev._json_default) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
