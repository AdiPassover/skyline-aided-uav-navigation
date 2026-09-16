"""EXP-INT-004: comparison arms for the frozen INT method — VO_ONLY, NAIVE_1V, NAIVE_2V, FROZEN —
replayed over every committed VO_ONLY track (thesis claim-closure pass 2026-09, Phase 3).

    $PY evaluation/tools/claim_closure/int_comparison_arms.py --out evaluations/claim-closure-2026-09/int

The two naive policies are the pre-registered files under
evaluation/eval_configs/int/claim-closure-2026-09/ (written before the first replay and asserted to
differ from the frozen policy only in the declared keys); the frozen policy's SHA-256 is re-verified.
Replays use RelocalizationReplayApp (byte-identical to the live loop; re-verified against the
committed live arms where they exist). Evaluation: evaluate_int_arms.py unchanged, plus sidecar
counts (requests by cause, rejections by reason, queries per km / minute, UNKNOWN time, hard-loss
recovery latency) and exact zero-count upper bounds.
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "evaluation/tools/int"))
from evaluate_int_arms import load_gt                                            # noqa: E402

PY = sys.executable
JAVA_CP = str(REPO / "build/install/skyline-aided-uav-navigation/lib/*")
CFG_DIR = REPO / "evaluation/eval_configs/int/claim-closure-2026-09"
FROZEN_CFG = REPO / "evaluation/eval_configs/int/exp-int-002/reloc-c0-primary-retry20.json"
FROZEN_SHA = "b5457092c59767319bf8c40d993549c667ff4238b13eb0f7789660643ef80399"
ARMS = {"NAIVE_1V": CFG_DIR / "reloc-naive-1v-top1.json",
        "NAIVE_2V": CFG_DIR / "reloc-naive-2v-top1.json",
        "FROZEN": FROZEN_CFG}
NAIVE_ALLOWED_DIFF = {"region_rule", "region_gap_references", "fusion_rule", "temporal_confirmation", "margin_threshold",
                      "ambiguity_region_radius_m", "dual_agreement_radius_m", "temporal_region_radius_m"}

# (label, dataset, vo_only run, role, committed live FROZEN run if any)
TRACKS = [
    ("fig8-flat-const", "fig8-flat-const-v1", "exp-int-001-fig8-flat-const-v1-vo-only", "DEV", None),
    ("fig8-flat-vary", "fig8-flat-vary-v1", "exp-int-001-fig8-flat-vary-v1-vo-only", "DEV", None),
    ("mtn-r1-const", "mtn-r1-const-v1", "exp-int-001-mtn-r1-const-v1-vo-only", "DEV", None),
    ("mtn-r2-vary", "mtn-r2-vary-v1", "exp-int-001-mtn-r2-vary-v1-vo-only", "DEV", "exp-int-002-mtn-r2-vary-v1-int-c0-retry20"),
    ("mtn-r2-vary-synthetic-loss", "mtn-r2-vary-v1", "exp-int-001-mtn-r2-vary-v1-vo-only-synthetic-loss", "DEV-synthetic-loss",
     "exp-int-002-mtn-r2-vary-v1-int-c0-retry20-synthetic-loss"),
    ("mtn-r3-vary", "mtn-r3-vary-v1", "exp-int-001-mtn-r3-vary-v1-vo-only", "DEV", None),
    ("interesting-r1-vary", "interesting-r1-vary-v1", "exp-int-001-interesting-r1-vary-v1-vo-only", "evaluation", None),
    ("interesting-r3-const", "interesting-r3-const-v1", "exp-int-001-interesting-r3-const-v1-vo-only", "control", None),
    ("easier-sq-const", "easier-sq-const-v1", "exp-int-002-easier-sq-const-v1-vo-only", "first-look", None),
    ("ho1-mtn-fig8", "ho1-mtn-fig8-vary-v1", "exp-int-003-ho1-mtn-fig8-vary-v1-vo-only", "held-out", "exp-int-003-ho1-mtn-fig8-vary-v1-int-c0"),
    ("ho1-mtn-fig8-scaled", "ho1-mtn-fig8-scaled-vary-v1", "exp-int-003-ho1-mtn-fig8-scaled-vary-v1-vo-only", "held-out", "exp-int-003-ho1-mtn-fig8-scaled-vary-v1-int-c0"),
    ("ho1-vil-fig8", "ho1-vil-fig8-vary-v1", "exp-int-003-ho1-vil-fig8-vary-v1-vo-only", "held-out", "exp-int-003-ho1-vil-fig8-vary-v1-int-c0"),
    ("ho1-mtn-trinity", "ho1-mtn-trinity-vary-v1", "exp-int-003-ho1-mtn-trinity-vary-v1-vo-only", "held-out", "exp-int-003-ho1-mtn-trinity-vary-v1-int-c0"),
]
ROLE_ORDER = ["held-out", "evaluation", "control", "first-look", "DEV", "DEV-synthetic-loss"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upper95_zero(n: int) -> float | None:
    return None if n <= 0 else 1.0 - 0.05 ** (1.0 / n)


def check_configs() -> dict:
    out = {"frozen_sha256": sha256(FROZEN_CFG), "frozen_sha_ok": sha256(FROZEN_CFG) == FROZEN_SHA}
    if not out["frozen_sha_ok"]:
        raise SystemExit("the frozen policy file does not match DEC-INT-009's hash")
    frozen = json.loads(FROZEN_CFG.read_text(encoding="utf-8"))
    for arm in ("NAIVE_1V", "NAIVE_2V"):
        p = json.loads(ARMS[arm].read_text(encoding="utf-8"))
        diff = {k for k in set(frozen) | set(p) if frozen.get(k, "<absent>") != p.get(k, "<absent>")}
        if not diff <= NAIVE_ALLOWED_DIFF:
            raise SystemExit(f"{arm} differs from the frozen policy in undeclared keys: {sorted(diff - NAIVE_ALLOWED_DIFF)}")
        out[arm] = {"sha256": sha256(ARMS[arm]), "differs_in": sorted(diff)}
    return out


def replay(vo_run: Path, cfg: Path, profiles: Path, out: Path) -> float:
    if (out / "alignment_frames.csv").exists() and (out / "alignment_events.csv").exists():
        return float("nan")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run(["java", "-cp", JAVA_CP, "org.boofcv.evaluation.RelocalizationReplayApp",
                        "--vo-run", str(vo_run), "--relocalization-config", str(cfg),
                        "--skyline-profiles", str(profiles), "--out", str(out)],
                       capture_output=True, text=True, cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"replay failed for {out}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return time.time() - t0


def path_length_m(dataset: Path) -> float:
    gt = load_gt(dataset)
    ks = sorted(gt)
    xy = np.asarray([(gt[k][1], gt[k][2]) for k in ks])
    return float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))))


def sidecar_stats(run: Path, vo_frames: list[dict], err_curve: dict | None) -> dict:
    frames = read_csv(run / "alignment_frames.csv")
    events = read_csv(run / "alignment_events.csv")
    n = len(frames)
    t = np.asarray([float(r["timestamp_s"]) for r in frames])
    valid = np.asarray([r["global_position_valid"].strip().lower() == "true" for r in frames])
    dt = float(np.median(np.diff(t))) if n > 1 else 0.1
    kinds = Counter(r["kind"] for r in events)
    reasons = Counter(r["reason"] for r in events if r["kind"] == "candidate_rejected")
    verdicts = Counter(r["verdict"] for r in events if r["kind"] == "attempt")
    causes = Counter(r["cause"] for r in events if r["kind"] == "search_requested")
    losses = [int(r["frame_index"]) for r in events if r["kind"] == "hard_loss"]
    loss_sources = Counter(r["loss_source"] for r in events if r["kind"] == "hard_loss")
    recovery = []
    for f in losses:
        after = [(int(r["frame_index"])) for r in frames if int(r["frame_index"]) >= f
                 and r["global_position_valid"].strip().lower() == "true"]
        first = after[0] if after else None
        rec = {"loss_frame": f, "first_valid_frame": first,
               "latency_frames": (first - f) if first is not None else None,
               "latency_s": ((first - f) * dt) if first is not None else None}
        if first is not None and err_curve is not None and first in err_curve:
            rec["error_at_restoration_m"] = err_curve[first]
        recovery.append(rec)
    return {"frames": n, "seconds": float(t[-1] - t[0]) if n > 1 else 0.0,
            "unknown_frames": int((~valid).sum()), "unknown_fraction": float((~valid).mean()) if n else 0.0,
            "unknown_seconds": float((~valid).sum() * dt),
            "queries": int(kinds.get("retrieval", 0)), "attempts": int(kinds.get("attempt", 0)),
            "search_requests": int(kinds.get("search_requested", 0)), "requests_by_cause": dict(causes),
            "attempt_verdicts": dict(verdicts), "candidate_rejections_by_reason": dict(reasons),
            "references_inserted": int(kinds.get("reference_inserted", 0)),
            "hard_losses": losses, "loss_sources": dict(loss_sources), "recovery": recovery}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "evaluations/claim-closure-2026-09/int")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {"experiment": "EXP-INT-004", "configs": check_configs(), "arms": list(ARMS), "tracks": {}}

    for label, ds, vo_run, role, live in TRACKS:
        dataset = REPO / "datasets" / ds
        vo = REPO / "runs" / vo_run
        profiles = dataset / "skyline_profiles.csv"
        tr: dict = {"dataset": ds, "vo_only_run": vo_run, "role": role, "replays": {}}
        for arm, cfg in ARMS.items():
            rp = out / "replays" / label / arm
            secs = replay(vo, cfg, profiles, rp)
            tr["replays"][arm] = {"dir": str(rp.relative_to(REPO)), "replay_wall_s": secs}
        if live:
            lp = REPO / "runs" / live
            tr["frozen_replay_equals_committed_live"] = {
                "live_run": live,
                "events_identical": filecmp.cmp(out / "replays" / label / "FROZEN" / "alignment_events.csv", lp / "alignment_events.csv", shallow=False),
                "frames_identical": filecmp.cmp(out / "replays" / label / "FROZEN" / "alignment_frames.csv", lp / "alignment_frames.csv", shallow=False)}
        ev_dir = out / "eval" / label
        cmd = [PY, str(REPO / "evaluation/tools/int/evaluate_int_arms.py"), "--dataset", str(dataset), "--vo-only", str(vo),
               "--out", str(ev_dir)] + sum([["--int", f"{arm}={out / 'replays' / label / arm}"] for arm in ARMS], [])
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        if r.returncode != 0:
            raise SystemExit(f"evaluation failed for {label}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        metrics = json.loads((ev_dir / "metrics.json").read_text(encoding="utf-8"))
        # error curves per arm (for recovery error) from error_curve.csv
        curve_rows = read_csv(ev_dir / "error_curve.csv")
        err_by_arm = {arm: {int(r["frame"]): float(r[f"err_{arm}_m"]) for r in curve_rows if r.get(f"err_{arm}_m", "") != ""} for arm in ARMS}
        vo_frames = read_csv(vo / "alignment_frames.csv")
        tr["path_length_m"] = path_length_m(dataset)
        tr["vo_only"] = sidecar_stats(vo, vo_frames, None)
        tr["arms"] = {}
        for arm in ARMS:
            m = metrics["arms"][arm]
            s = sidecar_stats(out / "replays" / label / arm, vo_frames, err_by_arm[arm])
            re = m["reanchors"]
            tr["arms"][arm] = {
                "ate_vo_m": m["vo_only_on_common_frames"]["ate_rmse_m"], "ate_arm_m": m["int_on_common_frames"]["ate_rmse_m"],
                "ate_change_pct": m["ate_change_pct_negative_is_better"],
                "final_vo_m": m["vo_only_on_common_frames"]["final_m"], "final_arm_m": m["int_on_common_frames"]["final_m"],
                "max_vo_m": m["vo_only_on_common_frames"].get("max_m"), "max_arm_m": m["int_on_common_frames"].get("max_m"),
                "common_frames": m["int_on_common_frames"].get("n_frames"),
                "reanchors": re, "wrong_place_upper95_if_zero": upper95_zero(re["total"]) if re["wrong_place"] == 0 else None,
                "verdict_mechanical": m["verdict_mechanical"], "sidecar": s,
                "queries_per_km": s["queries"] / max(1e-9, tr["path_length_m"] / 1000.0),
                "queries_per_min": s["queries"] / max(1e-9, s["seconds"] / 60.0),
            }
        result["tracks"][label] = tr
        print(f"{label} [{role}]: " + "; ".join(
            f"{arm}: ATE {tr['arms'][arm]['ate_vo_m']:.1f}->{tr['arms'][arm]['ate_arm_m']:.1f} m ({tr['arms'][arm]['ate_change_pct']:+.1f} %), "
            f"acc {tr['arms'][arm]['reanchors']['total']} wp {tr['arms'][arm]['reanchors']['wrong_place']} harm {tr['arms'][arm]['reanchors']['harmful']}"
            for arm in ARMS))

    # ---- totals per arm, by role group -------------------------------------------------------------
    totals = {}
    for group, roles in (("held-out", ["held-out"]), ("all", ROLE_ORDER)):
        totals[group] = {}
        for arm in ARMS:
            acc = wp = harm = sev = gen = ben = 0
            better = worse = same = 0
            for label, tr in result["tracks"].items():
                if tr["role"] not in roles:
                    continue
                a = tr["arms"][arm]
                re = a["reanchors"]
                acc += re["total"]; wp += re["wrong_place"]; harm += re["harmful"]; sev += re["severely_harmful"]
                gen += re["genuine_revisit"]; ben += re["beneficial"]
                d = a["ate_change_pct"]
                if d < -1.0:
                    better += 1
                elif d > 1.0:
                    worse += 1
                else:
                    same += 1
            totals[group][arm] = {"accepts": acc, "genuine": gen, "wrong_place": wp, "harmful": harm, "severely_harmful": sev,
                                  "beneficial": ben, "recordings_ate_better": better, "recordings_ate_worse": worse,
                                  "recordings_ate_same": same,
                                  "wrong_place_upper95_if_zero": upper95_zero(acc) if wp == 0 else None,
                                  "wrong_or_severe_upper95_if_zero": upper95_zero(acc) if (wp + sev) == 0 else None}
    result["totals"] = totals
    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    # ---- summary.md ------------------------------------------------------------------------------
    lines = ["# EXP-INT-004 — comparison arms (generated by int_comparison_arms.py)", "",
             "All INT arms are replays of the frozen layer over the committed VO_ONLY tracks; FROZEN replays are "
             "byte-identical to the committed live arms where those exist (see results.json). ATE / final on the frames "
             "where VO_ONLY and the arm are both VALID (paired). 'wp' = wrong-place accept (> 50 m); 'harm' = drift "
             "correction with Δe₀ > 0; 'sev' = Δe₀ > +15 m; 'unk' = frames with the persistent position UNKNOWN.", ""]
    for role in ROLE_ORDER:
        rows = [(l, t) for l, t in result["tracks"].items() if t["role"] == role]
        if not rows:
            continue
        lines.append(f"## {role}\n")
        lines.append("| recording | arm | ATE VO → arm (m) | Δ ATE | final VO → arm (m) | accepts | genuine | wp | harm | sev | unk frames VO / arm | queries (per km / per min) | requests by cause | rejections |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for label, tr in rows:
            for arm in ARMS:
                a = tr["arms"][arm]; re = a["reanchors"]; s = a["sidecar"]
                lines.append(f"| {label} | {arm} | {a['ate_vo_m']:.1f} → {a['ate_arm_m']:.1f} | {a['ate_change_pct']:+.1f} % | "
                             f"{a['final_vo_m']:.1f} → {a['final_arm_m']:.1f} | {re['total']} | {re['genuine_revisit']} | **{re['wrong_place']}** | {re['harmful']} | {re['severely_harmful']} | "
                             f"{tr['vo_only']['unknown_frames']} / {s['unknown_frames']} | {s['queries']} ({a['queries_per_km']:.1f} / {a['queries_per_min']:.1f}) | "
                             f"{', '.join(f'{k} {v}' for k, v in sorted(s['requests_by_cause'].items()))} | "
                             f"{', '.join(f'{k} {v}' for k, v in sorted(s['candidate_rejections_by_reason'].items()))} |")
            rec = [(arm, r) for arm in ARMS for r in tr["arms"][arm]["sidecar"]["recovery"]]
            if rec:
                lines.append("")
                lines.append(f"Hard-loss recovery on {label} (loss sources {tr['vo_only']['loss_sources']}): " + "; ".join(
                    f"{arm} loss@{r['loss_frame']} → first valid {r['first_valid_frame']} (latency {r['latency_frames']} frames"
                    + (f", restored error {r['error_at_restoration_m']:.1f} m" if r.get('error_at_restoration_m') is not None else "") + ")"
                    if r["first_valid_frame"] is not None else f"{arm} loss@{r['loss_frame']} → never restored"
                    for arm, r in rec))
        lines.append("")
    lines.append("## Totals\n")
    lines.append("| group | arm | accepts | genuine | wrong-place | harmful | severely harmful | beneficial | recordings ATE better / same / worse | 95 % upper bound on P(wrong-place) if 0 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for group, per in totals.items():
        for arm, t in per.items():
            ub = t["wrong_place_upper95_if_zero"]
            lines.append(f"| {group} | {arm} | {t['accepts']} | {t['genuine']} | **{t['wrong_place']}** | {t['harmful']} | {t['severely_harmful']} | {t['beneficial']} | "
                         f"{t['recordings_ate_better']} / {t['recordings_ate_same']} / {t['recordings_ate_worse']} | {('%.3f' % ub) if ub is not None else '—'} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-12:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
