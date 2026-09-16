"""EXP-INT-005: resource accounting of the integrated system on one documented machine
(thesis claim-closure pass 2026-09, Phase 4). Assembles committed per-frame timings (VO process
time, motion-stage time, CONF refit time), the INT layer benchmark (IntLayerBench.java), the
matcher scaling benchmark, the recorded skyline-extraction runtimes, the query rates measured in
EXP-INT-004, and the storage measurements of EXP-SKY-014 into one machine-readable summary.

    $PY evaluation/tools/claim_closure/resources_summary.py --out evaluations/claim-closure-2026-09/resources \
        [--peak-working-set-mb name=value ...]

Every number carries its host; nothing is a target-hardware claim.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def col(path: Path, name: str) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        return np.asarray([float(r[name]) for r in csv.DictReader(f) if r.get(name, "") not in ("", None)])


def qt(x: np.ndarray, scale: float = 1.0) -> dict:
    x = x[np.isfinite(x)] * scale
    return {"n": int(x.size), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "mean": float(x.mean()), "max": float(x.max())} if x.size else {"n": 0}


def env_of(run: str) -> dict:
    m = json.loads((REPO / "runs" / run / "manifest.json").read_text(encoding="utf-8"))
    e = m["environment"]
    return {"host": e["hostname"], "cpu": e["cpu_model"], "jvm": e["jvm_version"], "heap_max_mb": e["heap_max_mb"],
            "downsample": m["estimator_config"].get("downsampleFactor"), "dataset": m["dataset_id"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "evaluations/claim-closure-2026-09/resources")
    ap.add_argument("--peak-working-set-mb", nargs="*", default=[])
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {"experiment": "EXP-INT-005", "machine_this_pass": {
        "host": "development laptop", "cpu": "11th Gen Intel Core i7-11390H @ 3.40 GHz, 4 cores / 8 threads",
        "ram_gb": 16, "os": "Windows 11 Home 10.0.26200", "jvm": "19.0.1", "python": "3.12.0",
        "is_target_hardware": False}}

    # --- continuous path: VO per frame -------------------------------------------------------------
    vo = {}
    for run, note in [("hkairport03-homography-rigid-conf-off-v1", "MARS-LVIG 2448x2048, downsample 2 (processed 1224x1024), 10 Hz, this laptop"),
                      ("amtown03-homography-rigid-conf-off-v1", "MARS-LVIG, as above, this laptop"),
                      ("hkairport01-b-homography-rigid-timing-baseline-v1", "MARS-LVIG, as above, this laptop (motion-timing run)"),
                      ("exp-int-003-ho1-mtn-fig8-vary-v1-vo-only", "UE5 nadir 1024x1024, downsample 1, 10 Hz, INT host (desktop, CPU model 165)"),
                      ("exp-int-003-ho1-mtn-trinity-vary-v1-vo-only", "UE5, as above, INT host"),
                      ("exp-int-001-interesting-r1-vary-v1-vo-only", "UE5, as above, INT host")]:
        p = REPO / "runs" / run / "frames.csv"
        if p.exists():
            vo[run] = {"note": note, "env": env_of(run), "process_time_ms": qt(col(p, "process_time_ns"), 1e-6)}
    res["vo_per_frame"] = vo
    motion = {}
    for seq in ("hkairport03", "amtown03"):
        p = REPO / f"evaluations/exp-conf-005/timing/{seq}_motion_timing.csv"
        if p.exists():
            motion[seq] = qt(col(p, "motion_time_ns"), 1e-6)
    res["vo_motion_stage_ms_this_laptop"] = motion
    # --- CONF overhead ----------------------------------------------------------------------------
    conf = {}
    for seq in ("hkairport03", "amtown03", "hkairport01-b", "amtown01-d"):
        p = REPO / "runs" / f"{seq}-homography-rigid-diagrefit-v1" / "diagnostic_refit.csv"
        if p.exists():
            conf[seq] = {"refit_ms": qt(col(p, "refit_time_ns"), 1e-6), "readout_us": qt(col(p, "readout_time_ns"), 1e-3)}
    res["conf_refit_overhead_this_laptop"] = conf
    # --- INT layer benchmark ------------------------------------------------------------------------
    bench = {}
    for p in sorted(glob.glob(str(out / "int_layer_bench_*.json"))):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        name = Path(p).stem.replace("int_layer_bench_", "")
        if name == "trial":
            continue
        bench[name] = {k: d[k] for k in ("frames", "memory_size_at_end", "wall_ns_last_rep", "per_frame_all", "per_frame_with_retrieval",
                                         "per_frame_without_retrieval", "jvm_peak_heap_used_bytes_sum_of_pools")}
        bench[name]["amortised_us_per_frame"] = d["wall_ns_last_rep"] / 1e3 / d["frames"]
        bench[name]["matcher_scaling"] = d["matcher_scaling"]
    res["int_layer_bench_this_laptop"] = bench
    # --- skyline extraction runtimes (recorded) ---------------------------------------------------
    ext = {}
    for key, p in [("poc_robust_dp_skyfinder_eval", "evaluations/ext-eval-poc-skyfinder-eval/metrics.json"),
                   ("poc_robust_dp_basaltweb", "evaluations/ext-eval-poc-basaltweb/metrics.json"),
                   ("prototype_basaltweb", "evaluations/ext-eval-prototype-basaltweb/metrics.json")]:
        d = json.loads((REPO / p).read_text(encoding="utf-8"))
        ext[key] = {"n_images": d["aggregates"]["n_images"], "runtime_s": d["aggregates"]["runtime_s"], "source": p,
                    "note": "EXP-SKY-002 extraction bench, Python, host not recorded in metrics (development laptop); real photographs"}
    ext["segformer_b0_cpu"] = {"seconds_per_frame_per_view": 0.96, "seconds_per_dual_capture": 1.92,
                               "source": "EXP-SKY-011 R9 (512 px, 4 CPU threads, development laptop)"}
    res["skyline_extraction_runtime"] = ext
    # --- query rates from EXP-INT-004 --------------------------------------------------------------
    int4 = json.loads((REPO / "evaluations/claim-closure-2026-09/int/results.json").read_text(encoding="utf-8"))
    rates = {}
    for label, tr in int4["tracks"].items():
        a = tr["arms"]["FROZEN"]
        rates[label] = {"queries": a["sidecar"]["queries"], "per_km": a["queries_per_km"], "per_min": a["queries_per_min"],
                        "frames": a["sidecar"]["frames"], "seconds": a["sidecar"]["seconds"], "references_at_end": a["sidecar"]["references_inserted"]}
    res["frozen_policy_query_rates"] = rates
    # --- storage (EXP-SKY-014 Q-D) -------------------------------------------------------------------
    comp = json.loads((REPO / "evaluations/claim-closure-2026-09/sky/compactness.json").read_text(encoding="utf-8"))
    res["storage"] = {"descriptor_bytes_dual_float64": comp["representation"]["descriptor_float64_bytes_dual"],
                      "view_png_median_bytes": comp["measured"]["view_image_png_bytes"]["median"],
                      "trusted_memories": comp["int_trusted_memories"]}
    # --- peak working set (optional CLI input) --------------------------------------------------------
    pws = {}
    for spec in args.peak_working_set_mb:
        k, _, v = spec.partition("=")
        pws[k] = float(v)
    res["replay_jvm_peak_working_set_mb_this_laptop"] = pws
    # --- amortised cost model --------------------------------------------------------------------------
    model = {}
    for label, r in rates.items():
        b = bench.get(label.replace("ho1-", "ho1-") if False else {
            "ho1-mtn-trinity": "ho1-mtn-trinity-vary-v1", "ho1-mtn-fig8": "ho1-mtn-fig8-vary-v1", "interesting-r1-vary": "interesting-r1-vary-v1"}.get(label, ""))
        q_per_s = r["queries"] / max(1e-9, r["seconds"])
        model[label] = {"queries_per_s": q_per_s,
                        "extraction_dp_cpu_s_per_flight_s_both_views": q_per_s * 2 * ext["poc_robust_dp_skyfinder_eval"]["runtime_s"]["median"],
                        "extraction_segformer_cpu_s_per_flight_s_both_views": q_per_s * ext["segformer_b0_cpu"]["seconds_per_dual_capture"],
                        "int_layer_amortised_us_per_frame": b["amortised_us_per_frame"] if b else None}
    res["amortised_model"] = model
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("vo_motion_stage_ms_this_laptop", "amortised_model")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
