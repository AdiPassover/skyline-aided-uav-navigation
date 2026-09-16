"""Emit `EXP-VO-012`'s Results tables as markdown, straight from `results.json`.

The closure audit found three transcription slips in hand-written tables whose own generator
produced the correct values (`VO_CLAIM_MATRIX.md` §3.16). This module exists so that this record's
tables are **generated, not typed**. `verify_record.py` then re-checks the committed markdown
against the same artifact, so a later hand-edit that drifts is caught.

    PY=.venv/Scripts/python.exe
    $PY evaluation/tools/exp_vo_012/emit_tables.py > /tmp/tables.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def t_regimes(res: dict) -> str:
    out = ["| regime | sequence · model | arm | ref-init endpoint (m) | **% of path** | ref-init ATE (m) | SE(2) endpoint (m) | path-length ratio | Sim(2) norm. ATE | Sim(2) fitted scale |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(res["regimes"], key=lambda k: (res["regimes"][k]["regime"], k)):
        r = res["regimes"][key]
        seq, model = key.split("/")
        for arm in ("fixed", "baro", "oracle"):
            a = r["arms"][arm]
            bold = "**" if arm == "baro" else ""
            out.append(
                f"| {r['regime']} | `{seq}` · {model} | {bold}{arm}{bold} | "
                f"{a['ref_init']['endpoint_error_m']:.3f} | "
                f"{bold}{a['ref_init']['endpoint_error_pct_of_path']:.3f} %{bold} | "
                f"{a['ref_init']['ate_rmse_m']:.3f} | {a['se2']['endpoint_error_m']:.3f} | "
                f"{a['path_length_ratio']:.5f} | "
                f"{100 * a['sim2']['ate_rmse_normalised']:.4f} % | "
                f"{a['sim2']['alignment_scale']:.5f} |")
    return "\n".join(out)


def t_regime_geometry(res: dict) -> str:
    out = ["| regime | sequence | frames | restarts | terrain (m) | true AGL (m) | barometer reports (m) |",
           "|---|---|---|---|---|---|---|"]
    seen = set()
    for key in sorted(res["regimes"], key=lambda k: (res["regimes"][k]["regime"], k)):
        r = res["regimes"][key]
        seq = key.split("/")[0]
        if seq in seen:
            continue
        seen.add(seq)
        out.append(f"| {r['regime']} | `{seq}` | {r['n_frames']} | {r['n_restarts']} | "
                   f"{r['terrain_range_m'][0]:.1f} → {r['terrain_range_m'][1]:.1f} | "
                   f"{r['agl_range_m'][0]:.1f} → {r['agl_range_m'][1]:.1f} | "
                   f"{r['baro_range_m'][0]:.1f} → {r['baro_range_m'][1]:.1f} |")
    return "\n".join(out)


def t_noise(res: dict) -> str:
    out = ["| σ (m) | endpoint, mean of 20 seeds (% path) | **spread across seeds, sd** | H4 closed form `(σ/h)/√n` | measured sd ÷ prediction |",
           "|---|---|---|---|---|"]
    for key, rows in res["r3_noise"].items():
        out.append(f"| *{key}* | | | | |")
        for r in rows:
            ratio = (r["endpoint_pct_sd"] / r["h4_prediction_pct"]
                     if r["h4_prediction_pct"] > 0 else float("nan"))
            out.append(f"| {r['sigma_m']:.2f} | {r['endpoint_pct_mean']:.5f} | "
                       f"**{r['endpoint_pct_sd']:.5f}** | {r['h4_prediction_pct']:.5f} | "
                       f"{ratio:.2f} |" if r["h4_prediction_pct"] > 0 else
                       f"| {r['sigma_m']:.2f} | {r['endpoint_pct_mean']:.5f} | "
                       f"**{r['endpoint_pct_sd']:.5f}** | — | — |")
    return "\n".join(out)


def t_drift(res: dict) -> str:
    out = ["| total drift D (m) | measured path-length ratio | H5 closed form `1 + D/2h` | error term, measured ÷ predicted | endpoint (% path) |",
           "|---|---|---|---|---|"]
    for key, rows in res["r4_drift"].items():
        out.append(f"| *{key}* | | | | |")
        for r in rows:
            m = r["path_length_ratio"] - 1.0
            p = r["h5_prediction_ratio"] - 1.0
            ratio = f"{m / p:.3f}" if abs(p) > 1e-9 else "—"
            out.append(f"| {r['drift_m']:+.2f} | {r['path_length_ratio']:.5f} | "
                       f"{r['h5_prediction_ratio']:.5f} | {ratio} | {r['endpoint_pct']:.3f} |")
    return "\n".join(out)


def t_rate(res: dict) -> str:
    out = ["| rate (Hz) | policy | causal | mean height error (m) | H6 discrete form `v(T−dt)/2` | endpoint (% path) |",
           "|---|---|---|---|---|---|"]
    for key, rows in res["r5_rate"].items():
        out.append(f"| *{key}* | | | | | |")
        for r in sorted([x for x in rows if x["latency_s"] == 0.0],
                        key=lambda x: (-x["rate_hz"], x["policy"])):
            out.append(f"| {r['rate_hz']:g} | {r['policy']} | {r['causal']} | "
                       f"{r['mean_height_error_m']:+.4f} | | {r['endpoint_pct']:.4f} |")
        out.append(f"| *latency at 10 Hz, causal ZOH* | | | | | |")
        for r in sorted([x for x in rows if x["latency_s"] > 0], key=lambda x: x["latency_s"]):
            out.append(f"| 10 (lat {r['latency_s']:g} s) | zoh | True | "
                       f"{r['mean_height_error_m']:+.4f} | | {r['endpoint_pct']:.4f} |")
    return "\n".join(out)


def t_dropout(res: dict) -> str:
    out = ["| dropout (s) | endpoint (% path), mean over 3 start positions | spread | degraded frames flagged (τ = 2 s) |",
           "|---|---|---|---|"]
    for key, rows in res["r6_dropout"].items():
        out.append(f"| *{key}* | | | |")
        by = {}
        for r in rows:
            if r["tau_stale_s"] != 2.0:
                continue
            by.setdefault(r["duration_s"], []).append(r)
        for d in sorted(by):
            e = [x["endpoint_pct"] for x in by[d]]
            n = [x["n_degraded_frames"] for x in by[d]]
            out.append(f"| {d:g} | {np.mean(e):.4f} | {np.ptp(e):.4f} | {int(np.mean(n))} |")
    return "\n".join(out)


def t_h0(res: dict) -> str:
    out = ["| η (%) | measured horizontal scale error (%) | H8 closed form `η·h₀/h_AGL` (%) | measured ÷ predicted |",
           "|---|---|---|---|"]
    for key, rows in res["phase6_h0"].items():
        out.append(f"| *{key}* | | | |")
        for r in rows:
            m = 100 * (r["path_length_ratio"] - 1.0)
            p = 100 * (r["h8_prediction_ratio"] - 1.0)
            ratio = f"{m / p:.4f}" if abs(p) > 1e-9 else "—"
            out.append(f"| {r['eta_pct']:+.0f} | {m:+.4f} | {p:+.4f} | {ratio} |")
    return "\n".join(out)


def t_diagnostics(res: dict) -> str:
    out = ["| sequence · model | accumulated visual scale (end) | true AGL ratio (end) | barometer-implied ratio (end) | **d log(visual) / d log(AGL)** |",
           "|---|---|---|---|---|"]
    for key in sorted(res["phase7_diagnostics"]):
        d = res["phase7_diagnostics"][key]
        s = d["slope_dlogvisual_dlogagl"]
        out.append(f"| `{key}` | {d['visual_accum_end']:.4f} | {d['agl_ratio_end']:.4f} | "
                   f"{d['baro_implied_ratio_end']:.4f} | "
                   f"**{'—' if s != s else f'{s:+.4f}'}** |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(REPO / "evaluations/exp-vo-012/results.json"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    # The tables are UTF-8 (arrows, sigmas, en dashes); Windows' console default is cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    sections = {"geometry": t_regime_geometry, "regimes": t_regimes, "noise": t_noise,
                "drift": t_drift, "rate": t_rate, "dropout": t_dropout, "h0": t_h0,
                "diagnostics": t_diagnostics}
    for name, fn in sections.items():
        if a.only and name not in a.only:
            continue
        print(f"\n<!-- {name} -->\n")
        print(fn(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
