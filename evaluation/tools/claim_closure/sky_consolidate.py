"""EXP-SKY-014 Q-A / Q-C: consolidate the committed one-view-vs-two-view and appearance evidence
under the adopted matcher from EXP-SKY-011 / -012 / -013 / -008 metrics (no re-run, no re-scoring).

    python evaluation/tools/claim_closure/sky_consolidate.py --out evaluations/claim-closure-2026-09/sky
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "evaluations/claim-closure-2026-09/sky")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    m13 = json.loads((REPO / "evaluations/sky-matcher-resolution/metrics.json").read_text(encoding="utf-8"))
    m11 = json.loads((REPO / "evaluations/sky-dual/metrics.json").read_text(encoding="utf-8"))
    cons: dict = {"sources": ["evaluations/sky-matcher-resolution/metrics.json (EXP-SKY-013)",
                              "evaluations/sky-dual/metrics.json (EXP-SKY-011)",
                              "EXP-SKY-012 R3 (table)", "EXP-SKY-008 Experiment 3 (table)"]}
    lines = ["# EXP-SKY-014 Q-A / Q-C — consolidated one-view vs two-view and appearance evidence (generated)", ""]

    # --- hard city, EXP-SKY-013 (C0 vs the C1 bounds), dense-LSO memory, both sources ---------------
    lines.append("## Q-A.1 Hard-city absent-place negatives (EXP-SKY-013 R7; city memory, leave-site-out; every accept is false)\n")
    lines.append("| source | gate | matcher | view A only | view B only | weakest-view | strict | n queries |")
    lines.append("|---|---|---|---|---|---|---|---|")
    hc = {}
    for source in ("sim_exact", "segformer"):
        blk = m13["hard_city"][source]["dense_lso"]
        for gate_name, gate in blk.items():
            for matcher in ("C0", "C1-4", "C1-32"):
                g = gate[matcher]
                af = g["accepted_false"]
                hc[f"{source}/{gate_name}/{matcher}"] = {**af, "n": g["n_queries"]}
                lines.append(f"| {source} | {gate_name} | {matcher} | {af['north']} | {af['west']} | {af['weakest']} | {af['strict']} | {g['n_queries']} |")
    cons["hard_city_dense_lso"] = hc

    # --- terrain, dense online memory (EXP-SKY-013 R5): selected-reference distance beyond τ ----------
    lines.append("\n## Q-A.2 Dense online memory, selected-reference distance (EXP-SKY-013 R5; fraction of queries whose selected reference lies beyond τ, and p95 selected distance in m)\n")
    lines.append("| level / source | matcher | view A only | view B only | weakest-view | strict | n |")
    lines.append("|---|---|---|---|---|---|---|")
    ter = {}
    for key in ("village/sim_exact", "village/segformer", "mountains/sim_exact", "mountains/segformer", "city/sim_exact", "city/segformer"):
        d = m13["terrain"][key]["dense_online"]
        for matcher in ("C0", "C1-32"):
            g = d[matcher]
            cells = []
            for rule in ("north", "west", "weakest", "strict"):
                r = g[rule]
                cells.append(f"{r['frac_beyond_tau']:.3f} / p95 {r['p95']:.1f}")
                ter[f"{key}/{matcher}/{rule}"] = {"frac_beyond_tau": r["frac_beyond_tau"], "p95_m": r["p95"], "max_m": r["max"], "n": r["n"]}
            lines.append(f"| {key} | {matcher} | " + " | ".join(cells) + f" | {g['north']['n']} |")
    cons["terrain_dense_online"] = ter

    # --- EXP-SKY-011 ablation at the DEV-frozen gates (C1-32; sparse anchor memory) -------------------
    lines.append("\n## Q-A.3 Sparse anchor memory at DEV-frozen gates (EXP-SKY-011 R2, matcher C1-32): coverage / accepted-false / OOC accepted / LSO accepted\n")
    lines.append("| level / source | rule | coverage | accepted false | OOC accepted | LSO accepted |")
    lines.append("|---|---|---|---|---|---|")
    ab = {}
    for key, v in m11["ablation"].items():
        lvl, src, rule = key.split("/")
        ab[key] = {k: v.get(k) for k in ("coverage", "n_accepted_false", "ooc_n_accepted", "ooc_n", "lso_n_accepted", "lso_n", "theta_s", "theta_m", "region_top1")}
        lines.append(f"| {lvl}/{src} | {rule} | {v.get('coverage', 0):.2f} | {v.get('n_accepted_false')} | {v.get('ooc_n_accepted')}/{v.get('ooc_n')} | {v.get('lso_n_accepted')}/{v.get('lso_n')} |")
    cons["exp_sky_011_ablation"] = ab

    # --- EXP-SKY-012 R3 (transcribed, the record's table) ---------------------------------------------
    cons["exp_sky_012_R3"] = {
        "population": "87 horizontal HardSwipe frames, city, frozen gate 0.90/0.15 (frame) / DEV gate",
        "north_false_sparse_segformer": 42, "north_false_sparse_sim_exact": 8, "west_false": 0,
        "strict_false": 0, "weakest_false": 0, "north_false_eliminated_by_west_refusal": 82, "by_west_disagreement": 0,
        "temporal_confirmation_north_k3_false": "30/87 (EXP-SKY-012 R5)"}
    lines.append("\n## Q-A.4 EXP-SKY-012 R3 (transcribed): of 82 North false accepts at the frozen gate, 82 were eliminated by West refusal, 0 by disagreement; strict and weakest-view accepted 0; North-only k=3 temporal confirmation still 30/87 false.")

    # --- appearance, EXP-SKY-008 Experiment 3 (transcribed) -------------------------------------------
    cons["exp_sky_008_experiment_3"] = {
        "design": "67 anchors x complete 9-condition TIME{DAY,DAWN,DUSK} x CLOUD{CLEAR,CLOUDY,VERY_CLOUDY} matrix; exact-pose retrieval, databases 10/30/27",
        "gt_c0_top1": {"village": 1.00, "mountains": 1.00, "city": 1.00},
        "segformer_c0_top1": {"village": 0.938, "mountains": 1.00, "city": 0.880},
        "city_segformer_per_condition_range": "85–89 % in every one of the 8 query conditions, zero strict confident-false",
        "supported_conditions": "DAY / DAWN / DUSK x CLEAR / CLOUDY / VERY_CLOUDY (rendered)",
        "not_supported": "night, rain, real-world appearance (T3/T4), arbitrary yaw"}
    lines.append("\n## Q-C EXP-SKY-008 Experiment 3 (transcribed): exact-pose top-1 under the 9-condition matrix — GT+C0 100 % on all three levels; SegFormer+C0 93.8 / 100 / 88.0 % (village / mountains / city), the city misses condition-independent (85–89 % in every condition). Supported: DAY/DAWN/DUSK × CLEAR/CLOUDY/VERY_CLOUDY rendered conditions. Not supported: night, rain, real appearance.")
    (out / "consolidated.json").write_text(json.dumps(cons, indent=1), encoding="utf-8")
    (out / "consolidated_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
