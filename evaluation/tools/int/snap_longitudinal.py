"""Longitudinal snap safety: what an accepted re-anchor did to the trajectory *after* the frame it fired on.

`evaluate_int_arms.py` reports the instantaneous ``Δe_0 = e_after − e_before``. This tool follows
each accepted drift correction forward and explains, numerically, how a snap with ``Δe_0 < 0`` can
still worsen the trajectory. It relies on one structural fact of the layer — the alignment is a
**translation** and the VO increments are identical in both arms (asserted here from the two
``alignment_frames.csv`` files) — which makes every post-snap quantity exact, not modelled:

    e_iso(f)  = d(f) + a        the error after the snap, with no later snap (isolated form)
    e_nok(f)  = d(f) + b        the error had this snap not fired (counterfactual)
    d(f)      = [GT(f) − GT(q)] − [VO(f) − VO(q)]      the VO's incremental error since the snap frame q
    a         = GT(q) − p_ref   = u + r                 the post-snap error vector: the true query-to-reference
                                                        displacement u plus the stored reference's own pose error r
    b         = GT(q) − p_before                        the pre-snap error vector
    c         = p_ref − p_before = b − a                the correction applied

so that  |e_iso(f)|² − |e_nok(f)|² = |a|² − |b|² − 2 d(f)·c  exactly. The first term is the
instantaneous gain (negative when Δe_0 < 0); the second grows with the incremental drift and is
negative — i.e. harmful — whenever the drift that follows points *against* the correction. That is
the "compensating drift" hypothesis in one line: the snap was harmful later if the pre-snap offset
b happened to cancel the drift that followed (|d + b| < |d|) while the post-snap offset a does not.
Both are reported per horizon, with the integrated difference ΔA_H = Σ(|e_iso| − |e_nok|)·dt.

Two horizons families: **isolated** (this snap only, to each horizon and to the end of the segment,
ignoring later snaps — computable exactly from the VO_ONLY track; the classification uses only
these) and **actual** (the INT arm as run: the fixed horizons are *not* censored, so a later snap
inside the horizon shows in them; ``to_next_snap`` is the censored view). The comparison against the
VO_ONLY arm (the milestone's baseline) is reported beside the counterfactual "same arm without this
snap"; for the first snap of a run the two coincide.

The tool **refuses** to run when the precondition fails: the two arms must share every local
increment (max difference below 1e-6 m) and the ``--vo-only`` arm must never have re-anchored
(its persistent position must equal its local position up to one constant), because every
counterfactual is computed from that arm's track.

Classification per snap (pre-declared): ``A`` immediately and persistently beneficial (Δe_0 < 0,
every horizon Δe_H ≤ 0, ΔA to next snap/end ≤ 0); ``B`` immediately beneficial, later harmful
(Δe_0 < 0 but some Δe_H > 0 or ΔA > 0); ``C`` immediately harmful (Δe_0 > 0); ``D`` neutral
(|Δe_0| < 0.5 m and |ΔA| < 0.5 m·s per second of horizon). Hard-loss recoveries have no ``b``;
they are listed with ``a``, ``u``, ``r`` and the post-snap error only.

Run from the repository root::

    python evaluation/tools/int/snap_longitudinal.py --dataset datasets/<id> --vo-only runs/<vo-only run> \\
        --int runs/<int run> --out <dir> [--horizons 5,10,20,30]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import evaluate_int_arms as ev  # noqa: E402


def arrays(rows: list[dict]):
    fr = np.array([int(r["frame_index"]) for r in rows])
    t = np.array([float(r["timestamp_s"]) for r in rows])
    loc = np.array([[ev._f(r["local_east_m"]), ev._f(r["local_north_m"])] for r in rows])
    glob = np.array([[ev._f(r["global_east_m"]), ev._f(r["global_north_m"])] for r in rows])
    valid = np.array([r["global_position_valid"].strip().lower() == "true" for r in rows])
    seg = np.array([int(r["segment_id"]) for r in rows])
    return fr, t, loc, glob, valid, seg


def norm(v):
    return float(np.hypot(v[0], v[1]))


def analyse(dataset: Path, vo_run: Path, int_run: Path, out: Path, horizons: list[float], neutral_m: float) -> dict:
    gt = ev.load_gt(dataset)
    vo_rows = ev.read_csv(vo_run / "alignment_frames.csv")
    in_rows = ev.read_csv(int_run / "alignment_frames.csv")
    events = ev.read_csv(int_run / "alignment_events.csv")
    fr, t, loc_vo, g_vo, v_vo, seg_vo = arrays(vo_rows)
    fr2, _, loc_in, g_in, v_in, seg_in = arrays(in_rows)
    if not np.array_equal(fr, fr2):
        raise SystemExit("the two arms do not cover the same frames")
    both_local = np.isfinite(loc_vo).all(axis=1) & np.isfinite(loc_in).all(axis=1)
    max_local_diff = float(np.max(np.abs(loc_vo[both_local] - loc_in[both_local]))) if both_local.any() else float("nan")
    same_vo = both_local.any() and max_local_diff < 1e-6
    if not same_vo:
        raise SystemExit(f"the two arms do not share the VO's local increments (max |Δlocal| = {max_local_diff} m); "
                         "every counterfactual here rests on that — refused")
    # The --vo-only arm must be an identity-alignment track: global − local constant on its valid frames
    # (an INT run passed by mistake would pass the local check and break d(f) at its own snaps).
    off = (g_vo - loc_vo)[v_vo & np.isfinite(loc_vo).all(axis=1)]
    if off.size and float(np.max(np.abs(off - off[0]))) > 1e-6:
        raise SystemExit("the --vo-only arm's persistent position is not its local position up to one constant "
                         "(it re-anchored): pass the VO_ONLY run, not an INT run — refused")
    f0 = int(fr[0]); _, e0, n0 = gt[f0]
    gt_xy = np.array([[gt[int(f)][1] - e0, gt[int(f)][2] - n0] for f in fr])
    idx = {int(f): i for i, f in enumerate(fr)}
    ref_frames = ev.reference_frames(events)
    dt = float(np.median(np.diff(t)))
    reanchors = [r for r in events if r["kind"] == "reanchor"]
    snap_frames = [int(r["frame_index"]) for r in reanchors]
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for k, r in enumerate(reanchors):
        qf = int(r["frame_index"]); qi = idx[qf]
        ref_id = int(r["reference_id"])
        ref_frame = ref_frames.get(ref_id)
        p_ref = np.array([ev._f(r["global_after_east_m"]), ev._f(r["global_after_north_m"])])
        p_before = np.array([ev._f(r["global_before_east_m"]), ev._f(r["global_before_north_m"])])
        drift_case = r["recovery_case"] == "drift_correction" and np.isfinite(p_before).all()
        GTq = gt_xy[qi]
        a = GTq - p_ref
        u = GTq - gt_xy[idx[ref_frame]] if ref_frame is not None else np.array([np.nan, np.nan])
        rr = gt_xy[idx[ref_frame]] - p_ref if ref_frame is not None else np.array([np.nan, np.nan])   # stored-reference pose error
        b = GTq - p_before if drift_case else np.array([np.nan, np.nan])
        b_vo = GTq - g_vo[qi]
        c = p_ref - p_before if drift_case else np.array([np.nan, np.nan])
        # drift accumulated by the VO between the reference's insertion and the query (the quantity a snap undoes)
        D_gq = (b_vo - (gt_xy[idx[ref_frame]] - g_vo[idx[ref_frame]])) if ref_frame is not None else np.array([np.nan, np.nan])
        next_snap = min([s for s in snap_frames if s > qf], default=None)
        seg_end = qi
        while seg_end + 1 < len(fr) and seg_vo[seg_end + 1] == seg_vo[qi] and v_vo[seg_end + 1]:
            seg_end += 1
        hz = {}
        series = []
        d_all = (gt_xy - GTq) - (g_vo - g_vo[qi])          # incremental VO error since q, every frame
        for i in range(qi + 1, seg_end + 1):
            d = d_all[i]
            e_iso = d + a; e_nok = d + b if drift_case else np.array([np.nan, np.nan]); e_vo = gt_xy[i] - g_vo[i]
            e_act = gt_xy[i] - g_in[i] if v_in[i] else np.array([np.nan, np.nan])
            series.append({"frame": int(fr[i]), "time_s": float(t[i]), "since_snap_s": float(t[i] - t[qi]),
                           "e_iso_m": norm(e_iso), "e_nok_m": norm(e_nok) if drift_case else float("nan"),
                           "e_vo_only_m": norm(e_vo), "e_int_actual_m": norm(e_act),
                           "d_m": norm(d), "d_dot_c": float(np.dot(d, c)) if drift_case else float("nan"),
                           "d_dot_a": float(np.dot(d, a)), "d_dot_b": float(np.dot(d, b)) if drift_case else float("nan"),
                           "identity_residual": (float(norm(e_iso) ** 2 - norm(e_nok) ** 2 - (norm(a) ** 2 - norm(b) ** 2 - 2 * np.dot(d, c)))
                                                 if drift_case else float("nan")),
                           "censored_by_next_snap": next_snap is not None and fr[i] >= next_snap})
        with (out / f"snap_{qf}_series.csv").open("w", newline="") as f:
            if series:
                w = csv.DictWriter(f, fieldnames=list(series[0].keys())); w.writeheader(); w.writerows(series)

        def horizon(H_s, censor_next: bool):
            sel = [s for s in series if s["since_snap_s"] <= H_s + 1e-9 and not (censor_next and s["censored_by_next_snap"])]
            if not sel:
                return None
            last = sel[-1]
            i_last = idx[last["frame"]]
            d = d_all[i_last]
            res = {"reached_s": last["since_snap_s"], "n_frames": len(sel), "e_iso_m": last["e_iso_m"],
                   "e_vo_only_m": last["e_vo_only_m"], "e_int_actual_m": last["e_int_actual_m"],
                   "d_m": last["d_m"], "d_dot_a": last["d_dot_a"],
                   "dA_iso_vs_vo_only_m_s": float(sum(s["e_iso_m"] - s["e_vo_only_m"] for s in sel) * dt),
                   "dA_actual_vs_vo_only_m_s": float(np.nansum([s["e_int_actual_m"] - s["e_vo_only_m"] for s in sel]) * dt)}
            if drift_case:
                res.update({"e_nok_m": last["e_nok_m"], "delta_e_iso_m": last["e_iso_m"] - last["e_nok_m"],
                            "delta_e_actual_vs_nok_m": last["e_int_actual_m"] - last["e_nok_m"],
                            "dA_iso_vs_nok_m_s": float(sum(s["e_iso_m"] - s["e_nok_m"] for s in sel) * dt),
                            "d_dot_c": last["d_dot_c"], "d_dot_b": last["d_dot_b"],
                            "instantaneous_term_a2_minus_b2": norm(a) ** 2 - norm(b) ** 2,
                            "drift_term_minus_2_d_dot_c": -2.0 * last["d_dot_c"],
                            "compensation_index_of_b": (norm(d) - norm(d + b)) / norm(b) if norm(b) > 1e-9 else float("nan"),
                            "compensation_index_of_a": (norm(d) - norm(d + a)) / norm(a) if norm(a) > 1e-9 else float("nan"),
                            "cos_b_d": float(np.dot(b, d) / (norm(b) * norm(d))) if norm(b) * norm(d) > 1e-9 else float("nan"),
                            "cos_c_d": float(np.dot(c, d) / (norm(c) * norm(d))) if norm(c) * norm(d) > 1e-9 else float("nan")})
            return res

        for H in horizons:
            hz[f"{H:g}s"] = horizon(H, censor_next=False)
        hz["to_next_snap"] = horizon(1e12, censor_next=True) if next_snap is not None else None
        hz["to_end_isolated"] = horizon(1e12, censor_next=False)
        e_before = norm(b) if drift_case else float("nan")
        e_after = norm(a)
        delta0 = e_after - e_before if drift_case else float("nan")
        # classification
        cls = None
        if drift_case:
            later = [hz[f"{H:g}s"]["delta_e_iso_m"] for H in horizons if hz.get(f"{H:g}s")]
            tail = hz["to_next_snap"] or hz["to_end_isolated"]
            dA_tail = tail["dA_iso_vs_nok_m_s"] if tail else 0.0
            span = tail["reached_s"] if tail else 1.0
            if abs(delta0) < neutral_m and abs(dA_tail) < neutral_m * max(span, 1e-9):
                cls = "D_neutral"
            elif delta0 > 0:
                cls = "C_immediately_harmful"
            elif any(x > 0 for x in later) or dA_tail > 0:
                cls = "B_beneficial_then_harmful"
            else:
                cls = "A_persistently_beneficial"
        rows_out.append({
            "snap_index": k, "frame": qf, "time_s": float(t[qi]), "recovery_case": r["recovery_case"],
            "reference_id": ref_id, "reference_frame": ref_frame,
            "reference_age_s": ev._f(r["reference_age_s"]), "next_snap_frame": next_snap,
            "gt_query_to_reference_m": norm(u), "u_vec_m": u.tolist(),
            "stored_reference_pose_error_m": norm(rr), "r_vec_m": rr.tolist(),
            "vo_drift_since_reference_insertion_m": norm(D_gq), "D_gq_vec_m": D_gq.tolist(),
            "e_before_m": e_before, "b_vec_m": b.tolist(), "e_before_vo_only_m": norm(b_vo), "b_vo_only_vec_m": b_vo.tolist(),
            "e_after_m": e_after, "a_vec_m": a.tolist(),
            "correction_m": norm(c) if drift_case else float("nan"), "c_vec_m": c.tolist(),
            "position_jump_m": ev._f(r["position_jump_m"]),
            "delta_e_0_m": delta0,
            "cos_a_u": float(np.dot(a, u) / (norm(a) * norm(u))) if norm(a) * norm(u) > 1e-9 else float("nan"),
            "cos_b_c": float(np.dot(b, c) / (norm(b) * norm(c))) if drift_case and norm(b) * norm(c) > 1e-9 else float("nan"),
            "horizons": hz, "classification": cls,
            "identity_max_abs_residual": float(np.nanmax([abs(s["identity_residual"]) for s in series])) if series and drift_case else None,
        })
    summary = {
        "dataset": str(dataset), "vo_only_run": str(vo_run), "int_run": str(int_run),
        "same_vo_increments": {"max_abs_local_position_difference_m": max_local_diff, "identical": same_vo},
        "horizons_s": horizons, "neutral_threshold_m": neutral_m, "frame_dt_s": dt,
        "n_snaps": len(rows_out),
        "classification_counts": {c: sum(1 for r in rows_out if r["classification"] == c)
                                  for c in ("A_persistently_beneficial", "B_beneficial_then_harmful", "C_immediately_harmful", "D_neutral")},
        "snaps": rows_out,
    }
    (out / "snap_longitudinal.json").write_text(json.dumps(summary, indent=2, default=_default) + "\n")
    with (out / "snap_longitudinal.csv").open("w", newline="") as f:
        cols = ["snap_index", "frame", "time_s", "recovery_case", "reference_id", "reference_frame", "reference_age_s",
                "gt_query_to_reference_m", "stored_reference_pose_error_m", "vo_drift_since_reference_insertion_m",
                "e_before_m", "e_before_vo_only_m", "e_after_m", "correction_m", "delta_e_0_m"]
        hcols = []
        for H in horizons:
            hcols += [f"delta_e_{H:g}s_m", f"dA_{H:g}s_vs_nok_m_s", f"e_iso_{H:g}s_m", f"e_vo_{H:g}s_m", f"d_dot_c_{H:g}s"]
        hcols += ["delta_e_to_next_or_end_m", "dA_to_next_or_end_vs_nok_m_s", "dA_actual_to_next_or_end_vs_vo_m_s", "classification"]
        w = csv.writer(f); w.writerow(cols + hcols)
        for r in rows_out:
            vals = [r[c] for c in cols]
            for H in horizons:
                h = r["horizons"].get(f"{H:g}s")
                vals += [h.get("delta_e_iso_m") if h else None, h.get("dA_iso_vs_nok_m_s") if h else None,
                         h.get("e_iso_m") if h else None, h.get("e_vo_only_m") if h else None, h.get("d_dot_c") if h else None]
            tail = r["horizons"]["to_next_snap"] or r["horizons"]["to_end_isolated"]
            vals += [tail.get("delta_e_iso_m") if tail else None, tail.get("dA_iso_vs_nok_m_s") if tail else None,
                     tail.get("dA_actual_vs_vo_only_m_s") if tail else None, r["classification"]]
            w.writerow(["" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else v) for v in vals])
    return summary


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return ev._json_default(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--vo-only", required=True, type=Path)
    ap.add_argument("--int", dest="int_run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--horizons", default="5,10,20,30")
    ap.add_argument("--neutral-m", type=float, default=0.5)
    a = ap.parse_args(argv)
    horizons = [float(x) for x in a.horizons.split(",") if x.strip()]
    s = analyse(a.dataset, a.vo_only, a.int_run, a.out, horizons, a.neutral_m)
    print(f"same VO increments: {s['same_vo_increments']}; {s['n_snaps']} snaps; classes {s['classification_counts']}")
    for r in s["snaps"]:
        hz = r["horizons"]
        line = (f"  snap@{r['frame']} ({r['time_s']:.1f}s) {r['recovery_case']} ref {r['reference_id']}@{r['reference_frame']} "
                f"age {r['reference_age_s']:.0f}s | u={r['gt_query_to_reference_m']:.2f} r={r['stored_reference_pose_error_m']:.2f} "
                f"D_gq={r['vo_drift_since_reference_insertion_m']:.2f} | e_before={r['e_before_m']:.2f} e_after={r['e_after_m']:.2f} "
                f"c={r['correction_m']:.2f} de0={r['delta_e_0_m']:+.2f} | ")
        for H in s["horizons_s"]:
            h = hz.get(f"{H:g}s")
            line += f"{H:g}s:{'' if not h or h.get('delta_e_iso_m') is None else format(h['delta_e_iso_m'], '+.2f')} "
        tail = hz["to_next_snap"] or hz["to_end_isolated"]
        if tail:
            line += (f"| tail {tail['reached_s']:.0f}s de={tail.get('delta_e_iso_m', float('nan')):+.2f} "
                     f"dA={tail.get('dA_iso_vs_nok_m_s', float('nan')):+.1f} m.s -> {r['classification']}")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
