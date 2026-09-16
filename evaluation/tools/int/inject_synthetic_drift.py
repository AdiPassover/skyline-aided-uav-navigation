"""Derive a VO_ONLY run with a **declared synthetic translation drift** added to its recorded track, for
the controlled mechanism study of `EXP-INT-002` §18: real imagery, real skyline captures, real
retrieval and real reference memory — only the VO's position error is controlled.

The derived directory is a complete VO_ONLY run for `RelocalizationReplayApp` (`metric_track.csv`
+ `alignment_frames.csv`, kept mutually consistent: the replay refuses them otherwise) and for
`evaluate_int_arms.py` (the drifted track *is* the VO_ONLY arm of the study — the comparison is
"drifted VO" versus "INT on the same drifted VO"). Nothing else changes: the segment structure,
the validity flags, the heading channel, the timestamps and the skyline pairing are the recording's.

Drift models (all in metres, added to the segment-relative and persistent positions):

* ``linear``      d(t) = rate · (t − t0) · (sin θ, cos θ)            — a constant-velocity bias
* ``path``        d(s) = frac · s(t) · (sin θ, cos θ)                — bias proportional to path flown
* ``scale``       d(t) = (k − 1) · p_local(t)                        — a scale error (position-dependent)
* ``step``        d(t) = step · (sin θ, cos θ) for frames ≥ f_step   — one mis-estimated increment
* ``random_walk`` d(t) = Σ N(0, σ² · Δt) per axis, fixed seed        — a diffusive error

θ is a compass bearing (degrees CW from North). The manifest records the model, its parameters,
the source run and the SHA-256 of the source files; the run id carries ``-synthetic-drift-`` and the
manifest an explicit evidence note. **A derived run is a controlled diagnostic input and is never a
navigation result**.

    python evaluation/tools/int/inject_synthetic_drift.py --vo-run runs/<vo-only run> --out runs/<derived> \\
        --model linear --rate 0.1 --bearing-deg 45
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read_rows(p: Path):
    with p.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def write_rows(p: Path, rows, fields):
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fmt(x: float) -> str:
    return repr(float(x))


def drift_profile(model: str, t: np.ndarray, s: np.ndarray, local: np.ndarray, a) -> np.ndarray:
    u = np.array([math.sin(math.radians(a.bearing_deg)), math.cos(math.radians(a.bearing_deg))])
    if model == "linear":
        return np.outer(a.rate * (t - t[0]), u)
    if model == "path":
        return np.outer(a.frac * s, u)
    if model == "scale":
        return (a.scale - 1.0) * local
    if model == "step":
        d = np.zeros((len(t), 2))
        d[np.arange(len(t)) >= a.step_frame] = a.step * u
        return d
    if model == "random_walk":
        rng = np.random.default_rng(a.seed)
        dt = np.diff(t, prepend=t[0])
        inc = rng.normal(0.0, 1.0, size=(len(t), 2)) * np.sqrt(np.maximum(dt, 0.0))[:, None] * a.sigma
        inc[0] = 0.0
        return np.cumsum(inc, axis=0)
    raise SystemExit(f"unknown model {model}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vo-run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", required=True, choices=["linear", "path", "scale", "step", "random_walk"])
    ap.add_argument("--rate", type=float, default=0.0, help="linear: m/s")
    ap.add_argument("--frac", type=float, default=0.0, help="path: metres of drift per metre flown")
    ap.add_argument("--scale", type=float, default=1.0, help="scale: multiplicative factor on the local position")
    ap.add_argument("--step", type=float, default=0.0, help="step: metres")
    ap.add_argument("--step-frame", type=int, default=1, help="step: first frame carrying the step (>= 1: frame 0 has no increment)")
    ap.add_argument("--sigma", type=float, default=0.0, help="random_walk: m/sqrt(s) per axis")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bearing-deg", type=float, default=0.0, help="drift direction, degrees CW from North")
    ap.add_argument("--allow-segments", action="store_true", help="allow runs with hard losses (drift is added per row regardless)")
    a = ap.parse_args(argv)

    if a.model == "step" and a.step_frame < 1:
        raise SystemExit("--step-frame must be >= 1: frame 0 carries no increment, and a non-zero root position "
                         "would be refused by the evaluator's frame-0 registration")
    metric, mfields = read_rows(a.vo_run / "metric_track.csv")
    align, afields = read_rows(a.vo_run / "alignment_frames.csv")
    if len(metric) != len(align):
        raise SystemExit("metric_track.csv and alignment_frames.csv differ in length")
    segs = {r["segment_index"].strip() for r in metric}
    if len(segs) > 1 and not a.allow_segments:
        raise SystemExit(f"{a.vo_run} has {len(segs)} VO segments (hard losses); pass --allow-segments to drift it anyway")
    t = np.array([float(r["timestamp_s"]) for r in metric])
    local = np.array([[float(r["segment_east_m"]), float(r["segment_north_m"])] for r in metric])
    for m, al in zip(metric, align):
        if float(al["local_east_m"]) != float(m["segment_east_m"]) or float(al["local_north_m"]) != float(m["segment_north_m"]):
            raise SystemExit("source files are not from one run (local != segment position)")
    step = np.hypot(*np.diff(local, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(step), step, 0.0))])
    d = drift_profile(a.model, t, s, local, a)
    d[~np.isfinite(d)] = 0.0

    for i, (m, al) in enumerate(zip(metric, align)):
        de, dn = float(d[i, 0]), float(d[i, 1])
        for key_e, key_n in (("segment_east_m", "segment_north_m"), ("metric_east_m", "metric_north_m")):
            if m.get(key_e, "") != "":
                m[key_e] = fmt(float(m[key_e]) + de); m[key_n] = fmt(float(m[key_n]) + dn)
        for key_e, key_n in (("local_east_m", "local_north_m"), ("global_east_m", "global_north_m")):
            if al.get(key_e, "") not in ("", "NaN", "nan"):
                al[key_e] = fmt(float(al[key_e]) + de); al[key_n] = fmt(float(al[key_n]) + dn)
    a.out.mkdir(parents=True, exist_ok=True)
    write_rows(a.out / "metric_track.csv", metric, mfields)
    write_rows(a.out / "alignment_frames.csv", align, afields)
    for name in ("frames.csv", "alignment_events.csv"):
        if (a.vo_run / name).exists():
            (a.out / name).write_bytes((a.vo_run / name).read_bytes())
    src_manifest = json.loads((a.vo_run / "manifest.json").read_text()) if (a.vo_run / "manifest.json").exists() else {}
    params = {k: getattr(a, k) for k in ("rate", "frac", "scale", "step", "step_frame", "sigma", "seed", "bearing_deg")}
    final = d[-1]
    manifest = dict(src_manifest)
    manifest["run_id"] = f"{src_manifest.get('run_id', a.vo_run.name)}-synthetic-drift-{a.model}"
    manifest["synthetic_drift"] = {
        "model": a.model, "parameters": params,
        "final_drift_m": {"east": float(final[0]), "north": float(final[1]), "magnitude": float(np.hypot(*final))},
        "max_drift_m": float(np.max(np.hypot(*d.T))),
        "applied_to": ["metric_track.csv: segment_east_m/segment_north_m, metric_east_m/metric_north_m",
                       "alignment_frames.csv: local_east_m/local_north_m, global_east_m/global_north_m"],
        "derived_from": str(a.vo_run),
        "source_sha256": {n: sha256(a.vo_run / n) for n in ("metric_track.csv", "alignment_frames.csv") if (a.vo_run / n).exists()},
        "evidence_tier_note": "T2 imagery, skyline captures, retrieval and memory are the recording's; the VO position "
                              "error is SYNTHETIC and declared. A controlled mechanism demonstration — never a navigation result.",
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{a.out}: {a.model} drift, final ({final[0]:+.2f}, {final[1]:+.2f}) m = {np.hypot(*final):.2f} m, "
          f"max {np.max(np.hypot(*d.T)):.2f} m over {len(t)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
