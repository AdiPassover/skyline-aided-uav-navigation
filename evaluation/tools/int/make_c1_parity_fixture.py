"""Produce the Python<->Java C1 parity fixture from the ACTUAL SKY matcher implementation.

``DEC-INT-002``: the Java ``SkylineMatcher`` must reproduce ``hsreloc.matchers`` (C1 bounded-lag NCC
and the C0 baseline) -- score, winning lag, overlap and ordering -- within numerical tolerance. A
hand-computed NCC is not evidence of that; this fixture is. It runs the real ``BoundedLagNccMatcher``
and ``FrozenNccMatcher`` on deterministic synthetic profiles (the ``tests/test_matchers.py`` signal,
transformed in known ways) and records every result. ``src/test/java/.../C1ParityTest`` replays it.

Run with the repository venv from the repository root::

    python evaluation/tools/int/make_c1_parity_fixture.py \\
        --out src/test/resources/relocalization/c1_parity_fixture.json

Regenerate only when ``hsreloc.matchers`` changes; the fixture records the matchers version.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HSRELOC_ROOT = REPO / "skyline"
if str(HSRELOC_ROOT) not in sys.path:
    sys.path.insert(0, str(HSRELOC_ROOT))

import numpy as np                                                        # noqa: E402

import hsreloc                                                            # noqa: E402,F401
from hsreloc.matchers import (MATCHERS_VERSION, BoundedLagNccMatcher, FrozenNccMatcher,  # noqa: E402
                              lag_samples_for_degrees)

N = 256


def signal(t):
    """The test_matchers.py signal: smooth, structured, non-periodic at N; defined for all real t."""
    t = np.asarray(t, dtype=np.float64)
    return (np.sin(2 * np.pi * t / 37.0) + 0.5 * np.sin(2 * np.pi * t / 13.0 + 1.0)
            + 0.25 * np.sin(2 * np.pi * t / 71.0 - 0.4))


def other_signal(t):
    t = np.asarray(t, dtype=np.float64)
    return (np.sin(2 * np.pi * t / 23.0 + 0.7) + 0.6 * np.sin(2 * np.pi * t / 53.0)
            + 0.2 * np.sin(2 * np.pi * t / 9.0 - 1.1))


def profiles() -> dict:
    t = np.arange(N, dtype=np.float64)
    rng = np.random.default_rng(20260905)
    centre = (N - 1) / 2.0
    p = {
        "q0": signal(t),
        "shift_m20": signal(t - 20.0),
        "shift_m7": signal(t - 7.0),
        "shift_p5": signal(t + 5.0),
        "shift_p13": signal(t + 13.0),
        "shift_p40": signal(t + 40.0),
        "scale_097": signal(centre + 0.97 * (t - centre)),
        "noisy": signal(t) + rng.normal(0.0, 0.05, N),
        "noisy_shift_p9": signal(t + 9.0) + rng.normal(0.0, 0.08, N),
        "offset": signal(t) + 3.7,
        "unrelated_a": other_signal(t),
        "unrelated_b": other_signal(t + 31.0),
        "negated": -signal(t),
        "ramp_plus_signal": signal(t) + 0.004 * (t - centre),
    }
    return {k: [float(v) for v in arr] for k, arr in p.items()}


def matchers() -> dict:
    return {
        "c0": FrozenNccMatcher(),
        "c1_32": BoundedLagNccMatcher(max_lag_samples=32),
        "c1_8": BoundedLagNccMatcher(max_lag_samples=8),
        "c1_32_overlap_090": BoundedLagNccMatcher(max_lag_samples=32, min_overlap_frac=0.9),
    }


def describe(m) -> dict:
    d = m.describe()
    return {"variant": d["variant"], "max_lag_samples": d.get("max_lag_samples", 0),
            "min_overlap_frac": d["min_overlap_frac"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    prof = profiles()
    names = list(prof)
    ms = matchers()

    pairs = []
    for mkey, m in ms.items():
        for q in ("q0", "noisy", "unrelated_a"):
            for r in names:
                res = m.match(np.asarray(prof[q]), np.asarray(prof[r]))
                pairs.append({
                    "matcher": mkey, "query": q, "reference": r,
                    "score": None if res.score == -np.inf else float(res.score),
                    "shift": float(res.shift), "overlap": int(res.overlap),
                    "overlap_frac": float(res.overlap_frac), "accepted": bool(res.accepted),
                    "score_at_zero_lag": res.diagnostics.get("score_at_zero_lag"),
                    "n_alignments_searched": res.diagnostics.get("n_alignments_searched"),
                })

    # Ranking case: a database of references, one query; expected order = (-score, then name order
    # index as the id tie-break), matching ReferenceMemory's (-score, ascending id).
    ranking_refs = ["shift_m20", "shift_m7", "shift_p5", "shift_p13", "shift_p40", "scale_097",
                    "unrelated_a", "unrelated_b", "ramp_plus_signal"]
    rankings = []
    for mkey in ("c1_32", "c0"):
        m = ms[mkey]
        scored = []
        for idx, r in enumerate(ranking_refs):
            res = m.match(np.asarray(prof["noisy_shift_p9"]), np.asarray(prof[r]))
            scored.append((idx, r, float(res.score) if res.score != -np.inf else None,
                           float(res.shift), bool(res.accepted)))
        admissible = [s for s in scored if s[4]]
        admissible.sort(key=lambda s: (-s[2], s[0]))
        rankings.append({"matcher": mkey, "query": "noisy_shift_p9", "references": ranking_refs,
                         "expected_order": [s[1] for s in admissible],
                         "expected_scores": [s[2] for s in admissible],
                         "expected_shifts": [s[3] for s in admissible],
                         "refused": [s[1] for s in scored if not s[4]]})

    fixture = {
        "generator": "evaluation/tools/int/make_c1_parity_fixture.py",
        "hsreloc_matchers_version": MATCHERS_VERSION,
        "n_samples": N,
        "note": ("Produced by the real hsreloc.matchers code on deterministic synthetic profiles. "
                 "Java must reproduce score within 1e-9 and shift/overlap/accepted exactly. The C1 "
                 "lag carries no pose semantics (DEC-INT-002)."),
        "lag_bound_example": {"fov_deg": 90.0, "max_lag_deg": 11.25, "n_samples": N,
                              "lag_samples": lag_samples_for_degrees(11.25, 90.0, N)},
        "matchers": {k: describe(m) for k, m in ms.items()},
        "profiles": prof,
        "pairs": pairs,
        "rankings": rankings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    n_refused = sum(1 for p in pairs if not p["accepted"])
    print(f"[int] wrote {args.out}: {len(pairs)} pairs ({n_refused} refused alignments), "
          f"{len(rankings)} ranking cases, matchers version {MATCHERS_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
