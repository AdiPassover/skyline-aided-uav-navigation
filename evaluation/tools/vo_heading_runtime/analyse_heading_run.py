"""Report what a `DEC-VO-010` heading run actually did, from its committed artifacts alone.

Reads a run's ``metric_track.csv`` (and, where a dataset supplies them, ``groundtruth.csv`` /
``attitude.csv``) and prints:

* the heading channel's own behaviour -- sample age, freshness distribution, rejected samples;
* the visual-vs-external yaw disagreement, wrap-safe;
* the accumulated drift the substitution removes, measured differentially so the constant
  camera-to-body frame offset cancels;
* every hard-loss event, with the heading evolution the visual path missed across it;
* a raw-pose regression check against a baseline run, which must be bit-identical.

It states rather than assumes: every quantity it prints names the file it came from, and it refuses
to compute a comparison whose inputs are absent.

Usage:
  python evaluation/tools/vo_heading_runtime/analyse_heading_run.py \
      --run runs/hkairport01-a-homography-heading-v1 \
      --dataset datasets/hkairport01-a \
      [--baseline runs/hkairport01-a-homography-rigid-v1] \
      [--control runs/hkairport01-a-homography-metric-visualyaw-v1]
"""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import math
import pathlib
import sys


def rows(path: pathlib.Path) -> list[dict]:
    with io.open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wrap180(d: float) -> float:
    return ((d + 180.0) % 360.0) - 180.0


def unwrap(seq: list[float]) -> list[float]:
    out = [seq[0]]
    for v in seq[1:]:
        out.append(out[-1] + wrap180(v - out[-1]))
    return out


def interp_ang(ts: list[float], vs: list[float], t: float) -> float:
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return vs[0]
    if i >= len(ts):
        return vs[-1]
    f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return vs[i - 1] + wrap180(vs[i] - vs[i - 1]) * f


def fnum(cell: str):
    cell = (cell or "").strip()
    return None if cell == "" else float(cell)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--control")
    a = ap.parse_args()

    run = pathlib.Path(a.run)
    ds = pathlib.Path(a.dataset)
    mt = rows(run / "metric_track.csv")
    fr = rows(run / "frames.csv")

    print("=" * 78)
    print("RUN      %s   (%d metric rows, %d frames)" % (run.name, len(mt), len(fr)))
    if not mt:
        print("(no metric rows -- the run produced nothing; refusing to report on an empty record)")
        return 1

    # ---------------------------------------------------------------- heading channel behaviour
    ages = [fnum(r["yaw_nav_age_s"]) for r in mt]
    ages = [x for x in ages if x is not None]
    statuses: dict[str, int] = {}
    for r in mt:
        statuses[r["yaw_nav_status"]] = statuses.get(r["yaw_nav_status"], 0) + 1
    if ages:
        s = sorted(ages)
        print("HEADING  sample age at a frame: median %.4f s  p99 %.4f  max %.4f"
              % (s[len(s) // 2], s[int(0.99 * len(s))], s[-1]))
    print("HEADING  freshness: " + "  ".join("%s=%d" % kv for kv in sorted(statuses.items())))
    navs = [fnum(r["yaw_nav_deg"]) for r in mt]
    have = [x for x in navs if x is not None]
    if have:
        print("HEADING  yaw_nav spans %.3f .. %.3f deg over %d frames (%d distinct)"
              % (min(have), max(have), len(have), len(set(round(x, 6) for x in have))))
    else:
        print("HEADING  no authoritative heading was ever available in this run")

    # ---------------------------------------------------------------- visual vs external
    dis = [fnum(r["yaw_disagreement_deg"]) for r in mt]
    dis = [x for x in dis if x is not None]
    if dis:
        sd = sorted(abs(x) for x in dis)
        print("YAW DIFF |visual - external| per frame: median %.4f  p99 %.4f  max %.4f deg (n=%d)"
              % (sd[len(sd) // 2], sd[int(0.99 * len(sd))], sd[-1], len(sd)))

    # accumulated drift the substitution removes, measured differentially
    vis = [fnum(r["yaw_visual_deg"]) for r in mt]
    if have and all(v is not None for v in vis) and len(have) == len(mt):
        vu = unwrap([v for v in vis])
        nu = unwrap(navs)
        err = [(vu[i] - vu[0]) - (nu[i] - nu[0]) for i in range(len(mt))]
        rms = math.sqrt(sum(e * e for e in err) / len(err))
        print("DRIFT    visual yaw vs authoritative heading, differentially: rms %.3f deg, "
              "final %+.3f deg" % (rms, err[-1]))
        print("         (this is the accumulated error the substitution removes; it is measured as "
              "a DIFFERENCE so the constant camera-to-body frame offset cancels)")

    # ---------------------------------------------------------------- hard-loss events
    losses = [(i, r) for i, r in enumerate(mt) if r["event"] == "restart"]
    print("SEGMENTS %d hard-loss event(s); final segment_index %s"
          % (len(losses), mt[-1]["segment_index"]))
    att = ds / "attitude.csv"
    ats = ayaw = None
    if att.exists():
        A = rows(att)
        ats = [float(r["timestamp_s"]) for r in A]
        ayaw = unwrap([float(r["yaw_compass_deg"]) for r in A])
    for i, r in losses:
        prev = mt[i - 1] if i > 0 else None
        print("  restart at frame %s (t=%.4f): unknown_translation_gap=%s  "
              "heading_known_across_gap=%s  yaw_nav=%s  status=%s"
              % (r["frame_index"], float(r["timestamp_s"]), r["unknown_translation_gap"],
                 r["heading_known_across_gap"], r["yaw_nav_deg"] or "-", r["yaw_nav_status"]))
        if prev is not None:
            de = float(r["metric_east_m"]) - float(prev["metric_east_m"])
            dn = float(r["metric_north_m"]) - float(prev["metric_north_m"])
            print("     displacement inserted across the gap: (%.3e, %.3e) m -- must be exactly 0"
                  % (de, dn))
            pn, rn = fnum(prev["yaw_nav_deg"]), fnum(r["yaw_nav_deg"])
            if pn is not None and rn is not None:
                print("     heading change the external channel measured across it: %+.4f deg"
                      % wrap180(rn - pn))
            pv, rv = fnum(prev["yaw_visual_deg"]), fnum(r["yaw_visual_deg"])
            if pv is not None and rv is not None:
                print("     rotation the VISUAL path inserted across it:            %+.4f deg"
                      % wrap180(rv - pv))
        if ats is not None and i > 0:
            t0 = float(mt[i - 1]["timestamp_s"])
            t1 = float(r["timestamp_s"])
            print("     ground-truth FC yaw change over the same interval:      %+.4f deg (%.4f s)"
                  % (interp_ang(ats, ayaw, t1) - interp_ang(ats, ayaw, t0), t1 - t0))

    # ------------------------------------------------- what the substitution changed, exactly
    # The visual-yaw track is EXACTLY reconstructible from this run's own sidecar, so no second
    # replay is needed. RigidNavigationState accumulates (centreX, centreY) from the same
    # increment MetricNavigationState scales, so the raw pose increment IS R(theta_{k-1})*dq_k in
    # Pose3D axes and the visual-yaw metric increment is that increment times the SAME gsd this row
    # already reports:  d_east = d_raw_x * gsd,  d_north = d_raw_y * gsd.
    if have and len(have) == len(mt):
        ve = vn = 0.0
        sep = 0.0
        for i in range(1, len(mt)):
            g = fnum(mt[i]["gsd_m_per_px"])
            if g is None or math.isnan(g):
                continue
            ve += (float(mt[i]["raw_x_px"]) - float(mt[i - 1]["raw_x_px"])) * g
            vn += (float(mt[i]["raw_y_px"]) - float(mt[i - 1]["raw_y_px"])) * g
            sep = max(sep, math.hypot(float(mt[i]["metric_east_m"]) - ve,
                                      float(mt[i]["metric_north_m"]) - vn))
        print("SUBST    visual-yaw track reconstructed exactly from this run: ends (%.2f, %.2f) m "
              "against the heading track's (%.2f, %.2f) m"
              % (ve, vn, float(mt[-1]["metric_east_m"]), float(mt[-1]["metric_north_m"])))
        print("         max separation between the two %.2f m, endpoint separation %.2f m -- the "
              "size of what the heading substitution changed"
              % (sep, math.hypot(float(mt[-1]["metric_east_m"]) - ve,
                                 float(mt[-1]["metric_north_m"]) - vn)))

    # ---------------------------------------------------------------- raw-pose regression
    for label, other in (("baseline", a.baseline), ("control", a.control)):
        if not other:
            continue
        o = pathlib.Path(other)
        if not (o / "frames.csv").exists():
            print("REGRESS  %s missing: %s" % (label, o))
            continue
        ofr = rows(o / "frames.csv")
        n = min(len(fr), len(ofr))
        dmax = max(math.hypot(float(fr[i]["est_x"]) - float(ofr[i]["est_x"]),
                              float(fr[i]["est_y"]) - float(ofr[i]["est_y"])) for i in range(n))
        ymax = max(abs(wrap180(float(fr[i]["est_yaw_deg"]) - float(ofr[i]["est_yaw_deg"])))
                   for i in range(n))
        print("REGRESS  published pose vs %s (%s): max %.3e px, max yaw %.3e deg over %d frames"
              % (label, o.name, dmax, ymax, n))

    if a.control:
        c = pathlib.Path(a.control)
        if (c / "metric_track.csv").exists():
            cm = rows(c / "metric_track.csv")
            n = min(len(mt), len(cm))
            d = max(math.hypot(float(mt[i]["metric_east_m"]) - float(cm[i]["metric_east_m"]),
                               float(mt[i]["metric_north_m"]) - float(cm[i]["metric_north_m"]))
                    for i in range(n))
            print("SUBST    metric track vs the visual-yaw control: max separation %.3f m over %d "
                  "frames -- the size of what the heading substitution changed" % (d, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
