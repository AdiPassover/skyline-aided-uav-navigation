"""Build a `contracts/dataset.md` dataset from any MARS-LVIG MCAP sequence (`EXP-VO-007`).

`EXP-002` built `hkairport01-{a,b}` from an ad-hoc session script that was never committed; this is
the same job made reproducible and sequence-general. All MCAP parsing, RTK merging, cruise-window
selection, ENU conversion and dataset writing is `naveval`'s, unmodified. What lives here is only
what `naveval` deliberately does not own: the I/O policy (`mcap_source`) and the CLI.

Two things it does that the ad-hoc path could not:

1. **Sub-windowed reads.** `naveval.ingest_mars_lvig.load_hkairport01_mcap_window` materialises the
   whole overlapping chunk span as one `bytes` object. A 1,190 s MARS-LVIG window is ~29 GB, which
   does not fit in this machine's RAM. The window is split, each piece read separately, and the
   results concatenated -- the loader is called many times instead of once, unchanged.
2. **Frames stream to disk, not into a list.** 12,000 JPEGs at ~1.26 MB is ~15 GB of frame payload
   if held in memory. Each frame is written to `<out>/images/` under the exact name
   `build_dataset_from_mcap` will use, and a lazy handle is passed on in its place, so peak memory
   is one sub-window rather than one flight.

Attitude is decoded from the *same* bytes in the same pass when `--attitude` is given. `EXP-003`
paid the ~16.5 GB download a second time because the first pass discarded those messages.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mcap_source as src                                            # noqa: E402
from naveval import mcap_reader as mcap                              # noqa: E402
from naveval.ingest_mars_lvig import (                               # noqa: E402
    RTK_MERGE_TOLERANCE_NS,
    AttitudeSample,
    RtkSample,
    build_dataset_from_mcap,
    load_hkairport01_attitude_window,
    load_hkairport01_mcap_window,
    write_attitude_sidecar,
)


class StagedFrame:
    """Duck-types `McapFrame` for `build_dataset_from_mcap`, reading its bytes from disk.

    The staged file already carries the name the builder will write (`<t:.6f>.jpg` under
    `<out>/images/`), so the builder's own `write_bytes` re-writes the same bytes to the same path
    -- a no-op in content, and it keeps `naveval` the single writer of every dataset file.
    """
    __slots__ = ("timestamp_s", "path")

    def __init__(self, timestamp_s: float, path: Path) -> None:
        self.timestamp_s, self.path = timestamp_s, path

    @property
    def data(self) -> bytes:
        return self.path.read_bytes()


def chunk_span(cache: src.RangeCache, t_start_ns: int, t_end_ns: int) -> tuple[int, int]:
    """Byte range of the MCAP chunks overlapping a time window, from the summary index alone."""
    summary_start, _ = mcap.read_footer(cache.fetch_tail)
    _channels, chunk_indexes, _stats = mcap.parse_summary(
        cache.fetch_range(summary_start, cache.size - 1))
    over = mcap.chunks_overlapping(chunk_indexes, t_start_ns, t_end_ns)
    if not over:
        raise SystemExit(f"No MCAP chunks overlap [{t_start_ns}, {t_end_ns}] ns")
    return (min(c.chunk_start_offset for c in over),
            max(c.chunk_start_offset + c.chunk_length for c in over) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest one MARS-LVIG MCAP sequence window.")
    ap.add_argument("--scene", required=True, help="MARS-LVIG sequence name, e.g. AMtown01")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--dataset-revision", default="v1")
    ap.add_argument("--out-root", required=True, help="parent directory; <out-root>/<dataset-id>")
    ap.add_argument("--t-start", type=float, required=True, help="window start, UTC seconds")
    ap.add_argument("--t-end", type=float, required=True, help="window end, UTC seconds")
    ap.add_argument("--subwindow-s", type=float, default=60.0)
    ap.add_argument("--n-subwindows", type=int, default=0,
                    help="build only the first N sub-windows of the frozen tiling -- a technical "
                         "truncation for an acquisition that cannot complete (EXP-VO-007). The "
                         "tiling itself is unchanged, so every already-fetched sub-window resumes "
                         "from its checkpoint and the prefix boundary is a tiling boundary, not a "
                         "value anyone picked.")
    ap.add_argument("--target-agl", type=float, required=True)
    ap.add_argument("--nominal-speed", type=float, required=True)
    ap.add_argument("--yaw-offset", type=float, required=True,
                    help="rtk_yaw -> compass offset in degrees; MUST be re-derived per sequence "
                         "(LIT-006), never inherited from HKairport01")
    ap.add_argument("--yaw-derivation", default="", help="how --yaw-offset was obtained")
    ap.add_argument("--environment", default="")
    ap.add_argument("--capture-date", default="")
    ap.add_argument("--evidence-caveat", default="")
    ap.add_argument("--attitude-caveat", default="")
    ap.add_argument("--intrinsics-json", default="", help="path to a camera_intrinsics JSON object")
    ap.add_argument("--ground-datum", type=float, default=None,
                    help="whole-flight altitude datum; required for a cruise-only window")
    ap.add_argument("--notes", default="")
    ap.add_argument("--provenance", default="")
    ap.add_argument("--attitude", action="store_true", help="also write attitude.csv")
    ap.add_argument("--cache-dir", required=True, help="scratch space for one sub-window of bytes")
    args = ap.parse_args()

    url = src.mirror_url(args.scene)
    size = src.remote_size(url)
    out_root = Path(args.out_root) / args.dataset_id
    images = out_root / "images"
    images.mkdir(parents=True, exist_ok=True)
    cache = src.RangeCache(url=url, size=size, cache_dir=Path(args.cache_dir))
    print(f"{args.scene}: {size / 1e9:.2f} GB at {url}", flush=True)

    # Per-sub-window checkpoints. A full MARS-LVIG cruise is ~29 GB over hours on this link; a
    # failure at hour two must not restart the fetch. Imagery is already durable (each frame is
    # written as it is decoded), so only the RTK and attitude samples need persisting, and a
    # completed window is then skipped entirely on a re-run.
    #
    # They live beside the byte cache, not inside the dataset: `datasets/<id>/` holds the files
    # `contracts/dataset.md` names plus provenance, and nothing else. The checkpoints are
    # acquisition scratch that duplicates what `groundtruth.csv` and `attitude.csv` already carry.
    ckpt_dir = Path(args.cache_dir) / f"checkpoints_{args.dataset_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rtk_all, frames_all, att_all = [], [], []
    seen_rtk, seen_frame, seen_att = set(), set(), set()
    total_bytes, t0 = 0, time.time()
    windows = list(src.iter_subwindows(args.t_start, args.t_end, args.subwindow_s))
    if args.n_subwindows and args.n_subwindows < len(windows):
        windows = windows[:args.n_subwindows]
        print(f"  TRUNCATED to the first {len(windows)} of the frozen tiling", flush=True)
    t_end_effective = windows[-1][1] / 1e9
    for i, (a_ns, b_ns) in enumerate(windows, 1):
        ckpt = ckpt_dir / f"{a_ns}_{b_ns}.json"
        if ckpt.exists():
            saved = json.loads(ckpt.read_text())
            rtk = [RtkSample(**r) for r in saved["rtk"]]
            frames = [StagedFrame(t, images / f"{t:.6f}.jpg") for t in saved["frame_times"]]
            att = [AttitudeSample(**a) for a in saved["attitude"]]
            missing = [f for f in frames if not f.path.exists()]
            if missing:
                raise SystemExit(f"Checkpoint {ckpt.name} references {len(missing)} missing images")
            resumed = True
        else:
            # Loaded in short disjoint pieces INSIDE the checkpointed window (added 2026-08-30,
            # EXP-CONF-001 P5): a whole 45 s window materialises ~1.3 GB of chunk bytes plus every
            # frame's JPEG payload at once, which deterministically ran out of memory on the
            # acquisition machine (MemoryError at AMtown02 sub-window 12/14, twice). Pieces are
            # half-open [pa, pb) spans of the SAME window, so the union of messages is identical;
            # frames go to disk per piece; the tiling and checkpoint keys are unchanged.
            #
            # Piece reads are PADDED by the RTK merge tolerance on both sides (fix 2026-08-31,
            # EXP-CONF-002): the merged RTK sample is anchored at the position message but needs
            # its partner messages (yaw/info/connection) within RTK_MERGE_TOLERANCE_NS, and a
            # partner lying just across an unpadded piece seam orphaned the position message in
            # BOTH pieces -- one merged sample was silently lost per ~2.5k seam crossings
            # (observed once on the amtown01-d rebuild; zero occurrences in any committed
            # dataset, verified by 5 Hz-grid gap audit). Outputs are filtered back to
            # [a_ns, b_ns] so the frozen window semantics are unchanged; duplicates at seams
            # are removed by the existing timestamp dedup.
            piece_ns = int(15.0 * 1e9)
            pad_ns = RTK_MERGE_TOLERANCE_NS
            rtk, frames, att = [], [], []
            pa = a_ns
            while pa < b_ns:
                pb = min(pa + piece_ns, b_ns)
                qa, qb = pa - pad_ns, pb + pad_ns
                lo, hi = chunk_span(cache, qa, qb)
                total_bytes += cache.prefetch(lo, hi)
                p_rtk, p_raw = load_hkairport01_mcap_window(
                    cache.fetch_range, cache.fetch_tail, size, qa, qb)
                for f in p_raw:
                    if not (a_ns <= int(round(f.timestamp_s * 1e9)) <= b_ns):
                        continue
                    p = images / f"{f.timestamp_s:.6f}.jpg"
                    p.write_bytes(f.data)
                    frames.append(StagedFrame(f.timestamp_s, p))
                del p_raw
                rtk.extend(s for s in p_rtk
                           if a_ns <= int(round(s.timestamp_s * 1e9)) <= b_ns)
                if args.attitude:
                    att.extend(s for s in load_hkairport01_attitude_window(
                        cache.fetch_range, cache.fetch_tail, size, qa, qb)
                        if a_ns <= int(round(s.timestamp_s * 1e9)) <= b_ns)
                cache.release()
                pa = pb
            ckpt.write_text(json.dumps({
                "rtk": [asdict(s) for s in rtk],
                "frame_times": [f.timestamp_s for f in frames],
                "attitude": [asdict(s) for s in att],
            }))
            resumed = False

        for s in rtk:
            if s.timestamp_s not in seen_rtk:
                seen_rtk.add(s.timestamp_s)
                rtk_all.append(s)
        for f in frames:
            if f.timestamp_s not in seen_frame:
                seen_frame.add(f.timestamp_s)
                frames_all.append(f)
        for s in att:
            if s.timestamp_s not in seen_att:
                seen_att.add(s.timestamp_s)
                att_all.append(s)

        el = time.time() - t0
        print(f"  [{i}/{len(windows)}]{' resumed' if resumed else ''} "
              f"{(b_ns - a_ns) / 1e9:.0f}s window; "
              f"{len(frames_all)} frames, {len(rtk_all)} rtk, {len(att_all)} att; "
              f"{total_bytes / 1e9:.2f} GB in {el / 60:.1f} min", flush=True)

    frames_all.sort(key=lambda f: f.timestamp_s)
    rtk_all.sort(key=lambda s: s.timestamp_s)
    att_all.sort(key=lambda s: s.timestamp_s)

    intrinsics = json.loads(Path(args.intrinsics_json).read_text()) if args.intrinsics_json else None
    result = build_dataset_from_mcap(
        out_root, rtk_all, frames_all, args.dataset_id, args.dataset_revision,
        scene=args.scene,
        window=(args.t_start, t_end_effective),
        target_agl_m=args.target_agl,
        nominal_speed_ms=args.nominal_speed,
        provenance=args.provenance,
        notes=args.notes,
        mcap_source_url=url,
        ground_datum_override_m=args.ground_datum,
        rtk_yaw_to_compass_offset_deg=args.yaw_offset,
        heading_offset_derivation=args.yaw_derivation or None,
        environment=args.environment or None,
        capture_date=args.capture_date or None,
        camera_intrinsics=intrinsics,
        evidence_caveat=args.evidence_caveat or None,
        attitude_caveat=args.attitude_caveat or None,
    )
    if args.attitude:
        result["attitude_csv"] = str(write_attitude_sidecar(out_root, att_all))
        result["n_attitude"] = len(att_all)
    result["root"] = str(result["root"])
    result["fetched_bytes"] = total_bytes
    result["wall_time_s"] = round(time.time() - t0, 1)
    print(json.dumps(result, indent=2))
    (out_root / "ingest_provenance.json").write_text(
        json.dumps({"argv": sys.argv[1:], "url": url, "file_size": size,
                    "frozen_window": [args.t_start, args.t_end],
                    "built_window": [args.t_start, t_end_effective],
                    "n_subwindows_built": len(windows),
                    "n_subwindows_frozen": len(
                        list(src.iter_subwindows(args.t_start, args.t_end, args.subwindow_s))),
                    **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
