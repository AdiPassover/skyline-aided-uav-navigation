"""Regression test for the piece-wise window loading seam loss (found 2026-08-31 by
EXP-CONF-002's byte-identity gate on the amtown01-d rebuild).

Mechanism: a merged RTK sample is anchored at the position message but requires partner
messages (yaw/info/connection) within RTK_MERGE_TOLERANCE_NS. With unpadded half-open piece
reads, a position message just before a piece seam whose partners log just after it is orphaned
in BOTH pieces and the merged sample is silently lost. The fix in ingest_sequence.py pads each
piece read by RTK_MERGE_TOLERANCE_NS and filters outputs back to the window.

This test exercises the loader-level property the fix relies on: the padded piece union equals
the whole-window read; the unpadded union does not.

Run with pytest from this directory (same convention as test_analyse.py).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = TOOL_DIR.parents[1]
sys.path.insert(0, str(EVALUATION_DIR))

from naveval.ingest_mars_lvig import (  # noqa: E402
    RTK_MERGE_TOLERANCE_NS,
    load_hkairport01_mcap_window,
)

# Import the synthetic-MCAP builders from the shared test module by path (read-only reuse).
_spec = importlib.util.spec_from_file_location(
    "mcap_test_helpers", EVALUATION_DIR / "tests" / "test_ingest_mars_lvig_mcap.py")
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)

SEC = 1_000_000_000


def _window(full: bytes, a_ns: int, b_ns: int):
    fetch_range, fetch_tail = _helpers._fetchers(full)
    rtk, frames = load_hkairport01_mcap_window(fetch_range, fetch_tail, len(full), a_ns, b_ns)
    return rtk


def _build_straddling_file(seam_ns: int):
    """RTK cycles every 0.2 s; ONE cycle has its position 1 ms before the seam and its
    partner messages 1 ms after it (partner offset 2 ms << tolerance 150 ms)."""
    msgs = []
    t = 0
    while t <= 30 * SEC:
        if t == seam_ns - SEC // 5:      # the cycle just before the seam: straddle it
            pos_t = seam_ns - 1_000_000
            part_t = seam_ns + 1_000_000
            cyc = _helpers._rtk_cycle(pos_t, _helpers.LAT0, _helpers.LON0, _helpers.ALT0, 176)
            msgs.extend([(topic, pos_t if topic.endswith("rtk_position") else part_t, pay)
                         for topic, _, pay in cyc])
            # next regular cycle 0.2 s after the straddling position (real 5 Hz cadence:
            # neighbouring cycles are ~200 ms away, outside the 150 ms merge tolerance)
            t = seam_ns + SEC // 5
            continue
        msgs.extend(_helpers._rtk_cycle(t, _helpers.LAT0, _helpers.LON0, _helpers.ALT0, 176))
        t += SEC // 5
    full, _ = _helpers.build_synthetic_mcap([msgs])
    return full


class TestPieceSeamStraddling:
    seam = 15 * SEC
    a, b = 0, 30 * SEC

    def test_whole_window_read_keeps_the_straddling_sample(self):
        full = _build_straddling_file(self.seam)
        whole = _window(full, self.a, self.b)
        assert any(abs(s.timestamp_s * SEC - (self.seam - 1_000_000)) < 1e3 for s in whole)

    def test_unpadded_piece_union_loses_it(self):
        """Documents the regression: this is what the pre-fix piece loop computed."""
        full = _build_straddling_file(self.seam)
        whole = _window(full, self.a, self.b)
        union = {s.timestamp_s for s in _window(full, self.a, self.seam)} | \
                {s.timestamp_s for s in _window(full, self.seam, self.b)}
        lost = {s.timestamp_s for s in whole} - union
        assert lost == {(self.seam - 1_000_000) / SEC}

    def test_padded_piece_union_equals_whole_window(self):
        """The fixed semantics: pad reads by the merge tolerance, filter to the window."""
        full = _build_straddling_file(self.seam)
        whole = {s.timestamp_s for s in _window(full, self.a, self.b)}
        pad = RTK_MERGE_TOLERANCE_NS
        union = set()
        for pa, pb in ((self.a, self.seam), (self.seam, self.b)):
            for s in _window(full, pa - pad, pb + pad):
                if self.a <= int(round(s.timestamp_s * SEC)) <= self.b:
                    union.add(s.timestamp_s)
        assert union == whole
