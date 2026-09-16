"""Range-cached HTTP access to a MARS-LVIG MCAP mirror, for `EXP-VO-007`.

`naveval.mcap_reader` and `naveval.ingest_mars_lvig` deliberately have **no I/O policy** — both
window loaders take `fetch_range`/`fetch_tail` callables. This module is the production I/O policy
for a long window, and it exists because the obvious one does not scale:

- `load_..._mcap_window` fetches the whole overlapping chunk span as **one `bytes` object**. On
  `HKairport01`'s 616 s window that is ~16.5 GB; on a 1,200 s window it is ~29 GB. Neither fits in
  this machine's 15.7 GB of RAM, so a long sequence must be read in **sub-windows** and the results
  concatenated. That is what `iter_subwindows` is for.
- `EXP-002` recorded that a single `curl` range request over ~16.5 GB failed with exit 18 (partial
  transfer) three times at three different offsets, and was fixed by splitting into 512 MiB
  sub-requests with independent retries. That fix is reproduced here as `_fetch_to_file`, rather
  than left in a session's shell history.
- Chunks are channel-interleaved, so imagery and attitude for the same time window live in the
  *same* bytes. `EXP-003` paid the download twice because the first pass discarded them. Here each
  sub-window's bytes are cached **on disk** and served to both loaders before being deleted, so a
  sequence with an attitude sidecar costs one network pass, not two.

Nothing in this module parses MCAP: `naveval` does all of that, unmodified.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

CHUNK_BYTES = 512 * 1024 * 1024      # EXP-002's fetch-mechanics fix
MAX_ATTEMPTS = 7
RETRY_BACKOFF_S = (2, 5, 15, 40, 90, 180)   # a transient CDN failure outlives an immediate retry


def mirror_url(scene: str) -> str:
    """The ROS 2 / MCAP mirror of one MARS-LVIG sequence (`LIT-006` Stage 2, source B)."""
    return (
        "https://huggingface.co/datasets/DapengFeng/MCAP/resolve/main/"
        f"mars_lvig/{scene}/{scene}_0.mcap"
    )


def remote_size(url: str) -> int:
    """File size in bytes, from a HEAD that follows HuggingFace's signed redirect.

    The redirect chain emits one header block per hop, so the *last* `content-length` is the
    object's rather than the 302's. Parsed from the raw headers rather than from curl's
    `%{header_json}` (needs curl >= 7.83) and written without `-o /dev/null`, which fails with
    exit 23 on the Windows curl this project runs on.
    """
    out = subprocess.run(
        ["curl", "-sIL", "--max-time", "120", url],
        capture_output=True, text=True, check=True,
    ).stdout
    sizes = [line.split(":", 1)[1].strip() for line in out.splitlines()
             if line.lower().startswith("content-length:")]
    if not sizes:
        raise RuntimeError(f"No Content-Length in HEAD of {url}")
    return int(sizes[-1])


def _fetch_to_file(url: str, start: int, end_inclusive: int, out: Path) -> None:
    """Inclusive byte range -> file, in 512 MiB pieces, each retried independently.

    Written piece by piece and appended, so a failure costs at most one piece rather than the
    whole range (`EXP-002` *Limitations*, fetch-mechanics failure).

    **Retries back off.** The first version retried five times with no delay, and lost a whole
    sub-window to a transient CDN failure at byte 17,064,818,502: five immediate attempts all hit
    the same bad second, and the range fetched perfectly a minute later. Immediate retries are also
    exactly wrong against rate limiting, which is one of the few causes that a delay actually fixes.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    pos = start
    with out.open("wb") as fh:
        while pos <= end_inclusive:
            hi = min(pos + CHUNK_BYTES - 1, end_inclusive)
            piece = out.with_suffix(".part")
            for attempt in range(1, MAX_ATTEMPTS + 1):
                rc = subprocess.run(
                    ["curl", "-sL", "--fail", "--retry", "0",
                     "-H", f"Range: bytes={pos}-{hi}", url, "-o", str(piece)],
                ).returncode
                if rc == 0 and piece.exists() and piece.stat().st_size == hi - pos + 1:
                    break
                delay = RETRY_BACKOFF_S[min(attempt - 1, len(RETRY_BACKOFF_S) - 1)]
                print(f"    [retry {attempt}/{MAX_ATTEMPTS} in {delay}s] bytes {pos}-{hi} "
                      f"(rc={rc}, got {piece.stat().st_size if piece.exists() else 0})",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
            else:
                raise RuntimeError(f"Range {pos}-{hi} of {url} failed {MAX_ATTEMPTS} times")
            # Streamed copy in small blocks rather than one read_bytes(): a single 512 MiB
            # read() intermittently fails on Windows with OSError(22) (observed 2026-08-30,
            # EXP-CONF-001 P5, AMtown02 sub-window 8/14); block-wise copying sidesteps the
            # large-single-read path entirely, with one delayed retry for transient locks.
            for attempt in (1, 2):
                try:
                    with piece.open("rb") as src:
                        shutil.copyfileobj(src, fh, 8 * 1024 * 1024)
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(5)
            piece.unlink()
            pos = hi + 1


@dataclass
class RangeCache:
    """Serves `fetch_range`/`fetch_tail` for one MCAP file, from a disk cache.

    Two regions are cached: the **tail** (footer, fetched once) and the **current sub-window's
    chunk span** (`prefetch`). A request that falls outside both goes to HTTP directly — which in
    practice is only the summary block, ~4.3 MB, itself cached after the first read.
    """
    url: str
    size: int
    cache_dir: Path
    _tail: bytes = b""
    _summary: bytes = b""
    _summary_start: int = -1
    _lo: int = -1
    _hi: int = -1
    _blob: Path = Path()

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._blob = self.cache_dir / "window.bin"

    def fetch_tail(self, n: int) -> bytes:
        if len(self._tail) < n:
            p = self.cache_dir / f"tail_{n}.bin"
            if not p.exists() or p.stat().st_size != n:
                _fetch_to_file(self.url, self.size - n, self.size - 1, p)
            self._tail = p.read_bytes()
        return self._tail[-n:]

    def fetch_range(self, start: int, end_inclusive: int) -> bytes:
        if self._lo <= start and end_inclusive <= self._hi and self._blob.exists():
            with self._blob.open("rb") as fh:
                fh.seek(start - self._lo)
                return fh.read(end_inclusive - start + 1)
        if self._summary_start >= 0 and start == self._summary_start and \
                end_inclusive == self.size - 1 and self._summary:
            return self._summary
        p = self.cache_dir / "adhoc.bin"
        _fetch_to_file(self.url, start, end_inclusive, p)
        data = p.read_bytes()
        p.unlink()
        if end_inclusive == self.size - 1 and (end_inclusive - start) < 64 * 1024 * 1024:
            self._summary_start, self._summary = start, data   # summary block: reused every call
        return data

    def prefetch(self, lo: int, hi_inclusive: int) -> int:
        """Cache `[lo, hi_inclusive]` on disk; returns the byte count fetched."""
        self._lo, self._hi = lo, hi_inclusive
        _fetch_to_file(self.url, lo, hi_inclusive, self._blob)
        return hi_inclusive - lo + 1

    def release(self) -> None:
        if self._blob.exists():
            self._blob.unlink()
        self._lo = self._hi = -1


def iter_subwindows(t_start_s: float, t_end_s: float, span_s: float) -> Iterator[tuple[int, int]]:
    """Split `[t_start_s, t_end_s]` into half-open nanosecond sub-windows of `span_s`.

    Half-open on the right (except the last, which is closed) so that a message on a boundary is
    read exactly once and no de-duplication is needed downstream — the loaders filter on
    `t_start_ns <= log_time <= t_end_ns`, so overlapping closed windows would double-count.

    Boundaries are stepped in **integer nanoseconds**. Accumulating in seconds would drift by a few
    hundred nanoseconds per step at these epoch values (float64 gives ~2.4e-7 s of resolution near
    1.66e9), which is small but would open real sub-microsecond gaps between consecutive windows —
    and a message that falls in one is silently lost rather than reported.
    """
    t_ns, end_ns, span_ns = (int(round(t_start_s * 1e9)), int(round(t_end_s * 1e9)),
                             int(round(span_s * 1e9)))
    while t_ns < end_ns:
        nxt = min(t_ns + span_ns, end_ns)
        yield t_ns, (nxt if nxt >= end_ns else nxt - 1)
        t_ns = nxt
