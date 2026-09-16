"""Read a remote ZIP's central directory via HTTP range requests, and extract
individual entries -- without downloading the archive.

Stage-1 sampling tool for DEC-005's dataset validation. Deliberately minimal: this is
NOT a dataset adapter, it is a way to look at ~100 images inside a 28 GB archive.
Handles ZIP64 (required: the archive is >4 GB).
"""
from __future__ import annotations

import struct
import subprocess
import sys
import zlib
from pathlib import Path

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CEN_SIG = b"PK\x01\x02"


def fetch_range(url: str, start: int, end: int, out: Path) -> Path:
    """Inclusive byte range -> file, via curl (follows HF's signed redirect)."""
    subprocess.run(
        ["curl", "-sL", "-H", f"Range: bytes={start}-{end}", url, "-o", str(out)],
        check=True,
    )
    return out


def parse_eocd(tail: bytes, tail_offset: int) -> tuple[int, int]:
    """-> (central_directory_offset, central_directory_size), ZIP64-aware."""
    i = tail.rfind(EOCD_SIG)
    if i < 0:
        raise ValueError("No EOCD found in tail")
    cd_size = struct.unpack_from("<I", tail, i + 12)[0]
    cd_off = struct.unpack_from("<I", tail, i + 16)[0]

    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(EOCD64_LOC_SIG)
        if j < 0:
            raise ValueError("ZIP64 indicated but no EOCD64 locator")
        eocd64_off = struct.unpack_from("<Q", tail, j + 8)[0]
        k = eocd64_off - tail_offset
        if k < 0 or tail[k:k + 4] != EOCD64_SIG:
            raise ValueError("EOCD64 outside fetched tail")
        cd_size = struct.unpack_from("<Q", tail, k + 40)[0]
        cd_off = struct.unpack_from("<Q", tail, k + 48)[0]
    return cd_off, cd_size


def parse_central_directory(cd: bytes) -> list[dict]:
    """-> [{name, header_offset, comp_size, uncomp_size, method}] with ZIP64 extras."""
    entries = []
    p = 0
    n = len(cd)
    while p + 46 <= n and cd[p:p + 4] == CEN_SIG:
        method = struct.unpack_from("<H", cd, p + 10)[0]
        comp_size = struct.unpack_from("<I", cd, p + 20)[0]
        uncomp_size = struct.unpack_from("<I", cd, p + 24)[0]
        name_len = struct.unpack_from("<H", cd, p + 28)[0]
        extra_len = struct.unpack_from("<H", cd, p + 30)[0]
        cmt_len = struct.unpack_from("<H", cd, p + 32)[0]
        hdr_off = struct.unpack_from("<I", cd, p + 42)[0]
        name = cd[p + 46:p + 46 + name_len].decode("utf-8", "replace")

        # ZIP64 extended information extra field (0x0001) overrides the 0xFFFFFFFF
        # placeholders, in a fixed order, present only for the fields that overflowed.
        ex = cd[p + 46 + name_len:p + 46 + name_len + extra_len]
        q = 0
        while q + 4 <= len(ex):
            hid, hsz = struct.unpack_from("<HH", ex, q)
            if hid == 0x0001:
                r = q + 4
                if uncomp_size == 0xFFFFFFFF:
                    uncomp_size = struct.unpack_from("<Q", ex, r)[0]; r += 8
                if comp_size == 0xFFFFFFFF:
                    comp_size = struct.unpack_from("<Q", ex, r)[0]; r += 8
                if hdr_off == 0xFFFFFFFF:
                    hdr_off = struct.unpack_from("<Q", ex, r)[0]; r += 8
                break
            q += 4 + hsz

        entries.append({
            "name": name, "header_offset": hdr_off,
            "comp_size": comp_size, "uncomp_size": uncomp_size, "method": method,
        })
        p += 46 + name_len + extra_len + cmt_len
    return entries


def extract_entry(url: str, entry: dict, out_path: Path, tmp: Path) -> Path:
    """Fetch one member: local header (to learn its variable-length fields) + data."""
    ho = entry["header_offset"]
    fetch_range(url, ho, ho + 29, tmp)
    lh = tmp.read_bytes()
    if lh[:4] != b"PK\x03\x04":
        raise ValueError(f"Bad local header for {entry['name']}")
    name_len = struct.unpack_from("<H", lh, 26)[0]
    extra_len = struct.unpack_from("<H", lh, 28)[0]
    data_start = ho + 30 + name_len + extra_len
    fetch_range(url, data_start, data_start + entry["comp_size"] - 1, tmp)
    raw = tmp.read_bytes()
    data = raw if entry["method"] == 0 else zlib.decompress(raw, -15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


if __name__ == "__main__":
    url, size = sys.argv[1], int(sys.argv[2])
    here = Path(__file__).parent
    tail_len = 65536
    tail_off = size - tail_len
    tail = (here / "tail.bin").read_bytes()
    cd_off, cd_size = parse_eocd(tail, tail_off)
    print(f"central directory: offset={cd_off} size={cd_size} ({cd_size/1e6:.1f} MB)")
    cd_path = here / "cd.bin"
    if not cd_path.exists() or cd_path.stat().st_size != cd_size:
        fetch_range(url, cd_off, cd_off + cd_size - 1, cd_path)
    entries = parse_central_directory(cd_path.read_bytes())
    print(f"entries: {len(entries)}")
    import json
    (here / "entries.json").write_text(json.dumps(entries))
