"""Minimal MCAP + CDR reader: enough to ingest MARS-LVIG, and no more.

**Why this exists rather than a dependency.** The MARS-LVIG sequences are published as ROS 1
bags on Google Drive and mirrored as ROS 2 / MCAP on HuggingFace (`LIT-006`). The MCAP mirror is
the practical source because HuggingFace serves HTTP byte ranges and MCAP is index-first: a
Summary section at the end of the file lists every channel plus a per-chunk byte offset and time
span, so a 180 s window can be pulled out of a 20 GB file without downloading the rest.

Reading it needs two things -- MCAP record framing and CDR (ROS 2) deserialisation of five
message types. That is ~150 lines, versus adding `mcap` + `mcap-ros2-support` + a ROS message
definition stack to a package that `research.md` R4 deliberately keeps at numpy + matplotlib.
`DEC-002`'s file boundary is the reason this is a reasonable trade: the evaluator only ever needs
to *read* a handful of well-known fixed-layout messages, never to speak ROS.

**Scope:** the five message types MARS-LVIG's ground truth and imagery use. Anything else raises.
This is not a general MCAP library and must not grow into one.

Verified 2026-08-13 against `mars_lvig/HKairport01/HKairport01_0.mcap` (20.44 GB, 15,403 chunks,
32 channels): index read in 4.31 MB, all topics decoded, camera cadence measured at exactly
10.00 Hz. See research log 2026-08-13.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

MAGIC = b"\x89MCAP0\r\n"

OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_CHUNK = 0x06
OP_CHUNK_INDEX = 0x08
OP_STATISTICS = 0x0B
OP_FOOTER = 0x02

# Byte range fetcher: (start, end_inclusive) -> bytes.
Fetcher = Callable[[int, int], bytes]


class McapError(ValueError):
    """Malformed MCAP, or a message type this reader deliberately does not support."""


class _Cur:
    """Little-endian cursor. MCAP's own framing is always little-endian."""

    __slots__ = ("b", "p")

    def __init__(self, b: bytes, p: int = 0):
        self.b, self.p = b, p

    def u8(self) -> int:
        v = self.b[self.p]; self.p += 1; return v

    def u16(self) -> int:
        v = struct.unpack_from("<H", self.b, self.p)[0]; self.p += 2; return v

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.b, self.p)[0]; self.p += 4; return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.b, self.p)[0]; self.p += 8; return v

    def s(self) -> str:
        n = self.u32()
        v = self.b[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return v

    def take(self, n: int) -> bytes:
        v = self.b[self.p:self.p + n]; self.p += n; return v


@dataclass(frozen=True)
class Channel:
    id: int
    schema_id: int
    topic: str
    message_encoding: str


@dataclass(frozen=True)
class ChunkIndex:
    start_time: int          # ns
    end_time: int            # ns
    chunk_start_offset: int  # byte offset of the Chunk record
    chunk_length: int
    compression: str
    compressed_size: int
    uncompressed_size: int
    channel_ids: tuple


def read_footer(fetch_tail: Callable[[int], bytes]) -> tuple[int, int]:
    """-> (summary_start, summary_offset_start). `fetch_tail(n)` returns the last n bytes."""
    tail = fetch_tail(64)
    if len(tail) < 37 or tail[-8:] != MAGIC:
        raise McapError(f"Not an MCAP file: trailing magic is {tail[-8:]!r}")
    p = len(tail) - 8 - 20 - 8 - 1      # magic, payload, length, opcode
    if tail[p] != OP_FOOTER:
        raise McapError(f"Expected Footer opcode {OP_FOOTER:#x}, got {tail[p]:#x}")
    c = _Cur(tail, p + 9)
    return c.u64(), c.u64()


def parse_summary(buf: bytes) -> tuple[dict, list, Optional[dict]]:
    """Parse the Summary section -> (channels_by_id, chunk_indexes, statistics|None)."""
    channels: dict = {}
    chunks: list = []
    stats: Optional[dict] = None
    c = _Cur(buf)
    n = len(buf)
    while c.p + 9 <= n:
        op = c.u8()
        ln = c.u64()
        end = c.p + ln
        if end > n:
            break
        if op == OP_CHANNEL:
            r = _Cur(buf, c.p)
            cid, sid, topic, enc = r.u16(), r.u16(), r.s(), r.s()
            channels[cid] = Channel(cid, sid, topic, enc)
        elif op == OP_CHUNK_INDEX:
            r = _Cur(buf, c.p)
            st, et, off, clen = r.u64(), r.u64(), r.u64(), r.u64()
            mio_len = r.u32()
            stop = r.p + mio_len
            ids = []
            while r.p < stop:
                ids.append(r.u16()); r.u64()
            r.p = stop
            r.u64()                                   # message_index_length
            comp = r.s()
            csz, usz = r.u64(), r.u64()
            chunks.append(ChunkIndex(st, et, off, clen, comp, csz, usz, tuple(ids)))
        elif op == OP_STATISTICS:
            r = _Cur(buf, c.p)
            stats = {
                "message_count": r.u64(), "schema_count": r.u16(),
                "channel_count": r.u32(), "attachment_count": r.u32(),
                "metadata_count": r.u32(), "chunk_count": r.u32(),
                "message_start_time": r.u64(), "message_end_time": r.u64(),
            }
            cnt = r.u32(); stop = r.p + cnt; per = {}
            while r.p < stop:
                per[r.u16()] = r.u64()
            stats["channel_message_counts"] = per
        c.p = end
    return channels, chunks, stats


def chunks_overlapping(chunk_indexes: list, start_ns: int, end_ns: int) -> list:
    """Chunk indexes whose time span intersects [start_ns, end_ns]. This is what makes a
    time-windowed read possible; chunks are time-ordered but channel-interleaved, so a window
    still carries topics we discard."""
    return [c for c in chunk_indexes if c.end_time >= start_ns and c.start_time <= end_ns]


def decode_chunk(payload: bytes) -> bytes:
    """Chunk record payload -> its uncompressed inner records."""
    c = _Cur(payload)
    c.u64(); c.u64(); c.u64(); c.u32()          # start, end, uncompressed_size, crc
    comp = c.s()
    rec_len = c.u64()
    data = c.take(rec_len)
    if comp == "":
        return data
    if comp == "zstd":
        try:
            import zstandard
        except ImportError as exc:                                  # pragma: no cover
            raise McapError("chunk is zstd-compressed; `zstandard` not installed") from exc
        return zstandard.ZstdDecompressor().decompressobj().decompress(data)
    if comp == "lz4":
        try:
            import lz4.frame
        except ImportError as exc:                                  # pragma: no cover
            raise McapError("chunk is lz4-compressed; `lz4` not installed") from exc
        return lz4.frame.decompress(data)
    raise McapError(f"Unsupported chunk compression {comp!r}")


def iter_chunk_messages(inner: bytes, want: set) -> Iterator[tuple]:
    """Yield (channel_id, log_time_ns, publish_time_ns, payload) for channels in `want`."""
    c = _Cur(inner)
    n = len(inner)
    while c.p + 9 <= n:
        op = c.u8()
        ln = c.u64()
        end = c.p + ln
        if end > n:
            break
        if op == OP_MESSAGE:
            r = _Cur(inner, c.p)
            cid = r.u16(); r.u32()                  # channel_id, sequence
            lt, pt = r.u64(), r.u64()
            if cid in want:
                yield cid, lt, pt, inner[r.p:end]
        c.p = end


# --------------------------------------------------------------------------------------
# CDR (ROS 2) decoding -- only the fixed-layout types MARS-LVIG uses.
# --------------------------------------------------------------------------------------


class CDR:
    """XCDR1 reader. Primitives are aligned to their own size, relative to the start of the
    body (i.e. after the 4-byte encapsulation header) -- getting that base wrong silently
    shifts every subsequent field, so it is explicit here."""

    __slots__ = ("b", "little", "o", "base")

    def __init__(self, b: bytes):
        if len(b) < 4:
            raise McapError("CDR payload shorter than its encapsulation header")
        self.b = b
        self.little = b[1] in (0x01, 0x03)
        self.base = 4
        self.o = 4

    def _align(self, n: int) -> None:
        self.o += (-(self.o - self.base)) % n

    def _unpack(self, fmt: str, n: int):
        self._align(n)
        v = struct.unpack_from(("<" if self.little else ">") + fmt, self.b, self.o)[0]
        self.o += n
        return v

    def i8(self) -> int:
        v = struct.unpack_from("b", self.b, self.o)[0]; self.o += 1; return v

    def u8(self) -> int:
        v = self.b[self.o]; self.o += 1; return v

    def i16(self) -> int: return self._unpack("h", 2)
    def u16(self) -> int: return self._unpack("H", 2)
    def i32(self) -> int: return self._unpack("i", 4)
    def u32(self) -> int: return self._unpack("I", 4)
    def f32(self) -> float: return self._unpack("f", 4)
    def f64(self) -> float: return self._unpack("d", 8)

    def string(self) -> str:
        n = self.u32()
        if n == 0:
            return ""
        v = self.b[self.o:self.o + n - 1].decode("utf-8", "replace")
        self.o += n
        return v

    def header(self) -> tuple:
        """std_msgs/Header -> (stamp_seconds_float, frame_id)."""
        sec = self.i32()
        nsec = self.u32()
        return sec + nsec * 1e-9, self.string()


def dec_int16(payload: bytes) -> int:
    """std_msgs/msg/Int16."""
    return CDR(payload).i16()


def dec_uint8(payload: bytes) -> int:
    """std_msgs/msg/UInt8."""
    return CDR(payload).u8()


def dec_float32(payload: bytes) -> float:
    """std_msgs/msg/Float32."""
    return CDR(payload).f32()


def dec_navsatfix(payload: bytes) -> dict:
    """sensor_msgs/msg/NavSatFix.

    Note `status`/`position_covariance` are NOT populated by the DJI OSDK bridge in MARS-LVIG
    (verified 2026-08-13: status=0, covariance all zero, covariance_type=0=UNKNOWN). The real
    fix quality lives in the separate `/dji_osdk_ros/rtk_info_position` topic, so an adapter
    that judged quality from this message alone would silently under-read it.
    """
    c = CDR(payload)
    stamp, frame_id = c.header()
    status = c.i8()
    service = c.u16()
    lat, lon, alt = c.f64(), c.f64(), c.f64()
    cov = [c.f64() for _ in range(9)]
    cov_type = c.u8()
    return {"stamp": stamp, "frame_id": frame_id, "status": status, "service": service,
            "lat": lat, "lon": lon, "alt": alt, "position_covariance": cov,
            "position_covariance_type": cov_type}


def dec_compressed_image(payload: bytes) -> dict:
    """sensor_msgs/msg/CompressedImage. `data` is the original JPEG bytes, returned unchanged
    so an adapter can write them without a re-encode."""
    c = CDR(payload)
    stamp, frame_id = c.header()
    fmt = c.string()
    n = c.u32()
    return {"stamp": stamp, "frame_id": frame_id, "format": fmt,
            "data": payload[c.o:c.o + n]}


def dec_quaternion_stamped(payload: bytes) -> dict:
    """geometry_msgs/msg/QuaternionStamped."""
    c = CDR(payload)
    stamp, frame_id = c.header()
    x, y, z, w = c.f64(), c.f64(), c.f64(), c.f64()
    return {"stamp": stamp, "frame_id": frame_id, "x": x, "y": y, "z": z, "w": w}


def quaternion_to_compass_deg(x: float, y: float, z: float, w: float) -> float:
    """FLU-body -> ENU quaternion (DJI OSDK `/dji_osdk_ros/attitude`, frame_id `body_FLU`)
    to canonical compass heading, degrees clockwise from North in [0, 360) (`DEC-004`).

    The ENU-frame yaw is counter-clockwise from East; compass is clockwise from North, so the
    conversion is `90 - yaw_enu`. Verified empirically against direction of travel on four
    straight legs in different quadrants (research log 2026-08-13): the residual was
    -1.95 deg +/- 1.62, consistent with crab rather than a convention error.
    """
    yaw_enu = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return (90.0 - yaw_enu) % 360.0


def quaternion_to_roll_pitch_deg(x: float, y: float, z: float, w: float) -> tuple[float, float]:
    """FLU-body -> ENU quaternion to (roll_deg, pitch_deg), the airframe's tilt about its own
    forward and right axes. Companion to `quaternion_to_compass_deg`, which extracts the yaw
    component of the same message; added 2026-08-14 for `EXP-003`.

    Standard ZYX (yaw-pitch-roll) intrinsic decomposition. Roll is rotation about the body
    forward (X, FLU) axis, pitch about the body left (Y) axis. Both are returned in degrees in
    [-180, 180] and [-90, 90] respectively, signed -- `EXP-003` takes magnitudes at analysis
    time, but the raw signed values are what get written to the sidecar so a later question
    about *direction* of tilt is still answerable without re-fetching.

    Pitch is clamped rather than allowed to produce a domain error at gimbal lock: the argument
    to `asin` can exceed 1 by a floating-point epsilon for a near-vertical attitude. Gimbal lock
    is not reachable for the level cruise this is used on (`LIT-006` measured |pitch| <= 6.66
    deg), so the clamp is a numerical guard, not a modelling choice.

    NOT independently calibrated. DJI's onboard EKF solution with no published accuracy figure
    and no independent attitude reference in this dataset (`EXP-003` threat 7). Adequate as a
    relative signal for rank correlation; not adequate for an absolute tilt claim.
    """
    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = max(-1.0, min(1.0, sin_pitch))
    pitch = math.asin(sin_pitch)

    return math.degrees(roll), math.degrees(pitch)


def tilt_from_roll_pitch_deg(roll_deg: float, pitch_deg: float) -> float:
    """Combined tilt magnitude: the true angle between the camera boresight and nadir, in [0, 180].

    `acos(cos roll * cos pitch)`, NOT `sqrt(roll^2 + pitch^2)` -- the latter is only the
    small-angle approximation of it. At this dataset's magnitudes the two differ by under
    0.02 deg (measured: 0.0165 deg at the observed extreme of roll 9.06, pitch 6.66), far below
    anything `EXP-003`'s rank correlations could resolve, so the choice changes no result here;
    it is made correctly anyway so the function stays valid for a dataset with real excursions.
    """
    c = math.cos(math.radians(roll_deg)) * math.cos(math.radians(pitch_deg))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# `/dji_osdk_ros/rtk_yaw` does NOT report a compass heading. Measured on HKairport01
# (2026-08-13, four straight legs in four quadrants, 52 samples):
#
#     rtk_yaw - compass_heading_from_FC_attitude = +269.16 deg +/- 0.64
#
# so the correction applied here is its inverse, -269.16 deg, equivalently **+90.84 deg** -- a
# quarter turn. The most likely physical explanation is that the dual-antenna RTK baseline is
# mounted across the airframe rather than along it, so the reported yaw is the baseline's
# heading, not the nose's; MARS-LVIG Figure 1 does show the RTK antennas on the UAV's sides.
# That explanation is an INFERENCE and is not documented by the dataset authors.
#
# Using rtk_yaw as if it were already a compass heading would be wrong by ~90 deg while looking
# entirely plausible -- precisely the failure `DEC-004` exists to prevent, and the reason the
# convention was resolved empirically rather than from the field name.
#
# Sequence-specific until re-verified. It is NOT a constant of the dataset or of the platform.
HKAIRPORT01_RTK_YAW_TO_COMPASS_OFFSET_DEG = -269.16


def rtk_yaw_to_compass_deg(rtk_yaw_deg: float, offset_deg: float) -> float:
    """Convert `/dji_osdk_ros/rtk_yaw` (integer degrees) to canonical compass heading.

    `offset_deg` MUST come from an empirical verification for the sequence in question -- there
    is no documented convention to fall back on, and the field name is not evidence.
    """
    return (rtk_yaw_deg + offset_deg) % 360.0
