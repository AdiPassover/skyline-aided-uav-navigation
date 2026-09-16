"""Tests for `naveval.mcap_reader`, built entirely from constructed bytes.

The real source is a 20.4 GB remote file, so everything here is synthesised: MCAP records are
assembled field by field and CDR payloads are packed with `struct`. That is the right level to
test at -- the risky parts of this reader are record framing and CDR *alignment*, both of which
fail silently by shifting every subsequent field rather than by raising.

Three of the constants asserted below are pinned to values measured against the real
HKairport01 MCAP on 2026-08-13 (research log), so a regression in the decoders shows up as a
disagreement with a recorded measurement rather than only as a self-consistent unit test.
"""

from __future__ import annotations

import math
import struct

import pytest

from naveval.mcap_reader import (
    HKAIRPORT01_RTK_YAW_TO_COMPASS_OFFSET_DEG,
    MAGIC,
    OP_CHANNEL,
    OP_CHUNK,
    OP_CHUNK_INDEX,
    OP_FOOTER,
    OP_MESSAGE,
    OP_STATISTICS,
    McapError,
    chunks_overlapping,
    dec_compressed_image,
    dec_float32,
    dec_int16,
    dec_navsatfix,
    dec_quaternion_stamped,
    dec_uint8,
    decode_chunk,
    iter_chunk_messages,
    parse_summary,
    quaternion_to_compass_deg,
    read_footer,
    rtk_yaw_to_compass_deg,
)

# ---------------------------------------------------------------- builders


def rec(op: int, payload: bytes) -> bytes:
    return bytes([op]) + struct.pack("<Q", len(payload)) + payload


def mcap_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<I", len(b)) + b


def channel_rec(cid: int, topic: str, enc: str = "cdr") -> bytes:
    p = struct.pack("<HH", cid, 0) + mcap_str(topic) + mcap_str(enc) + struct.pack("<I", 0)
    return rec(OP_CHANNEL, p)


def chunk_index_rec(start, end, offset, length, channel_ids) -> bytes:
    mio = b"".join(struct.pack("<HQ", c, 0) for c in channel_ids)
    p = (struct.pack("<QQQQ", start, end, offset, length)
         + struct.pack("<I", len(mio)) + mio
         + struct.pack("<Q", 0) + mcap_str("") + struct.pack("<QQ", length, length))
    return rec(OP_CHUNK_INDEX, p)


def message_rec(cid: int, log_time: int, payload: bytes) -> bytes:
    return rec(OP_MESSAGE, struct.pack("<HIQQ", cid, 0, log_time, log_time) + payload)


def chunk_rec(messages: bytes) -> bytes:
    p = (struct.pack("<QQQ", 0, 0, len(messages)) + struct.pack("<I", 0)
         + mcap_str("") + struct.pack("<Q", len(messages)) + messages)
    return rec(OP_CHUNK, p)


def cdr(body: bytes) -> bytes:
    """Little-endian CDR encapsulation header + body."""
    return b"\x00\x01\x00\x00" + body


class CdrW:
    """CDR *writer* mirroring the reader's alignment rules.

    Written deliberately rather than hand-packing: a builder that forgets padding produces a
    test that fails against a correct reader, which is exactly the confusion this class avoids.
    Offsets are tracked relative to the start of the body, as the encapsulation header is 4
    bytes and alignment is measured from after it.
    """

    def __init__(self):
        self.b = bytearray()

    def _align(self, n: int) -> None:
        self.b += b"\x00" * ((-len(self.b)) % n)

    def _p(self, fmt: str, n: int, v) -> "CdrW":
        self._align(n)
        self.b += struct.pack("<" + fmt, v)
        return self

    def i8(self, v): self.b += struct.pack("<b", v); return self
    def u8(self, v): self.b += struct.pack("<B", v); return self
    def i16(self, v): return self._p("h", 2, v)
    def u16(self, v): return self._p("H", 2, v)
    def i32(self, v): return self._p("i", 4, v)
    def u32(self, v): return self._p("I", 4, v)
    def f32(self, v): return self._p("f", 4, v)
    def f64(self, v): return self._p("d", 8, v)

    def string(self, s: str) -> "CdrW":
        raw = s.encode() + b"\x00"
        self.u32(len(raw))
        self.b += raw
        return self

    def header(self, sec: int, nsec: int, frame_id: str = "") -> "CdrW":
        return self.i32(sec).u32(nsec).string(frame_id)

    def bytes_seq(self, data: bytes) -> "CdrW":
        self.u32(len(data))
        self.b += data
        return self

    def done(self) -> bytes:
        return cdr(bytes(self.b))


# ---------------------------------------------------------------- footer


class TestFooter:
    def _file_tail(self, summary_start: int, summary_offset_start: int) -> bytes:
        payload = struct.pack("<QQI", summary_start, summary_offset_start, 0)
        return b"\x00" * 16 + rec(OP_FOOTER, payload) + MAGIC

    def test_reads_summary_offsets(self):
        tail = self._file_tail(123456789, 987654321)
        got = read_footer(lambda n: tail[-n:])
        assert got == (123456789, 987654321)

    def test_rejects_missing_magic(self):
        tail = self._file_tail(1, 2)[:-8] + b"NOTMAGIC"
        with pytest.raises(McapError, match="trailing magic"):
            read_footer(lambda n: tail[-n:])

    def test_rejects_wrong_footer_opcode(self):
        payload = struct.pack("<QQI", 1, 2, 0)
        tail = b"\x00" * 16 + rec(0x07, payload) + MAGIC
        with pytest.raises(McapError, match="Footer opcode"):
            read_footer(lambda n: tail[-n:])


# ---------------------------------------------------------------- summary


class TestParseSummary:
    def test_parses_channels_chunk_indexes_and_statistics(self):
        stats_p = (struct.pack("<Q", 999) + struct.pack("<H", 1)
                   + struct.pack("<III", 2, 0, 0) + struct.pack("<I", 3)
                   + struct.pack("<QQ", 10, 20) + struct.pack("<I", 0))
        buf = (channel_rec(7, "/left_camera/image/compressed")
               + channel_rec(22, "/dji_osdk_ros/rtk_yaw")
               + chunk_index_rec(10, 15, 1000, 500, [7, 22])
               + chunk_index_rec(15, 20, 1500, 500, [7])
               + rec(OP_STATISTICS, stats_p))
        channels, chunks, stats = parse_summary(buf)
        assert {c.topic for c in channels.values()} == {
            "/left_camera/image/compressed", "/dji_osdk_ros/rtk_yaw"}
        assert channels[22].message_encoding == "cdr"
        assert len(chunks) == 2
        assert chunks[0].chunk_start_offset == 1000
        assert chunks[0].channel_ids == (7, 22)
        assert stats["message_count"] == 999
        assert stats["chunk_count"] == 3

    def test_unknown_records_are_skipped_not_fatal(self):
        buf = channel_rec(1, "/a") + rec(0x7E, b"\x01\x02\x03") + channel_rec(2, "/b")
        channels, _, _ = parse_summary(buf)
        assert len(channels) == 2

    def test_truncated_trailing_record_is_ignored(self):
        buf = channel_rec(1, "/a") + b"\x04" + struct.pack("<Q", 9999)
        channels, _, _ = parse_summary(buf)
        assert len(channels) == 1


class TestChunkSelection:
    def _idx(self):
        return [chunk_index_rec(0, 10, 0, 1, [1]), chunk_index_rec(10, 20, 1, 1, [1]),
                chunk_index_rec(20, 30, 2, 1, [1])]

    def test_selects_only_overlapping_chunks(self):
        _, chunks, _ = parse_summary(b"".join(self._idx()))
        assert len(chunks_overlapping(chunks, 11, 12)) == 1
        assert len(chunks_overlapping(chunks, 5, 25)) == 3
        assert len(chunks_overlapping(chunks, 100, 200)) == 0

    def test_boundary_touching_chunk_is_included(self):
        _, chunks, _ = parse_summary(b"".join(self._idx()))
        assert len(chunks_overlapping(chunks, 10, 10)) == 2  # both touch t=10


# ---------------------------------------------------------------- chunk + messages


class TestChunkMessages:
    def test_roundtrip_uncompressed_chunk_filters_by_channel(self):
        msgs = (message_rec(7, 100, cdr(struct.pack("<h", 42)))
                + message_rec(22, 110, cdr(struct.pack("<h", -7)))
                + message_rec(99, 120, cdr(struct.pack("<h", 1))))
        full = chunk_rec(msgs)
        inner = decode_chunk(full[9:])
        got = list(iter_chunk_messages(inner, {7, 22}))
        assert [g[0] for g in got] == [7, 22]
        assert [g[1] for g in got] == [100, 110]
        assert dec_int16(got[1][3]) == -7

    def test_unsupported_compression_raises(self):
        p = (struct.pack("<QQQ", 0, 0, 4) + struct.pack("<I", 0)
             + mcap_str("brotli") + struct.pack("<Q", 0))
        with pytest.raises(McapError, match="Unsupported chunk compression"):
            decode_chunk(p)


# ---------------------------------------------------------------- CDR decoding


class TestCdrScalars:
    def test_int16_including_negative(self):
        assert dec_int16(cdr(struct.pack("<h", 353))) == 353
        assert dec_int16(cdr(struct.pack("<h", -12))) == -12

    def test_uint8_rtk_status_code(self):
        # 50 == "integer narrow-lane ambiguity solution" == RTK fixed (DJI OSDK)
        assert dec_uint8(cdr(struct.pack("<B", 50))) == 50

    def test_float32(self):
        assert dec_float32(cdr(struct.pack("<f", 1.5))) == pytest.approx(1.5)

    def test_rejects_payload_shorter_than_encapsulation_header(self):
        with pytest.raises(McapError, match="encapsulation header"):
            dec_int16(b"\x00\x01")


class TestCdrAlignment:
    """The failure mode this reader is most exposed to: a mis-aligned field shifts every
    subsequent one and produces plausible-looking wrong numbers rather than an exception."""

    @staticmethod
    def _navsatfix(frame_id, lat, lon, alt, stamp=(1671606517, 406000000)):
        w = CdrW().header(*stamp, frame_id).i8(0).u16(0)
        w.f64(lat).f64(lon).f64(alt)
        for _ in range(9):
            w.f64(0.0)
        return w.u8(0).done()

    def test_navsatfix_doubles_are_correctly_aligned_after_variable_length_frame_id(self):
        # frame_id length changes the padding before the first float64. Every variant must
        # decode to identical values; if alignment is wrong, most of these produce garbage.
        for frame_id in ["", "a", "ab", "abc", "abcd", "abcde", "abcdef", "abcdefg"]:
            got = dec_navsatfix(self._navsatfix(frame_id, 22.4160794, 114.0422944, 179.643))
            assert got["lat"] == pytest.approx(22.4160794), f"frame_id={frame_id!r}"
            assert got["lon"] == pytest.approx(114.0422944), f"frame_id={frame_id!r}"
            assert got["alt"] == pytest.approx(179.643), f"frame_id={frame_id!r}"
            assert got["frame_id"] == frame_id

    def test_navsatfix_stamp_matches_header(self):
        got = dec_navsatfix(self._navsatfix("", 1.0, 2.0, 3.0))
        assert got["stamp"] == pytest.approx(1671606517.406, abs=1e-3)

    def test_navsatfix_status_and_covariance_are_read_verbatim(self):
        """MARS-LVIG leaves these unpopulated (status=0, covariance all zero) -- the real
        quality is on /dji_osdk_ros/rtk_info_position. The decoder must report what is there,
        not infer anything."""
        got = dec_navsatfix(self._navsatfix("", 1.0, 2.0, 3.0))
        assert got["status"] == 0
        assert got["position_covariance"] == [0.0] * 9
        assert got["position_covariance_type"] == 0


class TestCompressedImage:
    @pytest.mark.parametrize("frame_id", ["", "cam", "camera_link"])
    def test_jpeg_payload_returned_unchanged(self, frame_id):
        jpeg = b"\xff\xd8\xff\xe0" + b"payload-bytes" + b"\xff\xd9"
        payload = CdrW().header(1, 0, frame_id).string("jpeg").bytes_seq(jpeg).done()
        got = dec_compressed_image(payload)
        assert got["format"] == "jpeg"
        assert got["data"] == jpeg          # byte-for-byte; no re-encode anywhere

    def test_quaternion_stamped_roundtrip(self):
        payload = CdrW().header(5, 0, "body_FLU").f64(0.1).f64(0.2).f64(0.3).f64(0.9).done()
        got = dec_quaternion_stamped(payload)
        assert got["frame_id"] == "body_FLU"
        assert (got["x"], got["y"], got["z"], got["w"]) == pytest.approx((0.1, 0.2, 0.3, 0.9))


# ---------------------------------------------------------------- heading conventions


class TestHeadingConventions:
    def test_identity_quaternion_is_east(self):
        # FLU body with zero rotation points along ENU +X = East = compass 90.
        assert quaternion_to_compass_deg(0, 0, 0, 1) == pytest.approx(90.0)

    def test_quarter_turn_ccw_in_enu_is_north(self):
        s = math.sin(math.pi / 4)
        h = quaternion_to_compass_deg(0, 0, s, s)
        assert abs((h + 180) % 360 - 180) < 1e-6   # 0 deg, tolerant of the 0/360 wrap

    def test_output_is_normalised_to_0_360(self):
        for ang in [0, math.pi / 3, -math.pi / 3, math.pi]:
            h = quaternion_to_compass_deg(0, 0, math.sin(ang / 2), math.cos(ang / 2))
            assert 0.0 <= h < 360.0

    @pytest.mark.parametrize("rtk_yaw,track_deg", [(176, 268.1), (353, 84.6), (290, 21.5)])
    def test_measured_rtk_yaw_offset_reproduces_track_heading(self, rtk_yaw, track_deg):
        """Pinned to the 2026-08-13 measurement on the real HKairport01 MCAP: three straight
        legs in three quadrants, RTK antenna-baseline yaw vs RTK-derived direction of travel.
        Tolerance is 2 deg, covering the 1 deg Int16 quantisation plus crab."""
        got = rtk_yaw_to_compass_deg(rtk_yaw, HKAIRPORT01_RTK_YAW_TO_COMPASS_OFFSET_DEG)
        diff = abs((got - track_deg + 180) % 360 - 180)
        assert diff < 2.0, f"rtk_yaw={rtk_yaw} -> {got:.2f}, track {track_deg}"

    def test_correction_is_a_quarter_turn(self):
        """The measured correction is +90.84 deg (equivalently -269.16), consistent with the
        RTK antenna baseline being mounted across the airframe rather than along it. Recorded
        as a guard: if this constant is ever edited to something far from a quarter turn the
        physical story no longer holds, and that needs re-verification rather than a tolerance
        bump."""
        wrapped = (HKAIRPORT01_RTK_YAW_TO_COMPASS_OFFSET_DEG + 180) % 360 - 180
        assert 85.0 < wrapped < 95.0, f"correction wrapped to {wrapped:.2f} deg"
