"""A tiny, valid MP4, made here: the fixture ``tiny.mp4`` (PRD-251 US-202).

Two 32x32 frames of H.264 Constrained Baseline, one second each. Every frame is
an IDR picture of I_PCM macroblocks, whose samples go into the stream as they
are, so the encoder is a few lines of bit writing and needs no codec library.
The frames sit in an ISO-BMFF file: ftyp, then moov (the index), then mdat.
It holds nobody else's content, so it carries no licence.

Rewrite the fixture from ``orchestrator/``::

    python -m tests.fixtures.tiny_mp4
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Sequence

FIXTURE = Path(__file__).with_name("tiny.mp4")

WIDTH = HEIGHT = 32
MB = 16  # a macroblock is 16x16 luma samples
MACROBLOCKS = (WIDTH // MB) * (HEIGHT // MB)
LUMA_LEVELS = (0x60, 0xB0)  # one flat grey per frame
CHROMA_LEVEL = 0x80  # no colour
TIMESCALE = 1000
FRAME_TICKS = 1000  # one second a frame

PROFILE_IDC = 66  # Baseline
CONSTRAINT_FLAGS = 0xC0  # constraint_set0 and constraint_set1: Constrained Baseline
LEVEL_IDC = 30
NAL_REF_IDC = 3
NAL_SPS, NAL_PPS, NAL_IDR = 7, 8, 5
SLICE_TYPE_I = 7  # every slice of the picture is I
MB_TYPE_I_PCM = 25
LENGTH_BYTES = 4  # the NAL length prefix in a sample (avcC lengthSizeMinusOne = 3)

LANGUAGE_UND = 0x55C4  # "und", packed ISO-639-2/T
UNITY_MATRIX = struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
DPI_72 = 0x00480000


class _Bits:
    """An RBSP writer: fixed-width fields and exp-Golomb codes, most significant bit first."""

    def __init__(self) -> None:
        self._bits: List[int] = []

    def u(self, width: int, value: int) -> "_Bits":
        self._bits.extend((value >> shift) & 1 for shift in reversed(range(width)))
        return self

    def ue(self, value: int) -> "_Bits":
        code = value + 1
        return self.u(code.bit_length() - 1, 0).u(code.bit_length(), code)

    def se(self, value: int) -> "_Bits":
        return self.ue(2 * value - 1 if value > 0 else -2 * value)

    def align(self) -> "_Bits":
        return self.u(-len(self._bits) % 8, 0)

    def raw(self, data: bytes) -> "_Bits":
        for byte in data:
            self.u(8, byte)
        return self

    def rbsp(self) -> bytes:
        """The bytes, closed with the rbsp_stop_one_bit and zero alignment."""
        bits = self.u(1, 1).align()._bits
        return bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))


def _escaped(rbsp: bytes) -> bytes:
    """Emulation prevention: a 0x03 after two zero bytes that precede 0x00..0x03."""
    out = bytearray()
    zeros = 0
    for byte in rbsp:
        if zeros >= 2 and byte <= 3:
            out.append(3)
            zeros = 0
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def _nal(unit_type: int, rbsp: bytes) -> bytes:
    return bytes([(NAL_REF_IDC << 5) | unit_type]) + _escaped(rbsp)


def _sps() -> bytes:
    bits = _Bits().u(8, PROFILE_IDC).u(8, CONSTRAINT_FLAGS).u(8, LEVEL_IDC)
    bits.ue(0)  # seq_parameter_set_id
    bits.ue(0)  # log2_max_frame_num_minus4
    bits.ue(0)  # pic_order_cnt_type
    bits.ue(0)  # log2_max_pic_order_cnt_lsb_minus4
    bits.ue(1)  # max_num_ref_frames
    bits.u(1, 0)  # gaps_in_frame_num_value_allowed_flag
    bits.ue(WIDTH // MB - 1).ue(HEIGHT // MB - 1)  # pic_width_in_mbs_minus1, pic_height_in_map_units_minus1
    bits.u(1, 1)  # frame_mbs_only_flag
    bits.u(1, 1)  # direct_8x8_inference_flag
    bits.u(1, 0)  # frame_cropping_flag
    bits.u(1, 0)  # vui_parameters_present_flag
    return _nal(NAL_SPS, bits.rbsp())


def _pps() -> bytes:
    bits = _Bits().ue(0).ue(0)  # pic_parameter_set_id, seq_parameter_set_id
    bits.u(1, 0)  # entropy_coding_mode_flag: CAVLC
    bits.u(1, 0)  # bottom_field_pic_order_in_frame_present_flag
    bits.ue(0)  # num_slice_groups_minus1
    bits.ue(0).ue(0)  # num_ref_idx_l0/l1_default_active_minus1
    bits.u(1, 0).u(2, 0)  # weighted_pred_flag, weighted_bipred_idc
    bits.se(0).se(0).se(0)  # pic_init_qp_minus26, pic_init_qs_minus26, chroma_qp_index_offset
    bits.u(1, 0)  # deblocking_filter_control_present_flag
    bits.u(1, 0)  # constrained_intra_pred_flag
    bits.u(1, 0)  # redundant_pic_cnt_present_flag
    return _nal(NAL_PPS, bits.rbsp())


def _idr(idr_pic_id: int, luma: int) -> bytes:
    """One picture: a slice header, then every macroblock as I_PCM samples (4:2:0)."""
    bits = _Bits().ue(0).ue(SLICE_TYPE_I).ue(0)  # first_mb_in_slice, slice_type, pic_parameter_set_id
    bits.u(4, 0)  # frame_num (4 bits: log2_max_frame_num_minus4 = 0)
    bits.ue(idr_pic_id)  # differs between consecutive IDR pictures
    bits.u(4, 0)  # pic_order_cnt_lsb
    bits.u(1, 0).u(1, 0)  # dec_ref_pic_marking: no_output_of_prior_pics_flag, long_term_reference_flag
    bits.se(0)  # slice_qp_delta
    samples = bytes([luma]) * (MB * MB) + bytes([CHROMA_LEVEL]) * (MB * MB // 2)
    for _ in range(MACROBLOCKS):
        bits.ue(MB_TYPE_I_PCM).align().raw(samples)
    return _nal(NAL_IDR, bits.rbsp())


def _box(kind: bytes, *parts: bytes) -> bytes:
    payload = b"".join(parts)
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _full_box(kind: bytes, version: int, flags: int, *parts: bytes) -> bytes:
    return _box(kind, struct.pack(">I", (version << 24) | flags), *parts)


def _avc1(sps: bytes, pps: bytes) -> bytes:
    avcc = _box(
        b"avcC",
        bytes([1, sps[1], sps[2], sps[3], 0xFC | (LENGTH_BYTES - 1), 0xE0 | 1]),
        struct.pack(">H", len(sps)), sps,
        bytes([1]), struct.pack(">H", len(pps)), pps,
    )
    return _box(
        b"avc1",
        bytes(6), struct.pack(">H", 1),  # reserved, data_reference_index
        bytes(16),  # pre_defined, reserved, pre_defined[3]
        struct.pack(">HHII", WIDTH, HEIGHT, DPI_72, DPI_72),
        bytes(4), struct.pack(">H", 1),  # reserved, frame_count
        bytes(32),  # compressorname
        struct.pack(">Hh", 0x0018, -1),  # depth, pre_defined
        avcc,
    )


def _stbl(sps: bytes, pps: bytes, sizes: Sequence[int], chunk_offset: int) -> bytes:
    count = len(sizes)
    return _box(
        b"stbl",
        _full_box(b"stsd", 0, 0, struct.pack(">I", 1), _avc1(sps, pps)),
        _full_box(b"stts", 0, 0, struct.pack(">III", 1, count, FRAME_TICKS)),
        _full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, count, 1)),
        _full_box(b"stsz", 0, 0, struct.pack(">II", 0, count), *(struct.pack(">I", size) for size in sizes)),
        _full_box(b"stco", 0, 0, struct.pack(">II", 1, chunk_offset)),
    )


def _moov(sps: bytes, pps: bytes, sizes: Sequence[int], chunk_offset: int) -> bytes:
    duration = FRAME_TICKS * len(sizes)
    mvhd = _full_box(
        b"mvhd", 0, 0,
        struct.pack(">IIIIIH", 0, 0, TIMESCALE, duration, 0x00010000, 0x0100),  # times, scale, rate, volume
        bytes(10), UNITY_MATRIX, bytes(24), struct.pack(">I", 2),  # reserved, matrix, pre_defined, next_track_ID
    )
    tkhd = _full_box(
        b"tkhd", 0, 0x000003,  # enabled, in the movie
        struct.pack(">IIIII", 0, 0, 1, 0, duration), bytes(8),  # times, track_ID 1, reserved, duration
        struct.pack(">hhhH", 0, 0, 0, 0), UNITY_MATRIX,  # layer, alternate_group, volume, reserved
        struct.pack(">II", WIDTH << 16, HEIGHT << 16),
    )
    mdhd = _full_box(b"mdhd", 0, 0, struct.pack(">IIIIHH", 0, 0, TIMESCALE, duration, LANGUAGE_UND, 0))
    hdlr = _full_box(b"hdlr", 0, 0, struct.pack(">I4s", 0, b"vide"), bytes(12), b"VideoHandler\x00")
    vmhd = _full_box(b"vmhd", 0, 1, bytes(8))
    dinf = _box(b"dinf", _full_box(b"dref", 0, 0, struct.pack(">I", 1), _full_box(b"url ", 0, 1)))
    minf = _box(b"minf", vmhd, dinf, _stbl(sps, pps, sizes, chunk_offset))
    return _box(b"moov", mvhd, _box(b"trak", tkhd, _box(b"mdia", mdhd, hdlr, minf)))


def tiny_mp4() -> bytes:
    """The fixture's bytes: ftyp, moov, then mdat holding the frames (one chunk)."""
    sps, pps = _sps(), _pps()
    frames = [_idr(index, luma) for index, luma in enumerate(LUMA_LEVELS)]
    samples = [struct.pack(">I", len(frame)) + frame for frame in frames]
    sizes = [len(sample) for sample in samples]
    ftyp = _box(b"ftyp", b"isom", struct.pack(">I", 0x200), b"isom", b"iso2", b"avc1", b"mp41")
    # The chunk offset is a fixed-width field, so moov's size does not depend on it.
    chunk_offset = len(ftyp) + len(_moov(sps, pps, sizes, 0)) + 8
    return ftyp + _moov(sps, pps, sizes, chunk_offset) + _box(b"mdat", *samples)


if __name__ == "__main__":
    FIXTURE.write_bytes(tiny_mp4())
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")
