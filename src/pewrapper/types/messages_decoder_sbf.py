import struct
import sys


SBF_SYNCH_BYTES = [0x24, 0x40]


def decode_msg_endiannes(msg: bytes, fmt: str, is_stream_little_endian: bool) -> int:
    def is_machine_little_endian() -> bool:
        return sys.byteorder == "little"

    endian_prefix = (
        "<" if is_machine_little_endian() == is_stream_little_endian else ">"
    )
    endian_fmt = endian_prefix + fmt
    size = struct.calcsize(endian_fmt)

    raw_data = msg[:size]

    return struct.unpack(endian_fmt, raw_data)[0]
