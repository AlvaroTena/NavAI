from typing import Tuple
from enum import Enum
from dataclasses import dataclass

from navutils.logger import Logger
from pewrapper.types.constants import BITS_IN_BYTE

MAX_GNSS_MESSAGE = 50


class GNSS_MESSAGE_TYPES(Enum):
    MESSAGE_TYPE_UNDEFINED = 0
    RTCM_1077 = 1
    RTCM_1097 = 2
    RTCM_1127 = 3
    RTCM_1074 = 4
    RTCM_1094 = 5
    RTCM_1124 = 6
    RTCM_1075 = 7
    RTCM_1095 = 8
    RTCM_1125 = 9
    UBX_RXM_RAWX = 10
    UBX_RXM_SFRBX = 11
    UBX_RXM_MEASX = 12
    UBX_MON_COMMS = 13
    UBX_MON_RF = 14
    UBX_ESF_MEAS = 15
    UBX_SUBX_MON = 16
    SBF_4024 = 17
    SBF_4027 = 18
    SBF_5891 = 19
    SBF_4002 = 20
    MAX_MESSAGES_TYPES = 21


class GNSS_MSG_PROTOCOL(Enum):
    PROTOCOL_UNDEFINED = 0
    RTCM = 1
    UBX = 2
    SBF = 3


@dataclass
class Submsg_Decode_Info:
    protocol = GNSS_MSG_PROTOCOL.PROTOCOL_UNDEFINED
    msg_type = GNSS_MESSAGE_TYPES.MESSAGE_TYPE_UNDEFINED
    is_available = False
    length_bytes = 0
    position_bytes = 0
    time_sync = 0.0


Msg_Decode_Info = list[Submsg_Decode_Info]


def fill_msg_decode_info(
    protocol: int,
    msg_type: int,
    len_bytes: int,
    pos_bytes: int,
    time_sync: float,
    index: int,
    msg_decode_info: Msg_Decode_Info,
) -> Tuple[int, Msg_Decode_Info]:
    if protocol == GNSS_MSG_PROTOCOL.PROTOCOL_UNDEFINED:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.WRAPPER,
            "Message protocol not supported",
        )
    elif msg_type == GNSS_MESSAGE_TYPES.MESSAGE_TYPE_UNDEFINED:
        Logger.log_message(
            Logger.Category.DEBUG, Logger.Module.WRAPPER, "Message type not supported"
        )
    elif index >= len(msg_decode_info):
        Logger.log_message(
            Logger.Category.WARNING,
            Logger.Module.WRAPPER,
            "Number of message exceeds maximum supported",
        )
    else:
        msg_decode_info[index].protocol = protocol
        msg_decode_info[index].msg_type = msg_type
        msg_decode_info[index].is_available = len_bytes != 0
        msg_decode_info[index].length_bytes = len_bytes
        msg_decode_info[index].position_bytes = pos_bytes
        msg_decode_info[index].time_sync = time_sync
        index += 1

    return index, msg_decode_info


import pewrapper.types.messages_common as RTCM
import pewrapper.types.messages_decoder_sbf as SBF
import pewrapper.types.messages_decoder_ubx as UBX


def decode_check_preamble(msg: bytes, index_in_bits: int) -> Tuple[bool, int]:
    result = True
    gnss_protocol = GNSS_MSG_PROTOCOL.PROTOCOL_UNDEFINED

    # Check RTCM Preamble
    header_preamble = decode_unsigned_32(msg, index_in_bits, RTCM.SIZE_RTCM_PREAMBLE)
    check_RTCM_preamble = header_preamble == RTCM.RTCM_MSG_HEADER_PREAMBLE

    # Check UBX Preamble
    ubx_preamble = decode_unsigned_32(msg, index_in_bits, UBX.UBX_PREAMBLE_SIZE)
    check_UBX_preamble = ubx_preamble == UBX.UBX_PREAMBLE_ORDERED

    # Check SBF Preamble
    byteOffset = index_in_bits // BITS_IN_BYTE
    sync1 = SBF.decode_msg_endiannes(
        msg[byteOffset:], "B", RTCM.IS_STREAM_LITTLE_ENDIAN
    )
    check_SBF_preamble = sync1 == SBF.SBF_SYNCH_BYTES[0]

    # 2nd SBF Sync Byte
    byteOffset += 1
    sync2 = SBF.decode_msg_endiannes(
        msg[byteOffset:], "B", RTCM.IS_STREAM_LITTLE_ENDIAN
    )
    check_SBF_preamble = check_SBF_preamble and sync2 == SBF.SBF_SYNCH_BYTES[1]

    if check_RTCM_preamble:
        gnss_protocol = GNSS_MSG_PROTOCOL.RTCM
    elif check_UBX_preamble:
        gnss_protocol = GNSS_MSG_PROTOCOL.UBX
    elif check_SBF_preamble:
        gnss_protocol = GNSS_MSG_PROTOCOL.SBF
    else:
        Logger.log_message(
            Logger.Category.WARNING,
            Logger.Module.WRAPPER,
            "UNKNOWN Header decoded. RTCM mode decoded byte is: %s. SBF SYNC1 decoded byte is: %s. SBF SYNC2 decoded byte is: %s. UBX mode decoded byte is: %s",
            header_preamble,
            sync1,
            sync2,
            ubx_preamble,
        )
        result = False

    return result, gnss_protocol


def decode_unsigned_32(msg: bytes, offset: int, length: int) -> int:
    value = 0
    for i in range(offset, offset + length):
        bit = (msg[i // 8] >> 7 - (i % 8)) & 1
        value = (value << 1) + bit

    return value
