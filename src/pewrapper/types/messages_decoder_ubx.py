import struct
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pewrapper.types.monitor_types as monitoring
import pewrapper.types.utils_function as utils_function
from navutils.logger import Logger
from pewrapper.types.constants import BITS_IN_BYTE
from pewrapper.types.gps_time_wrapper import GPS_Time
from pewrapper.types.messages_details_common import GNSS_MESSAGE_TYPES


def U1(p: bytes) -> int:
    return int.from_bytes(p[:1], byteorder="little")


def U2(p: bytes) -> int:
    return int.from_bytes(p[:2], byteorder="little")


def U4(p: bytes) -> int:
    return int.from_bytes(p[:4], byteorder="little")


def R8(p: bytes) -> float:
    return struct.unpack("<d", p[:8])[0]


MAX_SEC_CRC_COUNTER_VALUE = 0xFFFF
SEC_CRC_COUNTER_CHECK_ROLLOVER_THRESHOLD = 10

CRC_LENGTH = 4
CHECKSUM_LENGTH = 2
CHECKSUM_HALF_LENGTH = 1

UBX_PREAMBLE_ORDERED = 0xB562
UBX_PREAMBLE_SIZE = 2 * BITS_IN_BYTE


class UBX_PREAMBLE:
    HEADER = 0x62B5
    CLASS_RXM = 0x02
    CLASS_SEC_CRC = 0x27
    CLASS_ESF_MEAS = 0x10
    CLASS_SUBX_MON = 0x80
    CLASS_MON = 0x0A
    ID_RAWX = 0x15
    ID_SFRBX = 0x13
    ID_SEC_CRC = 0x0B
    ID_ESF_MEAS = 0x02
    ID_SUBX_MON = 0x11
    ID_MEASX = 0x14
    ID_COMMS = 0x36
    ID_RF = 0x38


class UBX_SEC_CRC_OFFSETS_BYTES:
    HEADER = 0
    CLASS = 2
    ID = 3
    LENGTH = 4

    VERSION = 6
    RESERVED = 7
    COUNTER = 8
    MSG_CLASS = 10
    MSG_ID = 11
    MSG_LENGTH = 12
    MSG_PAYLOAD = 14


class UBX_PREAMBLE_OFFSETS_BYTES:
    HEADER = 0
    CLASS = 2
    ID = 3
    LENGTH = 4
    PAYLOAD = 6


class UBX_RAWX_OFFSETS_BYTES:
    RCVTOW = 0
    WEEK = 8


def decode_ubx_sec_crc(
    msg: bytes,
    crc_table: npt.NDArray[np.uint32],
    counter_init: bool,
    gnss_counter: int,
    monitor_state: monitoring.E2EMsgValidity,
) -> Tuple[bool, monitoring.E2EMsgValidity]:
    result = check_message_type_ubx(
        msg, UBX_PREAMBLE.CLASS_SEC_CRC, UBX_PREAMBLE.ID_SEC_CRC
    )

    if result:
        counter = U2(msg[UBX_SEC_CRC_OFFSETS_BYTES.COUNTER :])
        length = U2(msg[UBX_SEC_CRC_OFFSETS_BYTES.MSG_LENGTH :])
        crc = U4(msg[UBX_SEC_CRC_OFFSETS_BYTES.MSG_PAYLOAD + length :])

        if not counter_init:
            counter_init = True

            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.WRAPPER,
                "GNSS Message SEC-CRC Alive Counter initialization at: %s",
                counter,
            )
            monitor_state.aliveCounter = False

        else:
            valid_counter, _ = utils_function.check_counter(
                counter,
                gnss_counter,
                MAX_SEC_CRC_COUNTER_VALUE,
                SEC_CRC_COUNTER_CHECK_ROLLOVER_THRESHOLD,
                False,
            )

            if not valid_counter:
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.WRAPPER,
                    "GNSS Message SEC-CRC Alive Counter Check: %s Moving backwards from last reference: %s",
                    counter,
                    gnss_counter,
                )
                monitor_state.aliveCounter = False

        # Update memory counter
        gnss_counter = counter

        # Compute CRC
        crc_length = (
            length
            + UBX_SEC_CRC_OFFSETS_BYTES.MSG_PAYLOAD
            - UBX_SEC_CRC_OFFSETS_BYTES.CLASS
        )
        crc_calculated = utils_function.compute_crc_e2e(
            msg[UBX_SEC_CRC_OFFSETS_BYTES.CLASS :], crc_length, crc_table
        )

        if crc != crc_calculated:
            monitor_state.crc = False

            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.WRAPPER,
                "UBX Message: Decoded CRC: %s does not match with the expected value: %s",
                crc,
                crc_calculated,
            )

    else:
        monitor_state.headerDataId = False

    return result, counter_init, gnss_counter, monitor_state


def decode_ubx_rxm_rawx(msg: bytes, require_sec_crc: bool) -> Tuple[bool, GPS_Time]:
    result = True

    if require_sec_crc:
        offset = UBX_SEC_CRC_OFFSETS_BYTES.MSG_PAYLOAD
    else:
        result &= check_message_type_ubx(
            msg, UBX_PREAMBLE.CLASS_RXM, UBX_PREAMBLE.ID_RAWX
        )
        offset = UBX_PREAMBLE_OFFSETS_BYTES.PAYLOAD

    aux_result, epoch = decode_epoch_rawx_payload(msg[offset:])
    result &= aux_result

    if not result:
        Logger.log_message(
            Logger.Category.WARNING,
            Logger.Module.WRAPPER,
            "GNSS Message: Error decoding UBX-RXM-RAWX.",
        )
        Logger.log_message(
            Logger.Category.WARNING,
            Logger.Module.WRAPPER,
            "UBX Message: Header message does not match with the expected.",
        )

    return result, epoch


def compute_checksum(buff: bytes, lenght: int) -> tuple[int, int]:
    checksum_A = 0
    checksum_B = 0

    actual_len = min(lenght, len(buff))
    for i in range(actual_len):
        b = buff[i]
        checksum_A = (checksum_A + b) % 256
        checksum_B = (checksum_B + checksum_A) % 256

    return checksum_A, checksum_B


def is_correct_checksum(
    msg: bytes, lenght: int, checksum_A_value: int, checksum_B_value
) -> bool:
    checksum_A, checksum_B = compute_checksum(msg, lenght)

    result = checksum_A_value == checksum_A and checksum_B_value == checksum_B

    if not result:
        Logger.log_message(
            Logger.Category.WARNING,
            Logger.Module.WRAPPER,
            "GNSS Message: Incorrect UBX-RXM checksum.",
        )

    return result


def check_message_type_ubx(msg: bytes, preamble_class: int, preamble_id: int) -> bool:
    result = True
    message_header = U2(msg[UBX_PREAMBLE_OFFSETS_BYTES.HEADER :])

    if message_header == UBX_PREAMBLE.HEADER:
        message_class = U1(msg[UBX_PREAMBLE_OFFSETS_BYTES.CLASS :])
        message_id = U1(msg[UBX_PREAMBLE_OFFSETS_BYTES.ID :])

        if message_class == preamble_class and message_id == preamble_id:
            length = U2(msg[UBX_PREAMBLE_OFFSETS_BYTES.LENGTH :])
            checksum_A = U1(msg[UBX_PREAMBLE_OFFSETS_BYTES.PAYLOAD + length :])
            checksum_B = U1(
                msg[
                    UBX_PREAMBLE_OFFSETS_BYTES.PAYLOAD + length + CHECKSUM_HALF_LENGTH :
                ]
            )

            checksum_length = (
                length
                + UBX_PREAMBLE_OFFSETS_BYTES.PAYLOAD
                - UBX_PREAMBLE_OFFSETS_BYTES.CLASS
            )
            result = is_correct_checksum(
                msg[UBX_PREAMBLE_OFFSETS_BYTES.CLASS :],
                checksum_length,
                checksum_A,
                checksum_B,
            )

        else:
            result = False
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.WRAPPER,
                "UBX Message: Class or id UBX message does not match with the expected. Class message: %s | Class expected: %s | Id message: %s | Id expected: %s",
                message_class,
                preamble_class,
                message_id,
                preamble_id,
            )

    else:
        result = False
        Logger.log_message(
            Logger.Category.WARNING,
            Logger.Module.WRAPPER,
            "UBX Message: Header UBX message does not match with the expected.",
        )

    return result


def get_message_type_ubx(
    msg: bytes,
    require_sec_crc: bool,
    index_in_bytes: int,
    message_len_bytes: int,
    types_ubx: int,
):
    result = True
    header = 0

    if require_sec_crc:
        class_offset = UBX_SEC_CRC_OFFSETS_BYTES.MSG_CLASS
        id_offset = UBX_SEC_CRC_OFFSETS_BYTES.MSG_ID
        length_offset = UBX_SEC_CRC_OFFSETS_BYTES.MSG_LENGTH
        extra_payload_length = (
            UBX_SEC_CRC_OFFSETS_BYTES.MSG_PAYLOAD + CRC_LENGTH + CHECKSUM_LENGTH
        )

    else:
        header = U2(msg[index_in_bytes + UBX_PREAMBLE_OFFSETS_BYTES.HEADER :])
        class_offset = UBX_PREAMBLE_OFFSETS_BYTES.CLASS
        id_offset = UBX_PREAMBLE_OFFSETS_BYTES.ID
        length_offset = UBX_PREAMBLE_OFFSETS_BYTES.LENGTH
        extra_payload_length = UBX_PREAMBLE_OFFSETS_BYTES.PAYLOAD + CHECKSUM_LENGTH

    if require_sec_crc or header == UBX_PREAMBLE.HEADER:
        preamble_class = U1(msg[index_in_bytes + class_offset :])
        preamble_id = U1(msg[index_in_bytes + id_offset :])
        payload_length = U2(msg[index_in_bytes + length_offset :])

        message_len_bytes = payload_length + extra_payload_length

        types_ubx = GNSS_MESSAGE_TYPES.MESSAGE_TYPE_UNDEFINED

        if preamble_class == UBX_PREAMBLE.CLASS_RXM:
            if preamble_id == UBX_PREAMBLE.ID_RAWX:
                types_ubx = GNSS_MESSAGE_TYPES.UBX_RXM_RAWX
            elif preamble_id == UBX_PREAMBLE.ID_SFRBX:
                types_ubx = GNSS_MESSAGE_TYPES.UBX_RXM_SFRBX
            elif preamble_id == UBX_PREAMBLE.ID_MEASX:
                types_ubx = GNSS_MESSAGE_TYPES.UBX_RXM_MEASX
            else:
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.WRAPPER,
                    "GNSS Message: Unknown UBX RXM message.",
                )
        elif preamble_class == UBX_PREAMBLE.CLASS_ESF_MEAS:
            if preamble_id == UBX_PREAMBLE.ID_ESF_MEAS:
                types_ubx = GNSS_MESSAGE_TYPES.UBX_ESF_MEAS
            else:
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.WRAPPER,
                    "GNSS Message: Unknown UBX ESF message.",
                )
        elif preamble_class == UBX_PREAMBLE.CLASS_SUBX_MON:
            if preamble_id == UBX_PREAMBLE.ID_SUBX_MON:
                types_ubx = GNSS_MESSAGE_TYPES.UBX_SUBX_MON
            else:
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.WRAPPER,
                    "GNSS Message: Unknown UBX SUBX message.",
                )
        elif preamble_class == UBX_PREAMBLE.CLASS_MON:
            if preamble_id == UBX_PREAMBLE.ID_COMMS:
                types_ubx = GNSS_MESSAGE_TYPES.UBX_MON_COMMS
            elif preamble_id == UBX_PREAMBLE.ID_RF:
                types_ubx = GNSS_MESSAGE_TYPES.UBX_MON_RF
            else:
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.WRAPPER,
                    "GNSS Message: Unknown UBX MON message.",
                )
        else:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.WRAPPER,
                "GNSS Message: Unknown UBX message.",
            )
    else:
        result = False
        Logger.log_message(
            Logger.Category.WARNING,
            Logger.Module.WRAPPER,
            "GNSS Message: Not an UBX message.",
        )

    return result, message_len_bytes, types_ubx


def decode_epoch_rawx_payload(msg: bytes):
    result = True

    rcvTow = R8(msg[UBX_RAWX_OFFSETS_BYTES.RCVTOW :])
    week = U2(msg[UBX_RAWX_OFFSETS_BYTES.WEEK :])

    return result, GPS_Time(week, rcvTow)
