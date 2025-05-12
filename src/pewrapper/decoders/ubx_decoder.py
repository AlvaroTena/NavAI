from typing import Tuple

import numpy as np
import pewrapper.types.messages_decoder_ubx as UBX
import pewrapper.types.monitor_types as monitoring
import pewrapper.types.utils_function as util_functions
from navutils.logger import Logger
from pewrapper.managers.configuration_mgr import ConfigurationManager
from pewrapper.types.constants import BITS_IN_BYTE
from pewrapper.types.gps_time_wrapper import GPS_Time
from pewrapper.types.messages_details_common import (
    GNSS_MESSAGE_TYPES,
    GNSS_MSG_PROTOCOL,
    Msg_Decode_Info,
    Submsg_Decode_Info,
    fill_msg_decode_info,
)

CRC_TABLE_SIZE = 256


class UBX_Decoder:
    """Decoder for UBX (u-blox) GNSS messages with support for end-to-end validation."""

    def __init__(self, configurationMgr: "ConfigurationManager"):
        """Initialize the UBX decoder with a configuration manager.

        Args:
            configurationMgr: Configuration manager containing decoder settings
        """
        self._configurationMgr = configurationMgr
        self._initialized = False
        self._ubx_counter = 0
        self._counter_init = False
        self._crc_table = np.zeros(CRC_TABLE_SIZE, dtype=np.uint32)

    def initialise(self) -> bool:
        """Initialize the CRC table for end-to-end message validation.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        result, self._crc_table = util_functions.generate_table_e2e(self._crc_table)
        self._initialized = result
        return self._initialized

    def reset(self) -> None:
        """Reset the decoder state."""
        self._initialized = False
        self._ubx_counter = 0
        self._counter_init = False

    def extract_epoch_from_gnss_message(
        self,
        msg: bytes,
        single_decode_info: "Submsg_Decode_Info",
    ) -> Tuple[bool, Tuple[bool, "GPS_Time"]]:
        """Extract epoch information from a GNSS message.

        Args:
            msg: Raw message bytes
            single_decode_info: Information about the message to decode

        Returns:
            Tuple containing:
                - Decoder initialization status
                - Tuple with message decoding result and GPS time
        """
        result = self._initialized
        msg_result = False
        epoch = GPS_Time()
        require_sec_crc = self._configurationMgr.config_info_.require_e2e_msg

        if single_decode_info.is_available:
            if single_decode_info.msg_type == GNSS_MESSAGE_TYPES.UBX_RXM_RAWX:
                aux_result, epoch = UBX.decode_ubx_rxm_rawx(
                    msg[single_decode_info.position_bytes :], require_sec_crc
                )
                if not aux_result:
                    Logger.log_message(
                        Logger.Category.WARNING,
                        Logger.Module.WRAPPER,
                        "GNSS Message: UBX-RXM-RAWX not decoded",
                    )
                else:
                    msg_result = True

        return result, (msg_result, epoch)

    def parse_message_header(
        self,
        msg: bytes,
        global_monitor_state: bool,
        index_in_bits: int,
        message_len_bytes: int,
        msg_decode_info: "Msg_Decode_Info",
        msg_index_in_decode_info: int,
        msg_result: bool,
    ) -> Tuple[bool, bool, int, int, "Msg_Decode_Info", int, bool]:
        """Parse UBX message header and validate message integrity.

        Args:
            msg: Raw message bytes
            global_monitor_state: Current monitoring state
            index_in_bits: Current bit position in the message
            message_len_bytes: Current message length in bytes
            msg_decode_info: Message decode information structure
            msg_index_in_decode_info: Current index in decode info
            msg_result: Current message result status

        Returns:
            Tuple containing updated versions of all input parameters
        """
        result = self._initialized
        valid_msg = True
        monitor_state_sec_crc = monitoring.E2EMsgValidity()
        require_e2e_msg = self._configurationMgr.config_info_.require_e2e_msg
        position_in_bytes = index_in_bits // BITS_IN_BYTE

        # Validate message with end-to-end CRC if required
        if require_e2e_msg:
            valid_msg, self._counter_init, self._ubx_counter, monitor_state_sec_crc = (
                UBX.decode_ubx_sec_crc(
                    msg[position_in_bytes:],
                    self._crc_table,
                    self._counter_init,
                    self._ubx_counter,
                    monitor_state_sec_crc,
                )
            )

            if valid_msg:
                msg_result = True
                global_monitor_state &= monitoring.get_e2e_msg_validity(
                    monitor_state_sec_crc
                )

        # Process valid message with valid CRC
        if valid_msg and monitoring.get_e2e_msg_validity(monitor_state_sec_crc):
            message_type = GNSS_MESSAGE_TYPES.MESSAGE_TYPE_UNDEFINED
            msg_result, message_len_bytes, message_type = UBX.get_message_type_ubx(
                msg,
                require_e2e_msg,
                position_in_bytes,
                message_len_bytes,
                message_type,
            )

            index_in_bits += message_len_bytes * BITS_IN_BYTE

            msg_index_in_decode_info, msg_decode_info = fill_msg_decode_info(
                GNSS_MSG_PROTOCOL.UBX,
                message_type,
                message_len_bytes,
                position_in_bytes,
                0.0,
                msg_index_in_decode_info,
                msg_decode_info,
            )
        # Process valid message with initialized counter but invalid CRC
        elif valid_msg and self._counter_init:
            sec_crc_length = UBX.U2(
                msg[position_in_bytes + UBX.UBX_SEC_CRC_OFFSETS_BYTES.LENGTH :]
            )

            message_len_bytes = (
                sec_crc_length
                + UBX.UBX_SEC_CRC_OFFSETS_BYTES.VERSION
                + UBX.CHECKSUM_LENGTH
            )
            index_in_bits += message_len_bytes * BITS_IN_BYTE
        # Process message without SEC-CRC
        else:
            length = UBX.U2(
                msg[position_in_bytes + UBX.UBX_PREAMBLE_OFFSETS_BYTES.LENGTH :]
            )

            message_len_bytes = (
                length + UBX.UBX_PREAMBLE_OFFSETS_BYTES.PAYLOAD + UBX.CHECKSUM_LENGTH
            )
            index_in_bits += message_len_bytes * BITS_IN_BYTE

        return (
            result,
            global_monitor_state,
            index_in_bits,
            message_len_bytes,
            msg_decode_info,
            msg_index_in_decode_info,
            msg_result,
        )
