from typing import Tuple, List, Optional
import pewrapper.api.pe_api_types as PE_API
from pewrapper.managers.configuration_mgr import ConfigurationManager
from pewrapper.types.messages_details_common import (
    decode_check_preamble,
    GNSS_MSG_PROTOCOL,
)
from pewrapper.types.constants import BITS_IN_BYTE
from navutils.logger import Logger
from pewrapper.decoders.ubx_decoder import UBX_Decoder
from pewrapper.types.messages_details_common import (
    Msg_Decode_Info,
    Submsg_Decode_Info,
    MAX_GNSS_MESSAGE,
)
from pewrapper.types.gps_time_wrapper import GPS_Time


class Decoder_Manager:
    """Manager for decoding GNSS messages and extracting epoch information."""

    def __init__(self, configurationMgr: ConfigurationManager):
        """Initialize the decoder manager.

        Args:
            configurationMgr: Configuration manager containing GNSS protocol settings
        """
        self._config_mgr = configurationMgr
        self._gnss_protocol = PE_API.GNSSProtocol.PROTOCOL_UNKNOWN
        self._ubx_decoder = UBX_Decoder(configurationMgr)
        self._initialized = False

    def initialise(self) -> bool:
        """Initialize the decoder manager.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        result = self._ubx_decoder.initialise()
        self._gnss_protocol = self._config_mgr.config_info_.gnssProtocol
        self._initialized = result
        return result

    def reset(self) -> None:
        """Reset the decoder manager."""
        self._ubx_decoder.reset()

    def _identify_message_boundaries(
        self,
        msg: bytes,
        message_length: int,
        msg_decode_info: Msg_Decode_Info,
        msg_index_num_msg: int,
        msg_result: bool,
    ) -> Tuple[bool, Msg_Decode_Info, int, bool]:
        """Identify message boundaries and protocols in the input data.

        Args:
            msg: Raw GNSS message bytes
            message_length: Length of the message in bytes
            msg_decode_info: Array to store message decoding information
            msg_index_num_msg: Current message index
            msg_result: Current decoding result status

        Returns:
            Tuple containing:
            - result: Overall success status
            - msg_decode_info: Updated message decoding information
            - msg_index_num_msg: Updated message index
            - msg_result: Updated decoding result status
        """
        result = True
        msg_result = True
        msg_index_num_msg = 0
        index_in_bits = 0
        message_len_bytes = 0
        global_monitor_state = True

        total_bits = message_length * BITS_IN_BYTE

        while result and msg_result and index_in_bits < total_bits:
            gnss_protocol = GNSS_MSG_PROTOCOL.PROTOCOL_UNDEFINED
            msg_result, gnss_protocol = decode_check_preamble(msg, index_in_bits)

            if gnss_protocol == GNSS_MSG_PROTOCOL.RTCM:
                raise NotImplementedError("RTCM protocol is not implemented")
            elif (
                gnss_protocol == GNSS_MSG_PROTOCOL.UBX
                and self._gnss_protocol == PE_API.GNSSProtocol.UBX
            ):
                (
                    aux_result,
                    global_monitor_state,
                    index_in_bits,
                    message_len_bytes,
                    msg_decode_info,
                    msg_index_num_msg,
                    msg_result,
                ) = self._ubx_decoder.parse_message_header(
                    msg,
                    global_monitor_state,
                    index_in_bits,
                    message_len_bytes,
                    msg_decode_info,
                    msg_index_num_msg,
                    msg_result,
                )
                result &= aux_result
            elif (
                gnss_protocol == GNSS_MSG_PROTOCOL.SBF
                and self._gnss_protocol == PE_API.GNSSProtocol.SBF
            ):
                raise NotImplementedError("SBF protocol is not implemented")
            else:
                msg_result = False
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.WRAPPER,
                    "The GNSS protocol decoded and the GNSS protocol configured do not match",
                )

        return result, msg_decode_info, msg_index_num_msg, msg_result

    def extract_epoch_from_gnss_message(
        self, msg: bytes, message_length: int
    ) -> Tuple[bool, bool, GPS_Time]:
        """Decode GNSS message and extract epoch information.

        Args:
            msg: Raw GNSS message bytes
            message_length: Length of the message in bytes

        Returns:
            Tuple containing:
            - result: Overall success status
            - msg_result: Message decoding success status
            - epoch: Extracted GPS time information
        """
        result = self._initialized
        epoch = GPS_Time()

        if not result:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.WRAPPER,
                "Data Decoder class not initialised",
            )
            return result, False, epoch

        msg_result = True
        msg_decode_info: Msg_Decode_Info = [
            Submsg_Decode_Info() for _ in range(MAX_GNSS_MESSAGE)
        ]
        num_messages = 0

        # Identify message boundaries
        aux_result, msg_decode_info, num_messages, msg_result = (
            self._identify_message_boundaries(
                msg, message_length, msg_decode_info, num_messages, msg_result
            )
        )
        result &= aux_result

        # Process each identified message
        for idx_msg in range(num_messages):
            protocol = msg_decode_info[idx_msg].protocol

            if protocol == GNSS_MSG_PROTOCOL.RTCM:
                raise NotImplementedError("RTCM protocol is not implemented")
            elif (
                protocol == GNSS_MSG_PROTOCOL.UBX
                and self._gnss_protocol == PE_API.GNSSProtocol.UBX
            ):
                aux_result, epoch_decoded = (
                    self._ubx_decoder.extract_epoch_from_gnss_message(
                        msg, msg_decode_info[idx_msg]
                    )
                )
                result &= aux_result
                msg_result &= epoch_decoded[0]
                if epoch_decoded[0]:
                    epoch = epoch_decoded[1]
            elif (
                protocol == GNSS_MSG_PROTOCOL.SBF
                and self._gnss_protocol == PE_API.GNSSProtocol.SBF
            ):
                raise NotImplementedError("SBF protocol is not implemented")

        return result, msg_result, epoch
