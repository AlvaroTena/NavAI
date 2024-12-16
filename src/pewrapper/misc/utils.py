import ctypes as ct
from typing import Union

from navutils.logger import Logger
from pewrapper.api.common_api_types import LogCategoryPE
from pewrapper.api.pe_api_types import (
    Configuration_info,
    GalCorrectionDataType,
    GNSSProtocol,
    SensorQualifier,
    Signal_Obs,
    doppler_sign,
    type_receiver,
)


def get_pe_api_category(category: Logger.Category):
    if category == Logger.Category.NOTSET:
        cat = LogCategoryPE.NONE
    elif category == Logger.Category.DEBUG:
        cat = LogCategoryPE.DEBUG
    elif category == Logger.Category.INFO:
        cat = LogCategoryPE.INFO
    elif category == Logger.Category.WARNING:
        cat = LogCategoryPE.WARNING
    elif category == Logger.Category.ERROR:
        cat = LogCategoryPE.error
    else:
        cat = LogCategoryPE.error
    return ct.c_uint32(cat)


def GetSensorQualifier(value: int) -> SensorQualifier:
    if value == 0:
        return SensorQualifier.N_A
    elif value == 1:
        return SensorQualifier.VALID
    else:  # value == 2 and other
        return SensorQualifier.INVALID


def get_id(
    value: Union[
        Signal_Obs,
        GNSSProtocol,
        GalCorrectionDataType,
        type_receiver,
        doppler_sign,
    ],
    value_str: str = None,
) -> Union[ct.c_uint32, str]:
    if isinstance(value, Signal_Obs):
        if value_str is not None:
            signal_str = value_str
            if signal_str == "GPS_L1_C":
                signal = ct.c_uint32(Signal_Obs.GPS_L1_C)
            elif signal_str == "GPS_L1_C_DP":
                signal = ct.c_uint32(Signal_Obs.GPS_L1_C_DP)
            elif signal_str == "GPS_L2_CM":
                signal = ct.c_uint32(Signal_Obs.GPS_L2_CM)
            elif signal_str == "GPS_L2_CL":
                signal = ct.c_uint32(Signal_Obs.GPS_L2_CL)
            elif signal_str == "GPS_L2_Z":
                signal = ct.c_uint32(Signal_Obs.GPS_L2_Z)
            elif signal_str == "GPS_L5_Q":
                signal = ct.c_uint32(Signal_Obs.GPS_L5_Q)
            elif signal_str == "GAL_E1_C":
                signal = ct.c_uint32(Signal_Obs.GAL_E1_C)
            elif signal_str == "GAL_E1_BC":
                signal = ct.c_uint32(Signal_Obs.GAL_E1_BC)
            elif signal_str == "GAL_E5A_Q":
                signal = ct.c_uint32(Signal_Obs.GAL_E5A_Q)
            elif signal_str == "GAL_E5B_Q":
                signal = ct.c_uint32(Signal_Obs.GAL_E5B_Q)
            elif signal_str == "GAL_E5B_IQ":
                signal = ct.c_uint32(Signal_Obs.GAL_E5B_IQ)
            elif signal_str == "BDS_B1I_D1":
                signal = ct.c_uint32(Signal_Obs.BDS_B1I_D1)
            elif signal_str == "BDS_B1C_P":
                signal = ct.c_uint32(Signal_Obs.BDS_B1C_P)
            elif signal_str == "BDS_B2A_P":
                signal = ct.c_uint32(Signal_Obs.BDS_B2A_P)
            elif signal_str == "BDS_B3I":
                signal = ct.c_uint32(Signal_Obs.BDS_B3I)
            else:
                signal = ct.c_uint32(Signal_Obs.SIG_UNKNOWN)

            return signal
        else:
            signal = value.value
            if signal == Signal_Obs.GPS_L1_C:
                signal_str = "GPS_L1_C"
            elif signal == Signal_Obs.GPS_L1_C_DP:
                signal_str = "GPS_L1_C_DP"
            elif signal == Signal_Obs.GPS_L2_CM:
                signal_str = "GPS_L2_CM"
            elif signal == Signal_Obs.GPS_L2_CL:
                signal_str = "GPS_L2_CL"
            elif signal == Signal_Obs.GPS_L2_Z:
                signal_str = "GPS_L2_Z"
            elif signal == Signal_Obs.GPS_L5_Q:
                signal_str = "GPS_L5_Q"
            elif signal == Signal_Obs.GAL_E1_C:
                signal_str = "GAL_E1_C"
            elif signal == Signal_Obs.GAL_E1_BC:
                signal_str = "GAL_E1_BC"
            elif signal == Signal_Obs.GAL_E5A_Q:
                signal_str = "GAL_E5A_Q"
            elif signal == Signal_Obs.GAL_E5B_Q:
                signal_str = "GAL_E5B_Q"
            elif signal == Signal_Obs.GAL_E5B_IQ:
                signal_str = "GAL_E5B_IQ"
            elif signal == Signal_Obs.BDS_B1I_D1:
                signal_str = "BDS_B1I_D1"
            elif signal == Signal_Obs.BDS_B1C_P:
                signal_str = "BDS_B1C_P"
            elif signal == Signal_Obs.BDS_B2A_P:
                signal_str = "BDS_B2A_P"
            elif signal == Signal_Obs.BDS_B3I:
                signal_str = "BDS_B3I"
            else:
                signal_str = "SIG_UNKNOWN"

            return signal_str

    elif isinstance(value, GNSSProtocol):
        if value_str is not None:
            protocol_str = value_str
            if protocol_str == "RTCM":
                gnssProtocol = ct.c_uint32(GNSSProtocol.RTCM)
            elif protocol_str == "UBX":
                gnssProtocol = ct.c_uint32(GNSSProtocol.UBX)
            elif protocol_str == "SBF":
                gnssProtocol = ct.c_uint32(GNSSProtocol.SBF)
            else:
                gnssProtocol = ct.c_uint32(GNSSProtocol.PROTOCOL_UNKNOWN)

            return gnssProtocol
        else:
            if value.value == GNSSProtocol.RTCM:
                protocol_str = "RTCM"
            elif value.value == GNSSProtocol.UBX:
                protocol_str = "UBX"
            elif value.value == GNSSProtocol.SBF:
                protocol_str = "SBF"
            else:
                protocol_str = "UNKNOWN"
            return protocol_str

    elif isinstance(value, GalCorrectionDataType):
        if value_str is not None:
            gal_corr_data_type_str = value_str
            if gal_corr_data_type_str == "F_NAV":
                galCorrectionDataType_value = ct.c_uint32(GalCorrectionDataType.F_NAV)
            elif gal_corr_data_type_str == "I_NAV":
                galCorrectionDataType_value = ct.c_uint32(GalCorrectionDataType.I_NAV)
            else:
                galCorrectionDataType_value = ct.c_uint32(
                    GalCorrectionDataType.CORRECTION_DATA_UNKNOWN
                )

            return galCorrectionDataType_value
        else:
            if value.value == GalCorrectionDataType.F_NAV:
                gal_corr_data_type_str = "F_NAV"
            elif value.value == GalCorrectionDataType.I_NAV:
                gal_corr_data_type_str = "I_NAV"
            else:
                gal_corr_data_type_str = "UNKNOWN"
            return gal_corr_data_type_str

    elif isinstance(value, type_receiver):
        if value_str is not None:
            algo_profile_str = value_str
            if algo_profile_str == "TESEO":
                type_receiver_value = ct.c_uint32(type_receiver.E_TESEO)
            elif algo_profile_str == "UBLOX":
                type_receiver_value = ct.c_uint32(type_receiver.E_UBLOX)
            elif algo_profile_str == "UBLOX_A9":
                type_receiver_value = ct.c_uint32(type_receiver.E_UBLOX_A9)
            elif algo_profile_str == "MONITOR":
                type_receiver_value = ct.c_uint32(type_receiver.E_MONITOR)
            else:
                type_receiver_value = ct.c_uint32(type_receiver.UNKNOWN)

            return type_receiver_value
        else:
            if value.value == type_receiver.E_TESEO:
                algo_profile_str = "TESEO"
            elif value.value == type_receiver.E_UBLOX:
                algo_profile_str = "UBLOX"
            elif value.value == type_receiver.E_UBLOX_A9:
                algo_profile_str = "UBLOX_A9"
            elif value.value == type_receiver.E_MONITOR:
                algo_profile_str = "MONITOR"
            else:
                algo_profile_str = "UNKNOWN"
            return algo_profile_str

    elif isinstance(value, doppler_sign):
        if value_str is not None:
            doppler_sign_str = value_str
            if doppler_sign_str == "POSITIVE":
                doppler_sign_value = ct.c_uint32(doppler_sign.POSITIVE)
            elif doppler_sign_str == "NEGATIVE":
                doppler_sign_value = ct.c_uint32(doppler_sign.NEGATIVE)
            else:
                doppler_sign_value = ct.c_uint32(doppler_sign.UNKNOWN_SIGN)

            return doppler_sign_value
        else:
            if value.value == doppler_sign.POSITIVE:
                doppler_sign_str = "POSITIVE"
            elif value.value == doppler_sign.NEGATIVE:
                doppler_sign_str = "NEGATIVE"
            else:
                doppler_sign_str = "UNKNOWN"
            return doppler_sign_str


def deepcopy_config(config: Configuration_info):
    config_copy = Configuration_info(
        config.binEphem_handle,
        config.log_handle,
        config.ReportDTCStatus,
    )
    ct.memmove(
        ct.addressof(config_copy),
        ct.addressof(config),
        ct.sizeof(config),
    )

    return config_copy
