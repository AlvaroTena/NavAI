import ctypes as ct
from typing import Type, Union

import pewrapper.api as PE_API
from navutils.logger import Logger


def get_pe_api_category(category: Logger.Category):
    if category == Logger.Category.NOTSET:
        cat = PE_API.LogCategoryPE.NONE
    elif category == Logger.Category.DEBUG:
        cat = PE_API.LogCategoryPE.DEBUG
    elif category == Logger.Category.INFO:
        cat = PE_API.LogCategoryPE.INFO
    elif category == Logger.Category.WARNING:
        cat = PE_API.LogCategoryPE.WARNING
    elif category == Logger.Category.ERROR:
        cat = PE_API.LogCategoryPE.error
    else:
        cat = PE_API.LogCategoryPE.error
    return ct.c_uint32(cat)


def GetSensorQualifier(value: int) -> PE_API.SensorQualifier:
    if value == 0:
        return PE_API.SensorQualifier.N_A
    elif value == 1:
        return PE_API.SensorQualifier.VALID
    else:  # value == 2 and other
        return PE_API.SensorQualifier.INVALID


def get_id(
    value: Union[
        PE_API.Signal_Obs,
        PE_API.GNSSProtocol,
        PE_API.GalCorrectionDataType,
        PE_API.type_receier,
        PE_API.doppler_sign,
    ],
    value_str: str = None,
) -> Union[ct.c_uint32, str]:
    if isinstance(value, PE_API.Signal_Obs):
        if value_str is not None:
            signal_str = value_str
            if signal_str == "GPS_L1_C":
                signal = ct.c_uint32(PE_API.Signal_Obs.GPS_L1_C)
            elif signal_str == "GPS_L1_C_DP":
                signal = ct.c_uint32(PE_API.Signal_Obs.GPS_L1_C_DP)
            elif signal_str == "GPS_L2_CM":
                signal = ct.c_uint32(PE_API.Signal_Obs.GPS_L2_CM)
            elif signal_str == "GPS_L2_CL":
                signal = ct.c_uint32(PE_API.Signal_Obs.GPS_L2_CL)
            elif signal_str == "GPS_L2_Z":
                signal = ct.c_uint32(PE_API.Signal_Obs.GPS_L2_Z)
            elif signal_str == "GPS_L5_Q":
                signal = ct.c_uint32(PE_API.Signal_Obs.GPS_L5_Q)
            elif signal_str == "GAL_E1_C":
                signal = ct.c_uint32(PE_API.Signal_Obs.GAL_E1_C)
            elif signal_str == "GAL_E1_BC":
                signal = ct.c_uint32(PE_API.Signal_Obs.GAL_E1_BC)
            elif signal_str == "GAL_E5A_Q":
                signal = ct.c_uint32(PE_API.Signal_Obs.GAL_E5A_Q)
            elif signal_str == "GAL_E5B_Q":
                signal = ct.c_uint32(PE_API.Signal_Obs.GAL_E5B_Q)
            elif signal_str == "GAL_E5B_IQ":
                signal = ct.c_uint32(PE_API.Signal_Obs.GAL_E5B_IQ)
            elif signal_str == "BDS_B1I_D1":
                signal = ct.c_uint32(PE_API.Signal_Obs.BDS_B1I_D1)
            elif signal_str == "BDS_B1C_P":
                signal = ct.c_uint32(PE_API.Signal_Obs.BDS_B1C_P)
            elif signal_str == "BDS_B2A_P":
                signal = ct.c_uint32(PE_API.Signal_Obs.BDS_B2A_P)
            elif signal_str == "BDS_B3I":
                signal = ct.c_uint32(PE_API.Signal_Obs.BDS_B3I)
            else:
                signal = ct.c_uint32(PE_API.Signal_Obs.SIG_UNKNOWN)

            return signal
        else:
            signal = value.value
            if signal == PE_API.Signal_Obs.GPS_L1_C:
                signal_str = "GPS_L1_C"
            elif signal == PE_API.Signal_Obs.GPS_L1_C_DP:
                signal_str = "GPS_L1_C_DP"
            elif signal == PE_API.Signal_Obs.GPS_L2_CM:
                signal_str = "GPS_L2_CM"
            elif signal == PE_API.Signal_Obs.GPS_L2_CL:
                signal_str = "GPS_L2_CL"
            elif signal == PE_API.Signal_Obs.GPS_L2_Z:
                signal_str = "GPS_L2_Z"
            elif signal == PE_API.Signal_Obs.GPS_L5_Q:
                signal_str = "GPS_L5_Q"
            elif signal == PE_API.Signal_Obs.GAL_E1_C:
                signal_str = "GAL_E1_C"
            elif signal == PE_API.Signal_Obs.GAL_E1_BC:
                signal_str = "GAL_E1_BC"
            elif signal == PE_API.Signal_Obs.GAL_E5A_Q:
                signal_str = "GAL_E5A_Q"
            elif signal == PE_API.Signal_Obs.GAL_E5B_Q:
                signal_str = "GAL_E5B_Q"
            elif signal == PE_API.Signal_Obs.GAL_E5B_IQ:
                signal_str = "GAL_E5B_IQ"
            elif signal == PE_API.Signal_Obs.BDS_B1I_D1:
                signal_str = "BDS_B1I_D1"
            elif signal == PE_API.Signal_Obs.BDS_B1C_P:
                signal_str = "BDS_B1C_P"
            elif signal == PE_API.Signal_Obs.BDS_B2A_P:
                signal_str = "BDS_B2A_P"
            elif signal == PE_API.Signal_Obs.BDS_B3I:
                signal_str = "BDS_B3I"
            else:
                signal_str = "SIG_UNKNOWN"

            return signal_str

    elif isinstance(value, PE_API.GNSSProtocol):
        if value_str is not None:
            protocol_str = value_str
            if protocol_str == "RTCM":
                gnssProtocol = ct.c_uint32(PE_API.GNSSProtocol.RTCM)
            elif protocol_str == "UBX":
                gnssProtocol = ct.c_uint32(PE_API.GNSSProtocol.UBX)
            elif protocol_str == "SBF":
                gnssProtocol = ct.c_uint32(PE_API.GNSSProtocol.SBF)
            else:
                gnssProtocol = ct.c_uint32(PE_API.GNSSProtocol.PROTOCOL_UNKNOWN)

            return gnssProtocol
        else:
            gnssProtocol = value.value
            if gnssProtocol == PE_API.GNSSProtocol.RTCM:
                protocol_str = "RTCM"
            elif gnssProtocol == PE_API.GNSSProtocol.UBX:
                protocol_str = "UBX"
            elif gnssProtocol == PE_API.GNSSProtocol.SBF:
                protocol_str = "SBF"
            else:
                protocol_str = "UNKNOWN"
            return protocol_str

    elif isinstance(value, PE_API.GalCorrectionDataType):
        if value_str is not None:
            gal_corr_data_type_str = value_str
            if gal_corr_data_type_str == "F_NAV":
                galCorrectionDataType = ct.c_uint32(PE_API.GalCorrectionDataType.F_NAV)
            elif gal_corr_data_type_str == "I_NAV":
                galCorrectionDataType = ct.c_uint32(PE_API.GalCorrectionDataType.I_NAV)
            else:
                galCorrectionDataType = ct.c_uint32(
                    PE_API.GalCorrectionDataType.CORRECTION_DATA_UNKNOWN
                )

            return galCorrectionDataType
        else:
            galCorrectionDataType = value.value
            if galCorrectionDataType == PE_API.GalCorrectionDataType.F_NAV:
                gal_corr_data_type_str = "F_NAV"
            elif galCorrectionDataType == PE_API.GalCorrectionDataType.I_NAV:
                gal_corr_data_type_str = "I_NAV"
            else:
                gal_corr_data_type_str = "UNKNOWN"
            return gal_corr_data_type_str

    elif isinstance(value, PE_API.type_receier):
        if value_str is not None:
            algo_profile_str = value_str
            if algo_profile_str == "TESEO":
                type_receiver = ct.c_uint32(PE_API.type_receier.E_TESEO)
            elif algo_profile_str == "UBLOX":
                type_receiver = ct.c_uint32(PE_API.type_receier.E_UBLOX)
            elif algo_profile_str == "UBLOX_A9":
                type_receiver = ct.c_uint32(PE_API.type_receier.E_UBLOX_A9)
            elif algo_profile_str == "MONITOR":
                type_receiver = ct.c_uint32(PE_API.type_receier.E_MONITOR)
            else:
                type_receiver = ct.c_uint32(PE_API.type_receier.UNKNOWN)

            return type_receiver
        else:
            type_receiver = value.value
            if type_receiver == PE_API.type_receier.E_TESEO:
                algo_profile_str = "TESEO"
            elif type_receiver == PE_API.type_receier.E_UBLOX:
                algo_profile_str = "UBLOX"
            elif type_receiver == PE_API.type_receier.E_UBLOX_A9:
                algo_profile_str = "UBLOX_A9"
            elif type_receiver == PE_API.type_receier.E_MONITOR:
                algo_profile_str = "MONITOR"
            else:
                algo_profile_str = "UNKNOWN"
            return algo_profile_str

    elif isinstance(value, PE_API.doppler_sign):
        if value_str is not None:
            doppler_sign_str = value_str
            if doppler_sign_str == "POSITIVE":
                doppler_sign = ct.c_uint32(PE_API.doppler_sign.POSITIVE)
            elif doppler_sign_str == "NEGATIVE":
                doppler_sign = ct.c_uint32(PE_API.doppler_sign.NEGATIVE)
            else:
                doppler_sign = ct.c_uint32(PE_API.doppler_sign.UNKNOWN_SIGN)

            return doppler_sign
        else:
            doppler_sign = value.value
            if doppler_sign == PE_API.doppler_sign.POSITIVE:
                doppler_sign_str = "POSITIVE"
            elif doppler_sign == PE_API.doppler_sign.NEGATIVE:
                doppler_sign_str = "NEGATIVE"
            else:
                doppler_sign_str = "UNKNOWN"
            return doppler_sign_str


def deepcopy_config(config: PE_API.Configuration_info):
    config_copy = PE_API.Configuration_info(
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
