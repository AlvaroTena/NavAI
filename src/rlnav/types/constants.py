import pewrapper.api as PE_API

SENTINEL_VALUE = 0


class FREQUENCY:
    FREQ_L1_GPS = [
        PE_API.Signal_Obs.GPS_L1_C.value,
        PE_API.Signal_Obs.GPS_L1_C_DP.value,
        PE_API.Signal_Obs.GPS_L1_CP.value,
        PE_API.Signal_Obs.GPS_L1_P.value,
    ]
    FREQ_L2_GPS = [
        PE_API.Signal_Obs.GPS_L2_CL.value,
        PE_API.Signal_Obs.GPS_L2_CM.value,
        PE_API.Signal_Obs.GPS_L2_P.value,
        PE_API.Signal_Obs.GPS_L2_Z.value,
    ]
    FREQ_L5_GPS = [PE_API.Signal_Obs.GPS_L5_IQ.value, PE_API.Signal_Obs.GPS_L5_Q.value]
    FREQ_E1_GAL = [PE_API.Signal_Obs.GAL_E1_BC.value, PE_API.Signal_Obs.GAL_E1_C.value]
    FREQ_E5A_GAL = [
        PE_API.Signal_Obs.GAL_E5A_I.value,
        PE_API.Signal_Obs.GAL_E5A_IQ.value,
        PE_API.Signal_Obs.GAL_E5A_Q.value,
    ]
    FREQ_E5B_GAL = [
        PE_API.Signal_Obs.GAL_E5B_I.value,
        PE_API.Signal_Obs.GAL_E5B_IQ.value,
        PE_API.Signal_Obs.GAL_E5B_Q.value,
    ]
    FREQ_E5AB_GAL = [
        PE_API.Signal_Obs.GAL_E5AB_I.value,
        PE_API.Signal_Obs.GAL_E5AB_IQ.value,
        PE_API.Signal_Obs.GAL_E5AB_Q.value,
    ]
    FREQ_E6_GAL = [
        PE_API.Signal_Obs.GAL_E6_A.value,
        PE_API.Signal_Obs.GAL_E6_B.value,
        PE_API.Signal_Obs.GAL_E6_C.value,
    ]
    FREQ_B1A_BDS = [PE_API.Signal_Obs.BDS_B1A_1D.value]
    FREQ_B1C_BDS = [PE_API.Signal_Obs.BDS_B1C_P.value]
    FREQ_B1I_BDS = [PE_API.Signal_Obs.BDS_B1I_D1.value]
    FREQ_B2A_BDS = [PE_API.Signal_Obs.BDS_B2A_P.value]
    FREQ_B3I_BDS = [PE_API.Signal_Obs.BDS_B3I.value]


class SIGNAL_FREQ:
    FREQS = {
        1575.42e6: FREQUENCY.FREQ_L1_GPS
        + FREQUENCY.FREQ_E1_GAL
        + FREQUENCY.FREQ_B1A_BDS
        + FREQUENCY.FREQ_B1C_BDS,
        1227.60e6: FREQUENCY.FREQ_L2_GPS,
        1176.45e6: FREQUENCY.FREQ_L5_GPS
        + FREQUENCY.FREQ_E5A_GAL
        + FREQUENCY.FREQ_B2A_BDS,
        1207.14e6: FREQUENCY.FREQ_E5B_GAL,
        1191.795e6: FREQUENCY.FREQ_E5AB_GAL,
        1278.75e6: FREQUENCY.FREQ_E6_GAL,
        1561.098e6: FREQUENCY.FREQ_B1I_BDS,
        1268.52e6: FREQUENCY.FREQ_B3I_BDS,
    }

    SIGNALS = {signal: freq for freq, signals in FREQS.items() for signal in signals}


RAW_FEATURE_LIST = [
    "cons_idx",
    "sat_idx",
    "freq_idx",
    "doy",
    "seconds_of_day",
    "delta_time",
    "code",
    "phase",
    "doppler",
    "snr",
    "elevation",
    "residual",
    "iono",
    "delta_cmc",
    "crc",
]

PREPROCESSED_TIMESERIES_LIST = [
    "doy",
    "seconds_of_day",
]

PREPROCESSED_FEATURE_LIST = PREPROCESSED_TIMESERIES_LIST + [
    "delta_time",
    "code",
    "phase",
    "doppler",
    "snr",
    "elevation",
    "residual",
    "iono",
    "delta_cmc",
    "crc",
]

PROCESSED_TIMESERIES_LIST = [
    "DOY_sin",
    "DOY_cos",
    "Day_sin",
    "Day_cos",
]

PROCESSED_FEATURE_LIST = PROCESSED_TIMESERIES_LIST + [
    "delta_time",
    "code",
    "phase",
    "doppler",
    "snr",
    "elevation",
    "residual",
    "iono",
    "delta_cmc",
    "crc",
]

RTPPP_COLUMNS = [
    "Epoch",
    "RawEpoch",
    "X",
    "Y",
    "Z",
    "LAT",
    "LON",
    "HEI",
    "ALL_Fix",
    "ALL_Float",
    "SVG_Float",
    "SVE_Float",
    "SVC_Float",
    "ALL_One_Freq",
    "QUALITY",
    "Vx",
    "Vy",
    "Vz",
    "Heading",
    "TrackAngle",
    "VEL_CORE_VALIDITY",
]
