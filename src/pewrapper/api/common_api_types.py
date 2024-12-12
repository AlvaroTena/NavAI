import ctypes as ct

from pewrapper.common.cwrapper import C_Enum

SIZE_BINEPHEM = 933506


class Signal_Obs(C_Enum):
    SIG_UNKNOWN = 0
    GPS_L1_C = 1
    GPS_L1_P = 2
    GPS_L1_CP = 3
    GPS_L1_C_DP = 4
    GPS_L2_P = 5
    GPS_L2_Z = 6
    GPS_L2_CM = 7
    GPS_L2_CL = 8
    GPS_L5_Q = 9
    GPS_L5_IQ = 10
    GAL_E1_C = 11
    GAL_E1_BC = 12
    GAL_E5A_Q = 13
    GAL_E5A_I = 14
    GAL_E5A_IQ = 15
    GAL_E5B_Q = 16
    GAL_E5B_I = 17
    GAL_E5B_IQ = 18
    GAL_E5AB_Q = 19
    GAL_E5AB_I = 20
    GAL_E5AB_IQ = 21
    GAL_E6_A = 22
    GAL_E6_B = 23
    GAL_E6_C = 24
    BDS_B1A_1D = 25
    BDS_B1I_D1 = 26  # MEO beidou satellites of B1I, D2 is GEO D1 is MEO
    BDS_B1C_P = 27
    BDS_B2A_P = 28
    BDS_B3I = 29
    MAX_ID_SIGNALS = 30


class LogCategoryPE(C_Enum):
    NONE = 0
    TRACE = 1
    DEBUG = 2
    INFO = 3
    WARNING = 4
    error = 5


Log_Handle = ct.CFUNCTYPE(
    None,
    ct.c_uint32,
    ct.c_char_p,
    ct.c_char_p,
    ct.c_char_p,
    ct.c_uint16,
    ct.c_bool,
)
f_handle_bin_ephem_NVM = ct.CFUNCTYPE(
    ct.c_bool,
    ct.c_void_p,
    ct.c_size_t,
)
