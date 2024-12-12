import ctypes as ct

from navutils.logger import Logger
from pewrapper.api import DTC_ID_List, f_handle_ReportDTCStatus

MAX_DTCS = len(DTC_ID_List)
DTCS_STRING = [
    "DTC_MIN",
    "TELEM_SERVER_DOWN_TIMEOUT",
    "TELEM_INVALID_SIGNATURE",
    "CORR_SRVC_INVALID_E2E_STATUS",
    "GNSS_SERVER_DOWN_TIMEOUT",
    "GNSS_INVALID_E2E_STATUS",
    "JAMMING_DETECTION",
    "SPOOFING_DETECTION",
    "PE_ALG_ERROR",
    "INPUT_SENSOR_ERROR",
    "INVALID_GNSS_RAW_DATA",
    "INVALID_CORR_SRVC_RAW_DATA",
    "SAFETY_CONDITION",
    "DTC_MAX",
]


class DtcManager:
    active_dtcs_ = [False] * MAX_DTCS

    @staticmethod
    @f_handle_ReportDTCStatus
    def ReportDTCStatus(mDTC: ct.c_uint32, mDTCStatus: ct.c_uint32) -> ct.c_bool:
        result = mDTC < MAX_DTCS
        if result:
            ss = f"[DTC {DTCS_STRING[mDTC]}]: {'Cleared' if DtcManager.active_dtcs_[mDTC] else 'Raised'}"
            Logger.log_message(Logger.Category.WARNING, Logger.Module.ALGORITHM, ss)

            DtcManager.active_dtcs_[mDTC] = not DtcManager.active_dtcs_[mDTC]
        else:
            Logger.log_message(
                Logger.Category.ERROR, Logger.Module.ALGORITHM, "invalid DTC type"
            )

        return result
