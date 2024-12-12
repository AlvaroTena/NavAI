import ctypes as ct
import os
import sys
from typing import ByteString, Tuple

from navutils.logger import Logger
from pewrapper.api import (
    ApiVersion,
    Configuration_info,
    Event,
    GM_Time,
    Input_IMU_Measurements,
    Input_WheelSpeed_Measurements,
    PE_API_FeaturesAI,
    PE_API_PredictionsAI,
    PE_Output_str,
    SafeState,
    SafeStateMachineSignal,
)


class Position_Engine_API:
    MAX_CD_VERSION_STR_SIZE = 18

    def __init__(self, lib_path=os.getenv("LD_LIBRARY_PATH")):
        try:
            self.common_lib = ct.CDLL(
                os.path.join(lib_path, "libcommon_lib_PE_develop.so")
            )
        except OSError as e:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.NONE,
                f"Error to load shared library: {e}",
            )
            sys.exit(1)
        else:
            self.common_lib.get_PE_API_version.restype = ct.c_char_p

            pe_api_version = self.common_lib.get_PE_API_version()
            pe_api_version = (
                pe_api_version.decode("utf-8") if pe_api_version is not None else ""
            )
            wrapper_ai_pe_api_version_ = ApiVersion()

            if pe_api_version != wrapper_ai_pe_api_version_.GetApiVersion():
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.PE,
                    f"PE API versions mismatch. PE_commonlib_API version = {pe_api_version}. PE_Wrapper_API version = {wrapper_ai_pe_api_version_.GetApiVersion()}",
                )
                return

            self.common_lib.get_PE_API.restype = ct.c_void_p

            self.common_lib.Reboot.argtypes = [
                ct.c_void_p,
                ct.POINTER(Configuration_info),
                ct.POINTER(ct.c_uint32),
            ]
            self.common_lib.Reboot.restype = ct.c_bool

            self.common_lib.LoadCorrectionServiceMessage.argtypes = [
                ct.c_void_p,
                ct.POINTER(ct.c_uint8),
                ct.c_size_t,
            ]
            self.common_lib.LoadCorrectionServiceMessage.restype = ct.c_bool

            self.common_lib.GetGnssCdVersion.argtypes = [
                ct.c_void_p,
                ct.POINTER(ct.c_char),
                ct.c_size_t,
            ]
            self.common_lib.GetGnssCdVersion.restype = ct.c_bool

            self.common_lib.LoadGnssMessage.argtypes = [
                ct.c_void_p,
                ct.POINTER(ct.c_uint8),
                ct.c_size_t,
            ]
            self.common_lib.LoadGnssMessage.restype = ct.c_bool

            self.common_lib.LoadImuMessage.argtypes = [
                ct.c_void_p,
                ct.POINTER(Input_IMU_Measurements),
            ]
            self.common_lib.LoadImuMessage.restype = ct.c_bool

            self.common_lib.LoadWheelSpeedData.argtypes = [
                ct.c_void_p,
                ct.POINTER(Input_WheelSpeed_Measurements),
            ]
            self.common_lib.LoadWheelSpeedData.restype = ct.c_bool

            self.common_lib.PreCompute.argtypes = [
                ct.c_void_p,
                ct.POINTER(GM_Time),
                ct.POINTER(PE_Output_str),
                ct.POINTER(PE_API_FeaturesAI),
            ]
            self.common_lib.PreCompute.restype = ct.c_bool

            self.common_lib.Compute.argtypes = [
                ct.c_void_p,
                ct.c_uint32,
                ct.POINTER(PE_Output_str),
                ct.POINTER(PE_Output_str),
            ]
            self.common_lib.Compute.restype = ct.c_bool

            self.common_lib.LoadPredictionsAI.argtypes = [
                ct.c_void_p,
                ct.POINTER(PE_API_PredictionsAI),
            ]
            self.common_lib.LoadPredictionsAI.restype = ct.c_bool

            self.common_lib.ResetKalmanFilter.argtypes = [
                ct.c_void_p,
                ct.c_uint32,
            ]
            self.common_lib.ResetKalmanFilter.restype = None

            self.common_lib.TriggerEvent.argtypes = [
                ct.c_void_p,
                ct.c_uint32,
            ]
            self.common_lib.TriggerEvent.restype = ct.c_bool

            self.common_lib.get_common_lib_commit_id.argtypes = [
                ct.c_void_p,
                ct.POINTER(ct.c_char),
                ct.c_size_t,
            ]
            self.common_lib.get_common_lib_commit_id.restype = ct.c_char_p

            self.common_lib.init_log_PE.argtypes = [
                ct.c_void_p,
                ct.POINTER(Configuration_info),
            ]
            self.common_lib.init_log_PE.restype = None

            self.common_lib.closing_PE.argtypes = [
                ct.c_void_p,
            ]
            self.common_lib.closing_PE.restype = None

            self.common_lib.destroy_PE_API.argtypes = [
                ct.c_void_p,
            ]
            self.common_lib.destroy_PE_API.restype = None

            self.common_lib_ptr = self.common_lib.get_PE_API()

    def __del__(self):
        if hasattr(self, "common_lib") and self.common_lib is not None:
            self.common_lib.destroy_PE_API(self.common_lib_ptr)

    def Reboot(
        self, config_info: Configuration_info, flag: SafeStateMachineSignal
    ) -> Tuple[bool, SafeStateMachineSignal]:
        flag_ct = ct.c_uint32(flag)
        result = self.common_lib.Reboot(
            self.common_lib_ptr,
            ct.byref(config_info),
            ct.byref(flag_ct),
        )
        return result, flag_ct.value

    def LoadCorrectionServiceMessage(self, msg: ByteString, msg_length: int) -> bool:
        msg_array = (ct.c_uint8 * msg_length)(*msg)
        return self.common_lib.LoadCorrectionServiceMessage(
            self.common_lib_ptr, msg_array, msg_length
        )

    def GetGnssCdVersion(self) -> Tuple[bool, str]:
        cs_version_buffer = ct.create_string_buffer(
            Position_Engine_API.MAX_CD_VERSION_STR_SIZE
        )
        result = self.common_lib.GetGnssCdVersion(
            self.common_lib_ptr,
            cs_version_buffer,
            Position_Engine_API.MAX_CD_VERSION_STR_SIZE,
        )
        cs_version = cs_version_buffer.value.decode("utf-8")

        return result, cs_version

    def LoadGnssMessage(self, msg: ByteString, msg_length: int) -> bool:
        msg_array = (ct.c_uint8 * msg_length)(*msg)
        return self.common_lib.LoadGnssMessage(
            self.common_lib_ptr, msg_array, msg_length
        )

    def LoadImuMessage(
        self,
        imu_data_array: Input_IMU_Measurements,  # type: ignore
    ) -> bool:
        return self.common_lib.LoadImuMessage(
            self.common_lib_ptr, ct.byref(imu_data_array)
        )

    def LoadWheelSpeedData(
        self,
        wheel_speed_data_array: Input_WheelSpeed_Measurements,  # type: ignore
    ) -> bool:
        return self.common_lib.LoadImuMessage(
            self.common_lib_ptr, ct.byref(wheel_speed_data_array)
        )

    def PreCompute(
        self,
        gm_time: GM_Time,
        peOutputStrPVT: PE_Output_str,
        featuresMP: PE_API_FeaturesAI,  # type: ignore
    ) -> Tuple[bool, PE_Output_str, PE_API_FeaturesAI]:  # type: ignore
        return (
            self.common_lib.PreCompute(
                self.common_lib_ptr,
                ct.byref(gm_time),
                ct.byref(peOutputStrPVT),
                ct.byref(featuresMP),
            ),
            peOutputStrPVT,
            featuresMP,
        )

    def Compute(
        self,
        ssm_state: SafeState,
        peOutputStrPVT: PE_Output_str,
        peOutputStr: PE_Output_str,
    ) -> Tuple[bool, PE_Output_str]:
        return (
            self.common_lib.Compute(
                self.common_lib_ptr,
                ssm_state,
                ct.byref(peOutputStrPVT),
                ct.byref(peOutputStr),
            ),
            peOutputStr,
        )

    def LoadPredictionsAI(
        self,
        predictions: PE_API_PredictionsAI,  # type: ignore
    ) -> bool:
        return self.common_lib.LoadPredictionsAI(
            self.common_lib_ptr, ct.byref(predictions)
        )

    def ResetKalmanFilter(self, SSM_signal: SafeStateMachineSignal) -> None:
        return self.common_lib.ResetKalmanFilter(self.common_lib_ptr, SSM_signal)

    def TriggerEvent(self, event: Event) -> bool:
        return self.common_lib.TriggerEvent(self.common_lib_ptr, event)

    def get_common_lib_commit_id(self) -> str:
        commit_id = ct.create_string_buffer(31)
        result = self.common_lib.get_common_lib_commit_id(
            self.common_lib_ptr, commit_id, 31
        )
        return commit_id.value.decode("utf-8") if commit_id is not None else ""

    def init_log_PE(self, config_info: Configuration_info) -> None:
        self.common_lib.init_log_PE(self.common_lib_ptr, ct.byref(config_info))

    def close_PE(self) -> None:
        if self.common_lib and self.common_lib_ptr:
            return self.common_lib.closing_PE(self.common_lib_ptr)
