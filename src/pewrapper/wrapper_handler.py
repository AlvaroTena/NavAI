import re
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd

from navutils.logger import Logger
from pewrapper.api import (
    GM_Time,
    IMU_Measurements,
    Input_IMU_Measurements,
    Input_WheelSpeed_Measurements,
    Position_Engine_API,
    SafeState,
    SafeStateMachineSignal,
    SensorQualifier,
    WheelSpeedData,
)
from pewrapper.managers import ConfigurationManager, OutputStr, SafetyStateMachine
from pewrapper.managers.wrapper_data_mgr import WrapperDataManager
from pewrapper.misc import RELEASE_INFO, GetSensorQualifier, about_msg
from pewrapper.recorders import Position_Recorder
from pewrapper.types import (
    COMPUTE_INPUT_DATA_WITH_TIME,
    COMPUTE_INPUT_DATA_WITH_TIME_DELAY_TAG,
    COMPUTE_INPUT_DATA_WITHOUT_TIME,
    GPS_Time,
    PE_output_config,
)

NUMBER_IMU_MSG_FIELDS = 7
NUMBER_IMU_MSG_FIELDS_QUALIFIERS = 13
NUMBER_ODOMETER_MSG_FIELDS = 9


def get_imu_meas_array(
    input_imu: List[IMU_Measurements],
) -> Tuple[bool, Input_IMU_Measurements]:  # type: ignore
    output = Input_IMU_Measurements()
    for i in range(min(len(input_imu), len(output))):
        output[i] = input_imu[i]

    return len(input_imu) < len(output), output


def get_odo_meas_array(
    input_odo: List[WheelSpeedData],
) -> Tuple[bool, Input_WheelSpeed_Measurements]:  # type: ignore
    output = Input_WheelSpeed_Measurements()
    for i in range(min(len(input_odo), len(output))):
        output[i] = input_odo[i]

    return len(input_odo) < len(output), output


class Wrapper(ABC):

    def __init__(
        self,
        debug_level: str,
        tracing_config_file: str,
        log_path: str,
        initial_epoch_constr: GPS_Time,
        final_epoch_constr: GPS_Time,
        configMgr: ConfigurationManager = None,
        wrapper_file_data: WrapperDataManager = None,
    ):
        self.position_engine_ = Position_Engine_API()

        if not self.position_engine_.common_lib_ptr:
            Logger.log_message(
                Logger.Category.ERROR,
                "Null PE instance retrieved. Aborting execution...",
            )
            self.position_engine_.close_PE()
            exit(1)

        self.configMgr_ = (
            configMgr
            if configMgr is not None
            else ConfigurationManager(log_path, tracing_config_file)
        )
        self.wrapper_file_data_ = (
            wrapper_file_data
            if wrapper_file_data is not None
            else WrapperDataManager(
                initial_epoch_constr, final_epoch_constr, self.configMgr_
            )
        )
        self.state_machine_ = SafetyStateMachine(self.position_engine_, self.configMgr_)

        self.position_recorder_ = Position_Recorder()
        self.pe_output_config_ = PE_output_config()

    @abstractmethod
    def process_scenario(
        self,
        config_file_path: str,
        wrapper_file_path: str,
        output_path: str,
        parsing_rate: int,
    ) -> bool:
        pass

    @abstractmethod
    def _start_processing(
        self,
        output_path: str,
        pe_wrapper_commit: str,
        common_lib_commit: str,
    ) -> bool:
        pass

    def close_PE(self) -> bool:
        if hasattr(self, "position_engine_"):
            self.position_engine_.close_PE()
            del self.position_engine_

        return True

    def _activate_output_position_file(
        self, output_path: str, epoch_file: GPS_Time
    ) -> bool:
        result = False

        if self.position_engine_.common_lib_ptr:
            self.pe_output_config_.position_file = True
            self.pe_output_config_.position_path = output_path
            self.position_recorder_.initialize(output_path, epoch_file)
            result = True

        return result

    def _get_common_lib_commit_id(self) -> Tuple[bool, str]:
        if self.position_engine_:
            commit_id = self.position_engine_.get_common_lib_commit_id()
            return True, commit_id
        else:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.PE,
                f"Could not get common_lib_commit_id and has_lib_commit_id. Null pointer API instance retrieved",
            )
            return False, ""

    def _process_imu_msg(
        self, imu_msg: str, gnss_epoch: GPS_Time
    ) -> Tuple[bool, IMU_Measurements]:
        result = True
        imu_measurements = IMU_Measurements()

        imu_msg = imu_msg.strip()
        values = imu_msg.split(",")

        if (
            self.configMgr_.use_IMU_qualifier_
            and len(values) != NUMBER_IMU_MSG_FIELDS_QUALIFIERS
        ) or (
            not self.configMgr_.use_IMU_qualifier_
            and len(values) != NUMBER_IMU_MSG_FIELDS
        ):
            result = False
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.READER,
                f"Invalid number of parameters from IMU Message. Elements received: {len(values)}. Elements expected: {NUMBER_IMU_MSG_FIELDS}",
            )
        else:
            try:
                imu_measurements.sync_ns = int(float(values[0]))

                imu_measurements.linearAcceleration[0] = float(values[1])
                imu_measurements.linearAcceleration[1] = float(values[2])
                imu_measurements.linearAcceleration[2] = float(values[3])

                imu_measurements.angularVelocity[0] = np.rad2deg(float(values[4]))
                imu_measurements.angularVelocity[1] = np.rad2deg(float(values[5]))
                imu_measurements.angularVelocity[2] = np.rad2deg(float(values[6]))

            except ValueError:
                result = False
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.READER,
                    f"Invalid format in number from IMU Message. Could not convert from string to float",
                )

            if self.configMgr_.use_IMU_qualifier_:
                try:
                    imu_measurements.qualifierVelAcc = GetSensorQualifier(
                        int(float(values[7]))
                    )
                except ValueError:
                    result = False
                    Logger.log_message(
                        Logger.Category.WARNING,
                        Logger.Module.READER,
                        f"Invalid format in number from IMU Message. Could not convert from string to int",
                    )
            else:
                imu_measurements.qualifierVelAcc = SensorQualifier.VALID

        if not result:
            imu_measurements.qualifierVelAcc = SensorQualifier.INVALID

        return result, imu_measurements

    def _process_odometer_msg(
        self, odometer_msg: str, epoch: GPS_Time
    ) -> Tuple[bool, WheelSpeedData]:
        result = True
        odometer_measurements = WheelSpeedData()

        odometer_msg = odometer_msg.strip()
        values = odometer_msg.split(",")

        if len(values) >= NUMBER_ODOMETER_MSG_FIELDS:
            try:
                odometer_measurements.timestampActualValueRPMWheel = int(
                    float(values[0])
                )
                odometer_measurements.actualValuesRPMWheel[0] = float(values[1])
                odometer_measurements.actualValuesRPMWheel[1] = float(values[2])
                odometer_measurements.actualValuesRPMWheel[2] = float(values[3])
                odometer_measurements.actualValuesRPMWheel[3] = float(values[4])
            except ValueError:
                result = False
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.READER,
                    f"Invalid format in number from ODO Message. Could not convert from string to float",
                )

            try:
                odometer_measurements.qualifierActualValuesRPMWheel = (
                    GetSensorQualifier(int(float(values[5])))
                )
            except ValueError:
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.READER,
                    f"Invalid format in number from ODO Message. Could not convert from string to int",
                )

        else:
            result = False
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.READER,
                f"Invalid number of parameters from ODO Message. Elements received: {len(values)}. Elements expected: {NUMBER_ODOMETER_MSG_FIELDS}",
            )

        if not result:
            odometer_measurements.qualifierActualValuesRPMWheel = SensorQualifier.N_A

        return result, odometer_measurements

    def _load_buffered_sensors(
        self,
        imu_buffer: List[IMU_Measurements],
        odo_buffer: List[WheelSpeedData],
    ) -> bool:
        # Load imu meas
        if not (result := get_imu_meas_array(imu_buffer))[0]:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.MAIN,
                f"Error loading IMU, number of samples greater than input buffer size",
            )
        _, imu_measurements_array = result

        if len(imu_buffer) != 0 and not self.position_engine_.LoadImuMessage(
            imu_measurements_array
        ):
            return False

        # Load odo meas
        if not (result := get_odo_meas_array(odo_buffer))[0]:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.MAIN,
                f"Error loading ODO, number of samples greater than input buffer size",
            )
        _, odo_measurements_array = result

        if len(odo_buffer) != 0 and not self.position_engine_.LoadWheelSpeedData(
            odo_measurements_array
        ):
            return False

        return True

    def _perform_reconvergences(
        self,
        epoch: GPS_Time,
        output_path: str,
        pe_wrapper_commit: str,
        common_lib_commit: str,
    ):
        self.configMgr_.reconvergence_last_epoch_ = epoch
        self.state_machine_.ProcessSignal(SafeStateMachineSignal.NO_SOLUTION)

        if self.configMgr_.reconvergences_multiple_files_:
            self.position_recorder_.close_file()
            self._activate_output_position_file(
                output_path, self.configMgr_.reconvergence_last_epoch_
            )
            self.position_recorder_.write_pos_header(
                pe_wrapper_commit, common_lib_commit
            )

    def _get_compute_time(self, compute_msg: str) -> Tuple[bool, GM_Time]:
        result = True
        gm_time = GM_Time()

        compute_msg = compute_msg.strip()
        values = re.split(r"\s+", compute_msg.strip())

        if len(values) == COMPUTE_INPUT_DATA_WITHOUT_TIME:
            gm_time.valid = False
            gm_time.value = 0.0

        elif len(values) == COMPUTE_INPUT_DATA_WITH_TIME:
            try:
                gm_time.value = float(values[1])
                gm_time.valid = True
            except ValueError:
                result = False
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.READER,
                    f"Invalid format in GM_Time from COMPUTE Message. Could not convert from string to double",
                )

        elif len(values) == COMPUTE_INPUT_DATA_WITH_TIME_DELAY_TAG:
            gm_time.valid = True

        else:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.READER,
                f"Invalid number of fields in COMPUTE Message. Could not convert from string to double",
            )

        return result, gm_time

    @abstractmethod
    def _compute(
        self,
        epoch: GPS_Time,
        epoch_str: str,
        wrapper_epoch_data: pd.DataFrame,
        imu_buffer: List[IMU_Measurements],
        odo_buffer: List[WheelSpeedData],
        pe_output: OutputStr,
    ) -> Tuple[bool, List[IMU_Measurements], List[WheelSpeedData], OutputStr]:
        pass


class Wrapper_Handler(Wrapper):
    def __init__(
        self,
        debug_level: str,
        tracing_config_file: str,
        log_path: str,
        initial_epoch_constr: GPS_Time,
        final_epoch_constr: GPS_Time,
    ):
        super().__init__(
            debug_level,
            tracing_config_file,
            log_path,
            initial_epoch_constr,
            final_epoch_constr,
        )

    def process_scenario(
        self,
        config_file_path: str,
        wrapper_file_path: str,
        output_path: str,
        parsing_rate: int,
    ) -> bool:
        addInfo = ""

        pe_wrapper_commit_id = about_msg()
        if not (result := self._get_common_lib_commit_id())[0]:
            return False
        _, common_lib_commit_id = result

        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"{RELEASE_INFO}, {pe_wrapper_commit_id}, {common_lib_commit_id} started",
        )

        ################################################
        #################  PROCESSING  #################
        ################################################

        if not (result := self.configMgr_.parse_config_file(config_file_path))[0]:
            _, addInfo = result
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f" Error processing config file: {addInfo}",
            )
            return False

        if not (
            result := self.wrapper_file_data_.parse_wrapper_file(
                wrapper_file_path, parsing_rate
            )
        )[0]:
            _, addInfo = result
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f" Error processing wrapper file: {addInfo}",
            )
            return False

        if not self._start_processing(
            output_path,
            pe_wrapper_commit_id,
            common_lib_commit_id,
        ):
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f" Error processing PE: {addInfo}",
            )
            return False

        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f" Wrapper processing finished!",
        )
        return True

    def _start_processing(
        self,
        output_path: str,
        pe_wrapper_commit: str,
        common_lib_commit: str,
    ) -> bool:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f" Launching wrapper file data processing",
        )

        pe_output = OutputStr()
        self.state_machine_.reset()

        if not self.state_machine_.ProcessSignal(SafeStateMachineSignal.START_SIGNAL):
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"Could not initialize Safe State Machine",
            )
            return False

        if not (result := self.position_engine_.GetGnssCdVersion())[0]:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f" Error getting GnssCdVersion string",
            )
            return False
        _, cs_version = result

        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f" API GnssCdVersion: {cs_version}",
        )

        result_pe, pe_output.output_PE.pe_solution_info.SSM_Signal = (
            self.position_engine_.Reboot(
                self.configMgr_.get_config(),
                pe_output.output_PE.pe_solution_info.SSM_Signal,
            )
        )
        self.position_engine_.init_log_PE(self.configMgr_.get_config())

        result_pe &= self.state_machine_.ProcessSignal(pe_output)

        if (
            not result_pe
            or self.state_machine_.GetCurrentState() == SafeState.ERROR_STATE
        ):
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.ALGORITHM,
                f"Could not initialize Position Engine",
            )
            return False

        if self.wrapper_file_data_:
            self._activate_output_position_file(
                output_path, self.wrapper_file_data_.initial_epoch
            )

        self.position_recorder_.write_pos_header(pe_wrapper_commit, common_lib_commit)

        imu_buffer: List[IMU_Measurements] = []
        odo_buffer: List[WheelSpeedData] = []

        if self.configMgr_.reconvergences_:
            self.configMgr_.reconvergence_last_epoch_ = (
                self.wrapper_file_data_.initial_epoch
            )

        for epoch, wrapper_epoch_data in self.wrapper_file_data_:
            pe_output.reset()

            epoch_str = epoch.calendar_column_str_d()

            if any(cs_data_filled := wrapper_epoch_data["msg_type"] == "CS"):
                cs_msg = wrapper_epoch_data.loc[cs_data_filled].iloc[-1]["msg_data"]
                Logger.log_message(
                    Logger.Category.TRACE,
                    Logger.Module.MAIN,
                    f"Loading CS msg  {epoch_str} {len(cs_msg)} ( {cs_msg} )",
                )

                msg = bytes.fromhex(cs_msg)
                if not self.position_engine_.LoadCorrectionServiceMessage(
                    msg, len(msg)
                ):
                    return False

            if any(cp_data_filled := wrapper_epoch_data["msg_type"] == "CP"):
                cp_msg = wrapper_epoch_data.loc[cp_data_filled].iloc[-1]["msg_data"]
                Logger.log_message(
                    Logger.Category.TRACE,
                    Logger.Module.ALGORITHM,
                    f"Loading CP msg  {epoch_str} {len(cp_msg)} ( {cp_msg} )",
                )

                msg = bytes.fromhex(cp_msg)
                if not self.position_engine_.LoadGnssMessage(msg, len(msg)):
                    return False

            if any(imu_data_filled := wrapper_epoch_data["msg_type"] == "IMU"):
                _, imu_measurement = self._process_imu_msg(
                    wrapper_epoch_data.loc[imu_data_filled].iloc[-1]["msg_data"], epoch
                )
                imu_buffer.append(imu_measurement)

            if any(odometer_data_filled := wrapper_epoch_data["msg_type"] == "ODO"):
                _, odometer_measurement = self._process_odometer_msg(
                    wrapper_epoch_data.loc[odometer_data_filled].iloc[-1]["msg_data"],
                    epoch,
                )
                odo_buffer.append(odometer_measurement)

            if any(gnss_data_filled := wrapper_epoch_data["msg_type"] == "GNSS"):
                gnss_msg = wrapper_epoch_data.loc[gnss_data_filled].iloc[-1]["msg_data"]
                msg = bytes.fromhex(gnss_msg)
                if not self.position_engine_.LoadGnssMessage(msg, len(msg)):
                    return False

                Logger.log_message(
                    Logger.Category.TRACE,
                    Logger.Module.MAIN,
                    f"Launching process order {epoch_str}",
                )

                if not self.configMgr_.compute_log_:
                    if self.configMgr_.reconvergences_ and (
                        (epoch - self.configMgr_.reconvergence_last_epoch_)
                        > self.configMgr_.reconvergences_reset_rate_
                    ):
                        self._perform_reconvergences(
                            epoch, output_path, pe_wrapper_commit, common_lib_commit
                        )

                    if not (
                        result := self._compute(
                            epoch,
                            epoch_str,
                            wrapper_epoch_data,
                            imu_buffer,
                            odo_buffer,
                            pe_output,
                        )
                    )[0]:
                        return False
                    _, imu_buffer, odo_buffer, pe_output = result

            if (
                any(wrapper_epoch_data["msg_type"] == "COMPUTE")
                and self.configMgr_.compute_log_
            ):
                if any(wrapper_epoch_data["msg_type"] == "COMPUTE_RESET"):
                    self.state_machine_.ProcessSignal(
                        SafeStateMachineSignal.NO_SOLUTION
                    )

                elif self.configMgr_.reconvergences_ and (
                    (epoch - self.configMgr_.reconvergence_last_epoch_)
                    > self.configMgr_.reconvergences_reset_rate_
                ):
                    self._perform_reconvergences(
                        epoch, output_path, pe_wrapper_commit, common_lib_commit
                    )

                if not (
                    result := self._compute(
                        epoch,
                        epoch_str,
                        wrapper_epoch_data,
                        imu_buffer,
                        odo_buffer,
                        pe_output,
                    )
                )[0]:
                    return False
                _, imu_buffer, odo_buffer, pe_output = result

        if not self.state_machine_.ProcessSignal(
            SafeStateMachineSignal.TERMINATE_SIGNAL
        ):
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.MAIN,
                f"Error terminating processing",
            )
            return False

        self.position_recorder_.close_file()

        return True

    def _compute(
        self,
        epoch: GPS_Time,
        epoch_str: str,
        wrapper_epoch_data: pd.DataFrame,
        imu_buffer: List[IMU_Measurements],
        odo_buffer: List[WheelSpeedData],
        pe_output: OutputStr,
    ) -> Tuple[bool, List[IMU_Measurements], List[WheelSpeedData], OutputStr]:
        # Load Sensors Buffers
        if not self._load_buffered_sensors(imu_buffer, odo_buffer):
            return False, imu_buffer, odo_buffer, pe_output

        imu_buffer.clear()
        odo_buffer.clear()

        gm_time = GM_Time()

        if not self.configMgr_.compute_log_:
            gm_time.valid = True
            gm_time.value = epoch.get_GPS_abs_seconds()

        elif (
            any(compute_filled := wrapper_epoch_data["msg_type"] == "COMPUTE")
            and self.configMgr_.compute_log_
        ):
            if not (
                result := self._get_compute_time(
                    wrapper_epoch_data.loc[compute_filled].iloc[-1]["msg_data"]
                )
            )[0]:
                _, gm_time = result
                if gm_time.valid:
                    gm_time.value = epoch.get_GPS_abs_seconds()

                else:
                    return False, imu_buffer, odo_buffer, pe_output
            _, gm_time = result

        else:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.ALGORITHM,
                f"Error getting GM_TIME {epoch_str}",
            )

        result_pe, pe_output.output_PE = self.position_engine_.Compute(
            self.state_machine_.GetCurrentState(), gm_time, pe_output.output_PE
        )
        result_pe &= self.state_machine_.ProcessSignal(pe_output)

        if (
            not result_pe
            or self.state_machine_.GetCurrentState() == SafeState.ERROR_STATE
        ):
            return False, imu_buffer, odo_buffer, pe_output
        else:
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.MAIN,
                f" Processed output {epoch_str} (  )",
            )

            is_qm = self.configMgr_.config_info_.use_qm_variant
            self.position_recorder_.write_pos_epoch(epoch, is_qm, pe_output)

        return True, imu_buffer, odo_buffer, pe_output
