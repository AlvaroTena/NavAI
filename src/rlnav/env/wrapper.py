import copy
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import pewrapper.types.constants as pe_const
import rlnav.types.constants as const
from navutils.logger import Logger
from pewrapper.api.pe_api_types import (
    GM_Time,
    IMU_Measurements,
    PE_API_FeaturesAI,
    PE_API_PredictionsAI,
    PE_Output_str,
    PE_PredictionsAI,
    ResetStruct,
    SafeState,
    SafeStateMachineSignal,
    WheelSpeedData,
)
from pewrapper.managers import ConfigurationManager, OutputStr
from pewrapper.managers.wrapper_data_mgr import WrapperDataManager
from pewrapper.misc.version_wrapper_bin import RELEASE_INFO, about_msg
from pewrapper.types.gps_time_wrapper import GPS_Time
from pewrapper.wrapper_handler import Wrapper
from rlnav.data.reader import Reader
from rlnav.managers.reward_mgr import RewardManager
from rlnav.utils.common import get_global_sat_idx


class WrapperDataAttributeError(AttributeError):
    pass


class RLWrapper(Wrapper):
    def __init__(
        self,
        configMgr: ConfigurationManager,
        wrapper_file_data: WrapperDataManager,
        rewardMgr: RewardManager,
        use_AI: bool = True,
        generation: int = None,
    ):
        super().__init__(
            "",
            "",
            "",
            None,
            None,
            configMgr=configMgr,
            wrapper_file_data=wrapper_file_data,
        )
        self.rewardMgr = rewardMgr
        self.use_AI = use_AI
        self.generation = generation

        self.reset_times()

    def reset(
        self,
        configMgr: ConfigurationManager,
        wrapper_file_data: WrapperDataManager,
        rewardMgr: RewardManager,
        use_AI: bool = True,
        generation: int = None,
    ):
        self.__init__(configMgr, wrapper_file_data, rewardMgr, use_AI, generation)

    def reset_times(self):
        self._times = {}
        self._times["start_processing"] = []
        self._times["process_epoch"] = []
        self._times["precompute"] = []
        self._times["compute"] = []
        self._times["get_featuresAI"] = []
        self._times["load_predictionAI"] = []

    def get_times(self):
        times = copy.deepcopy(self._times)
        self.reset_times()
        return times

    def process_scenario(
        self,
        config_file_path: str,
        wrapper_file_path: str,
        output_path: str,
        parsing_rate: int,
    ):
        prev_pe_epoch = GPS_Time()
        pe_state = ""

        if not (result := self._get_common_lib_commit_id())[0]:
            raise ReferenceError("PE_Wrapper")
        _, pe_common_lib_commit_id = result

        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"{RELEASE_INFO}, Commit ID RL_Wrapper: {(ai_pe_wrapper_commit_id:=about_msg())}, {pe_common_lib_commit_id}, started",
        )

        if not self._start_processing(
            output_path,
            ai_pe_wrapper_commit_id,
            pe_common_lib_commit_id.split(" ")[-1],
        ):
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.WRAPPER,
                f"Error processing PE: ",
            )
            if not self.close_PE():
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.WRAPPER,
                    f"Error closing files of PE: ",
                )
            raise RuntimeError

        while pe_state != "finished_wrapper":
            if not (result := self.process_epoch())[0]:
                _, pe_state, _ = result
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.WRAPPER,
                    f"Error processing PE: {pe_state}",
                )
                return False, None

            _, pe_state, pe_pvt = result
            del result

            pe_epoch = GPS_Time(w=pe_pvt.timestamp_week, s=pe_pvt.timestamp_second)

            if prev_pe_epoch > pe_epoch:
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.WRAPPER,
                    f"PE_Wrapper processed old epoch: {pe_epoch.calendar_column_str_d()} | Prev epoch: {prev_pe_epoch.calendar_column_str_d()}",
                )

            prev_pe_epoch = pe_epoch

            if pe_state != "finished_wrapper":
                if not (result := self.compute())[0]:
                    Logger.log_message(
                        Logger.Category.ERROR,
                        Logger.Module.WRAPPER,
                        f"Error processing PE: ",
                    )
                    return False, None
                _, pe_out = result
                if (
                    GPS_Time(
                        w=pe_out.output_PE.timestamp_week,
                        s=pe_out.output_PE.timestamp_second,
                    )
                    != GPS_Time()
                ):
                    self.rewardMgr.update_base(pe_out)

        if not self.state_machine_.ProcessSignal(
            SafeStateMachineSignal.TERMINATE_SIGNAL
        ):
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.WRAPPER,
                f"Error terminating processing",
            )
            return False, None

        self.position_recorder_.close_file()

        pe_errors = self.rewardMgr.calculate_propagated_base_positions()

        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.WRAPPER,
            f" Wrapper processing finished!",
        )
        return True, pe_errors

    def _start_processing(
        self,
        output_path: str,
        pe_wrapper_commit: str,
        common_lib_commit: str,
    ) -> bool:
        start = time.time()

        self.output_path = output_path
        self.pe_wrapper_commit = pe_wrapper_commit
        self.common_lib_commit = common_lib_commit

        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.WRAPPER,
            f" Launching wrapper file data processing",
        )

        self.pe_output = OutputStr()
        self.pvt_output = OutputStr()

        self.featuresMP = PE_API_FeaturesAI()

        self.state_machine_.reset()

        if not self.state_machine_.ProcessSignal(SafeStateMachineSignal.START_SIGNAL):
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.STATE_MACHINE,
                f"Could not initialize Safe State Machine",
            )
            return False

        if not (result := self.position_engine_.GetGnssCdVersion())[0]:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.WRAPPER,
                f" Error getting GnssCdVersion string",
            )
            return False
        _, cs_version = result

        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f" API GnssCdVersion: {cs_version}",
        )

        config = self.configMgr_.get_config(
            self.output_path if self.use_AI else None, self.use_AI, self.generation
        )
        os.makedirs(config.log_path.decode("utf-8"), exist_ok=True)

        result_pe, self.pe_output.output_PE.pe_solution_info.SSM_Signal = (
            self.position_engine_.Reboot(
                config,
                self.pe_output.output_PE.pe_solution_info.SSM_Signal,
            )
        )
        self.position_engine_.init_log_PE(config)

        result_pe &= self.state_machine_.ProcessSignal(self.pe_output)

        if (
            not result_pe
            or self.state_machine_.GetCurrentState() == SafeState.ERROR_STATE
        ):
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.WRAPPER,
                f"Could not initialize Position Engine",
            )
            return False

        self._activate_output_position_file(
            config.log_path.decode("utf-8"), self.wrapper_file_data_.initial_epoch
        )
        self.position_recorder_.write_pos_header(pe_wrapper_commit, common_lib_commit)

        self.wrapper_data = iter(self.wrapper_file_data_)
        self.imu_buffer: List[IMU_Measurements] = []
        self.odo_buffer: List[WheelSpeedData] = []

        if True:  #!self.configMgr_.reconvergences_:
            self.configMgr_.reconvergence_last_epoch_ = (
                self.wrapper_file_data_.initial_epoch
            )

        self._times["start_processing"].append(time.time() - start)
        return result_pe

    def process_epoch(self) -> Tuple[bool, str, PE_Output_str]:
        start = time.time()
        state = ""
        epoch = GPS_Time()

        if not hasattr(self, "wrapper_data"):
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.WRAPPER,
                f"Wrapper not initialized",
            )
            state = "initialized_needed"
            raise WrapperDataAttributeError(state)

        try:
            while state not in ["action_needed", "compute_needed"]:
                wrapper_data = next(self.wrapper_data)

                epoch = GPS_Time(unix_seconds=wrapper_data[0].timestamp())
                wrapper_epoch_data = wrapper_data[1]

                self.pe_output.reset()
                self.pvt_output.reset()

                ResetStruct(self.featuresMP)

                epoch_str = epoch.calendar_column_str_d()

                self._process_data(epoch, epoch_str, wrapper_epoch_data)

                if (
                    not self.configMgr_.compute_log_
                    and any(wrapper_epoch_data["msg_type"] == "GNSS")
                ) or (
                    any(wrapper_epoch_data["msg_type"] == "COMPUTE")
                    and self.configMgr_.compute_log_
                ):
                    if any(wrapper_epoch_data["msg_type"] == "COMPUTE_RESET"):
                        self.state_machine_.ProcessSignal(
                            SafeStateMachineSignal.NO_SOLUTION
                        )

                    # elif self.configMgr_.reconvergences_ and (
                    #     (epoch - self.configMgr_.reconvergence_last_epoch_)
                    #     > self.configMgr_.reconvergences_reset_rate_
                    # ):
                    elif self.rewardMgr.check_reconvergence() and (
                        (epoch - self.configMgr_.reconvergence_last_epoch_) > 600
                    ):
                        self._perform_reconvergences(
                            epoch,
                            self.output_path,
                            self.pe_wrapper_commit,
                            self.common_lib_commit,
                        )
                        self.rewardMgr.reward.reset()

                    if not (
                        result := self._precompute(
                            epoch,
                            epoch_str,
                            wrapper_epoch_data,
                            self.imu_buffer,
                            self.odo_buffer,
                            self.pvt_output,
                        )
                    )[0]:
                        _, _, _, self.pvt_output = result
                        return False, "failed_precompute", self.pvt_output.output_PE

                    _, self.imu_buffer, self.odo_buffer, self.pvt_output = result
                    state = "action_needed" if self.use_AI else "compute_needed"

        except StopIteration:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.WRAPPER,
                f"Finished wrapper data",
            )
            del self.wrapper_data
            state = "finished_wrapper"

            if not self.state_machine_.ProcessSignal(
                SafeStateMachineSignal.TERMINATE_SIGNAL
            ):
                Logger.log_message(
                    Logger.Category.DEBUG,
                    Logger.Module.WRAPPER,
                    f"Error terminating processing",
                )
                return False, state, self.pvt_output.output_PE

            self.position_recorder_.close_file()

            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.WRAPPER,
                f" Wrapper processing finished!",
            )

        if state == "":
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.WRAPPER,
                "Wrapper data bad formed",
            )
            state = "data_error"
            return False, state, self.pvt_output.output_PE

        self._times["process_epoch"].append(time.time() - start)
        return True, state, self.pvt_output.output_PE

    def _process_data(
        self,
        epoch: GPS_Time,
        epoch_str: str,
        wrapper_epoch_data: pd.DataFrame,
    ) -> bool:
        if any(cs_data_filled := wrapper_epoch_data["msg_type"] == "CS"):
            cs_msg = wrapper_epoch_data.loc[cs_data_filled].iloc[-1]["msg_data"]
            Logger.log_message(
                Logger.Category.TRACE,
                Logger.Module.WRAPPER,
                f"Loading CS msg  {epoch_str} {len(cs_msg)} ( {cs_msg} )",
            )

            msg = bytes.fromhex(cs_msg)
            if not self.position_engine_.LoadCorrectionServiceMessage(msg, len(msg)):
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
            self.imu_buffer.append(imu_measurement)

        if any(odometer_data_filled := wrapper_epoch_data["msg_type"] == "ODO"):
            _, odometer_measurement = self._process_odometer_msg(
                wrapper_epoch_data.loc[odometer_data_filled].iloc[-1]["msg_data"], epoch
            )
            self.odo_buffer.append(odometer_measurement)

        if any(gnss_data_filled := wrapper_epoch_data["msg_type"] == "GNSS"):
            gnss_msg = wrapper_epoch_data.loc[gnss_data_filled].iloc[-1]["msg_data"]
            msg = bytes.fromhex(gnss_msg)
            if not self.position_engine_.LoadGnssMessage(msg, len(msg)):
                return False

        return True

    def _precompute(
        self,
        epoch: GPS_Time,
        epoch_str: str,
        wrapper_epoch_data: pd.DataFrame,
        imu_buffer: List[IMU_Measurements],
        odo_buffer: List[WheelSpeedData],
        pvt_output: OutputStr,
    ) -> Tuple[bool, List[IMU_Measurements], List[WheelSpeedData], OutputStr]:
        start = time.time()

        Logger.log_message(
            Logger.Category.TRACE,
            Logger.Module.WRAPPER,
            f"Launching process order {epoch_str}",
        )

        # Load Sensors Buffers
        if not self._load_buffered_sensors(imu_buffer, odo_buffer):
            return (False, imu_buffer, odo_buffer, pvt_output)

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
                    return False, imu_buffer, odo_buffer, pvt_output
            _, gm_time = result

        else:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.ALGORITHM,
                f"Error getting GM_TIME {epoch_str}",
            )

        result_pe, pvt_output.output_PE, self.featuresMP = (
            self.position_engine_.PreCompute(
                gm_time, pvt_output.output_PE, self.featuresMP
            )
        )

        self._times["precompute"].append(time.time() - start)
        return (
            result_pe,
            imu_buffer,
            odo_buffer,
            pvt_output,
        )

    def compute(self) -> Tuple[bool, OutputStr]:
        start = time.time()
        epoch = GPS_Time(
            w=self.pvt_output.output_PE.timestamp_week,
            s=self.pvt_output.output_PE.timestamp_second,
        )
        result, _, _, self.pe_output = self._compute(
            epoch, epoch.calendar_column_str_d(), None, None, None, self.pe_output
        )

        self._times["compute"].append(time.time() - start)
        return result, self.pe_output

    def _compute(
        self,
        epoch: GPS_Time,
        epoch_str: str,
        wrapper_epoch_data: pd.DataFrame,
        imu_buffer: List[IMU_Measurements],
        odo_buffer: List[WheelSpeedData],
        pe_output: OutputStr,
    ) -> Tuple[bool, List[IMU_Measurements], List[WheelSpeedData], OutputStr]:
        result_pe, pe_output.output_PE = self.position_engine_.Compute(
            self.state_machine_.GetCurrentState(),
            self.pvt_output.output_PE,
            pe_output.output_PE,
        )
        result_pe &= self.state_machine_.ProcessSignal(pe_output)

        if (
            not result_pe
            or self.state_machine_.GetCurrentState() == SafeState.ERROR_STATE
        ):
            return False, None, None, pe_output

        else:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.WRAPPER,
                f" Processed output {epoch_str} (  )",
            )

            is_qm = self.configMgr_.config_info_.use_qm_variant
            self.position_recorder_.write_pos_epoch(epoch, is_qm, pe_output)

        return result_pe, None, None, pe_output

    def get_features_AI(self) -> pd.DataFrame:
        start = time.time()
        rows = []

        for sat_group in self.featuresMP:
            for feature in sat_group:
                if feature.usable:
                    epoch_str = GPS_Time(
                        feature.timestamp_week, feature.timestamp_second
                    ).calendar_column_str_d()
                    epoch_datetime = pd.to_datetime(
                        epoch_str, format="%Y %m %d %H %M %S.%f"
                    )

                    if feature.constel == pe_const.E_CONSTELLATION.E_CONSTEL_GPS:
                        sat_id = f"G{str(feature.sat_id).zfill(2)}"
                    elif feature.constel == pe_const.E_CONSTELLATION.E_CONSTEL_GAL:
                        sat_id = f"E{str(feature.sat_id).zfill(2)}"
                    elif feature.constel == pe_const.E_CONSTELLATION.E_CONSTEL_BDS:
                        sat_id = f"B{str(feature.sat_id).zfill(2)}"
                    else:
                        continue

                    row = {
                        "epoch": epoch_datetime,
                        "sat_id": sat_id,
                        "freq": feature.freq,
                        "code": feature.code,
                        "phase": feature.phase,
                        "doppler": feature.doppler,
                        "snr": feature.snr,
                        "elevation": feature.elevation,
                        "residual": feature.residual,
                        "iono": feature.iono,
                        "delta_cmc": feature.delta_cmc,
                        "crc": feature.crc,
                        "delta_time": feature.delta_time,
                    }
                    rows.append(row)

        self._times["get_featuresAI"].append(time.time() - start)
        return pd.DataFrame(rows, columns=Reader.OUT_COLUMN_NAMES)

    def load_predictions_AI(self, epoch: GPS_Time, predictions: np.ndarray) -> bool:
        start = time.time()
        pe_api_predictions = PE_API_PredictionsAI()

        MAX_SATS = {
            pe_const.E_CONSTELLATION.E_CONSTEL_GPS: pe_const.MAX_SATS_GPS,
            pe_const.E_CONSTELLATION.E_CONSTEL_GAL: pe_const.MAX_SATS_GAL,
            pe_const.E_CONSTELLATION.E_CONSTEL_BDS: pe_const.MAX_SATS_BDS,
        }

        for cons_idx in range(pe_const.NUM_CONSTELS):
            constel_type = cons_idx + 1
            max_sats = MAX_SATS.get(constel_type, 0)

            for sat_idx in range(max_sats):
                global_sat_idx = get_global_sat_idx(cons_idx, sat_idx)

                for freq_idx in range(pe_const.NUM_CHANNELS):
                    valid = predictions[global_sat_idx, freq_idx, 0]
                    prediction_value = predictions[global_sat_idx, freq_idx, 1]

                    if valid:
                        constel = cons_idx + 1
                        sat_id = sat_idx + 1

                        # Crear y llenar la instancia de PE_PredictionsAI
                        prediction = PE_PredictionsAI()
                        prediction.timestamp_week = epoch.week
                        prediction.timestamp_second = epoch.second
                        prediction.constel = constel
                        prediction.sat_id = sat_id
                        if (
                            constel == pe_const.E_CONSTELLATION.E_CONSTEL_GPS
                            and freq_idx == 0
                        ):
                            prediction.freq = const.SIGNAL_FREQ.SIGNALS[
                                self.configMgr_.config_info_.Signal_1_GPS
                            ]
                        elif (
                            constel == pe_const.E_CONSTELLATION.E_CONSTEL_GPS
                            and freq_idx == 1
                        ):
                            prediction.freq = const.SIGNAL_FREQ.SIGNALS[
                                self.configMgr_.config_info_.Signal_2_GPS
                            ]
                        elif (
                            constel == pe_const.E_CONSTELLATION.E_CONSTEL_GAL
                            and freq_idx == 0
                        ):
                            prediction.freq = const.SIGNAL_FREQ.SIGNALS[
                                self.configMgr_.config_info_.Signal_1_GAL
                            ]
                        elif (
                            constel == pe_const.E_CONSTELLATION.E_CONSTEL_GAL
                            and freq_idx == 1
                        ):
                            prediction.freq = const.SIGNAL_FREQ.SIGNALS[
                                self.configMgr_.config_info_.Signal_2_GAL
                            ]
                        elif (
                            constel == pe_const.E_CONSTELLATION.E_CONSTEL_BDS
                            and freq_idx == 0
                        ):
                            prediction.freq = const.SIGNAL_FREQ.SIGNALS[
                                self.configMgr_.config_info_.Signal_1_BDS
                            ]
                        elif (
                            constel == pe_const.E_CONSTELLATION.E_CONSTEL_BDS
                            and freq_idx == 1
                        ):
                            prediction.freq = const.SIGNAL_FREQ.SIGNALS[
                                self.configMgr_.config_info_.Signal_2_BDS
                            ]
                        prediction.prediction = prediction_value
                        prediction.usable = True

                        # Asignar la instancia al array de ctypes en la posición correcta
                        pe_api_predictions[global_sat_idx][freq_idx] = prediction

        self._times["load_predictionAI"].append(time.time() - start)
        return self.position_engine_.LoadPredictionsAI(pe_api_predictions)
