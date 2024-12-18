import copy
import ctypes as ct
import os
import shutil
import time

import numpy as np
import pewrapper.misc.parser_utils as parser_utils
from navutils.logger import Logger
from neptune import Run
from pewrapper.api.common_api_types import Log_Handle, LogCategoryPE
from pewrapper.managers import configuration_mgr, wrapper_data_mgr
from pewrapper.misc.version_wrapper_bin import RELEASE_INFO, about_msg
from pewrapper.types.gps_time_wrapper import GPS_Time
from rlnav.env.wrapper import RLWrapper
from rlnav.managers import reward_mgr
from rlnav.recorder.reward_recorder import RewardRecorder
from rlnav.types.reference_types import ReferenceMode, ReferenceType


@Log_Handle
def RL_LogWrapper(
    category: ct.c_uint32,
    eventParticulars: ct.c_char_p,
    fileName: ct.c_char_p,
    function: ct.c_char_p,
    codeLine: ct.c_uint16,
    use_AI: ct.c_bool,
):
    level = {
        LogCategoryPE.NONE.value: Logger.Category.NOTSET,
        LogCategoryPE.TRACE.value: Logger.Category.TRACE,
        LogCategoryPE.DEBUG.value: Logger.Category.DEBUG,
        LogCategoryPE.INFO.value: Logger.Category.INFO,
        LogCategoryPE.WARNING.value: Logger.Category.WARNING,
        LogCategoryPE.error.value: Logger.Category.ERROR,
    }.get(category, Logger.Category.ERROR)
    message = eventParticulars.decode("utf-8")
    if level >= Logger.Category.ERROR:
        Logger.log_message(level, Logger.Module.PE, f"{message}", use_AI=use_AI)


class WrapperManager:
    def __init__(
        self,
        scenarios_path,
        n_scenarios,
        priority_scen,
        output_path,
        npt_run: Run,
        rewardMgr: reward_mgr.RewardManager,
        num_generations=5,
    ):
        if os.path.exists(scenarios_path):
            self.scenarios_path = scenarios_path
            all_scenarios = sorted(
                [
                    dir
                    for dir in os.listdir(self.scenarios_path)
                    if dir.startswith("scenario_") and not dir.endswith(".dvc")
                ]
            )

            if priority_scen in all_scenarios:
                all_scenarios.remove(priority_scen)
                all_scenarios.insert(0, priority_scen)
            else:
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.NONE,
                    f"Priority scenario '{priority_scen}' not found. Maintaining normal order.",
                )

            first_scen, last_scen = n_scenarios
            if last_scen != -1:
                self.scenarios = all_scenarios[first_scen:last_scen]

            else:
                self.scenarios = all_scenarios[first_scen:]

        else:
            log_msg = (
                f"Error getting scenarios from non existing directory {scenarios_path}"
            )
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.CONFIG,
                log_msg,
            )
            raise FileNotFoundError(log_msg)

        self.output_path = output_path

        self._log_npt = npt_run
        self.reset_times()

        self.configMgr = configuration_mgr.ConfigurationManager(
            output_path,
            "",
            log_handle=RL_LogWrapper,
        )
        self.wrapper_data = wrapper_data_mgr.WrapperDataManager(
            GPS_Time(),
            GPS_Time(),
            self.configMgr,
        )
        self.rewardMgr = rewardMgr

        self.ai_wrapper = RLWrapper(
            self.configMgr,
            self.wrapper_data,
            self.rewardMgr,
            use_AI=True,
        )
        self.pe_wrapper = RLWrapper(
            self.configMgr,
            self.wrapper_data,
            self.rewardMgr,
            use_AI=False,
        )
        self.prev_ai_epoch = GPS_Time()

        self.reward_rec = RewardRecorder("")

        self.scenario_it = iter(self.scenarios)
        self.scenario = next(self.scenario_it)

        self.num_generations = num_generations
        self.scenario_generation = {scenario: -1 for scenario in self.scenarios}

    def reset_times(self):
        self._times = {}
        self._times["next_scenario"] = []
        self._times["restart_scenario"] = []

    def get_times(self):
        times = copy.deepcopy(self._times)
        self.reset_times()
        times.update({"AI": self.ai_wrapper.get_times()})
        times.update({"noAI": self.pe_wrapper.get_times()})
        return times

    def next_scenario(self, parsing_rate=0):
        start = time.time()
        logs_npt = {}
        addInfo = ""

        self.close_pe_wrapper()
        self.pe_wrapper.reset(self.configMgr, self.wrapper_data, self.rewardMgr, False)

        if (
            first_it := self.scenario_generation[self.scenario] == -1
        ) or self.scenario_generation[self.scenario] >= self.num_generations:
            time_dst = "next_scenario"
            if first_it:
                self.scenario_generation[self.scenario] = 1
                logs_npt["generation"] = self.scenario_generation[self.scenario]

            else:
                try:
                    self.scenario = next(self.scenario_it)
                    self.scenario_generation[self.scenario] = 1
                    logs_npt["generation"] = self.scenario_generation[self.scenario]
                except StopIteration:
                    Logger.log_message(
                        Logger.Category.WARNING,
                        Logger.Module.MAIN,
                        f"No more scenarios in {self.scenarios_path}",
                    )
                    return False

            logs_npt["scenario"] = self.scenario

            scenario_path = os.path.join(self.scenarios_path, self.scenario)

            config_path = os.path.join(scenario_path, "CONFIG")
            session_path = os.path.join(config_path, "session.ini")
            scenario_info_path = os.path.join(scenario_path, "scenario.ini")
            reference_file_path = os.path.join(
                scenario_path, "INPUTS/REF/kinematic_reference.parquet"
            )

            if (dir_error := not os.path.isdir(config_path)) or not os.path.isfile(
                session_path
            ):
                log_msg = f"Session file ({session_path}) not exists or CONFIG directory not exists in {scenario_path}"
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.CONFIG,
                    log_msg,
                )
                if dir_error:
                    raise NotADirectoryError(log_msg)
                else:
                    raise FileNotFoundError(log_msg)

            if not (result := parser_utils.parse_session_file(session_path))[0]:
                _, _, _, _, addInfo_session, _, _ = result
                log_msg = f"Error processing session file: {addInfo_session}"
                Logger.log_message(
                    Logger.Category.DEBUG,
                    Logger.Module.READER,
                    log_msg,
                )
                raise ValueError(log_msg)

            (
                _,
                config_file_path,
                wrapper_file_path,
                tracing_config_file,
                _,
                initial_epoch_session,
                final_epoch_session,
            ) = result

            if not (result := parser_utils.parse_scenario_file(scenario_info_path))[0]:
                Logger.log_message(
                    Logger.Category.DEBUG,
                    Logger.Module.READER,
                    f"Error reading scenario info, default reference mode and type used",
                )
                ref_mode = ReferenceMode.KINEMATIC
                ref_type = ReferenceType.Kinematic.SPAN_FILE

            else:
                (
                    _,
                    ref_mode,
                    ref_type,
                ) = result

            output_path = os.path.join(self.output_path, self.scenario)
            os.makedirs(output_path, exist_ok=True)

            self.configMgr.reset(output_path, tracing_config_file)
            self.wrapper_data.reset(initial_epoch_session, final_epoch_session)

            self.rewardMgr.load_reference(reference_file_path, ref_mode, ref_type)
            self.rewardMgr.set_output_path(
                self.reward_rec,
                output_path,
                self.scenario,
                self.scenario_generation[self.scenario],
            )

            self._copy_files_from_scenario(
                session_path,
                config_file_path,
                tracing_config_file,
                output_path,
            )

            ################################################
            #################  PROCESSING  #################
            ################################################
            if not (result := self.configMgr.parse_config_file(config_file_path))[0]:
                _, addInfo = result
                log_msg = f"Error processing config file: {addInfo}"

                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.MAIN,
                    log_msg,
                )

                self.close_PE()
                raise ValueError(log_msg)

            if not (
                result := self.wrapper_data.parse_wrapper_file(
                    wrapper_file_path, parsing_rate
                )
            )[0]:
                _, addInfo = result
                log_msg = f"Error processing wrapper file: {addInfo}"

                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.MAIN,
                    log_msg,
                )

                self.close_PE()
                raise ValueError(log_msg)

            _, addInfo = result

            os.makedirs(
                (
                    output_noai := self.configMgr.get_config(
                        use_ai_multipath=False
                    ).log_path.decode("utf-8")
                ),
                exist_ok=True,
            )

            baseline_file_path = os.path.join(
                scenario_path, "INPUTS/AI/PE_Baseline.parquet"
            )

            if not os.path.isfile(baseline_file_path):
                if not self.pe_wrapper.process_scenario(None, None, output_noai, None):
                    Logger.log_message(
                        Logger.Category.ERROR,
                        Logger.Module.MAIN,
                        f"Error processing scenario",
                    )
                    return False

                self.rewardMgr.save_baseline(baseline_file_path)

            else:
                self.rewardMgr.load_baseline(baseline_file_path)

        else:
            time_dst = "restart_scenario"
            self.scenario_generation[self.scenario] += 1
            logs_npt["generation"] = self.scenario_generation[self.scenario]
            output_path = os.path.join(self.output_path, self.scenario)
            self.rewardMgr.next_generation(
                self.reward_rec,
                output_path,
                self.scenario,
                self.scenario_generation[self.scenario],
            )

        self._init_wrappers(addInfo)
        self._log_npt["training"] = logs_npt
        self._times[f"{time_dst}"].append(time.time() - start)

        return True

    def _copy_files_from_scenario(
        self,
        session_pe_file_path,
        wrapper_config_file_path,
        wrapper_tracing_file_path,
        output_path,
    ):
        # COPY INPUT FILES TO OUTPUT FOLDER
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        # Session Config file
        session_file_destination = os.path.join(output_path, "session.ini")
        if os.path.isfile(session_pe_file_path):
            shutil.copyfile(session_pe_file_path, session_file_destination)
        else:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.MAIN,
                f"Session file path: {session_pe_file_path} does not exist. Copy to output folder aborted",
            )

        # Wrapper Config file
        wrapper_config_file_destination = os.path.join(
            output_path, "wrapper_config.ini"
        )
        if os.path.isfile(wrapper_config_file_path):
            shutil.copyfile(wrapper_config_file_path, wrapper_config_file_destination)
        else:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.MAIN,
                f"Session file path: {wrapper_config_file_path} does not exist. Copy to output folder aborted",
            )

        # Tracing config file
        wrapper_tracing_file_destination = os.path.join(
            output_path, "tracing_config.ini"
        )
        if os.path.isfile(wrapper_tracing_file_path):
            shutil.copyfile(wrapper_tracing_file_path, wrapper_tracing_file_destination)
        else:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.MAIN,
                f"Session file path: {wrapper_tracing_file_path} does not exist. Copy to output folder aborted",
            )

    def close_PE(self):
        self.close_ai_wrapper()
        self.close_pe_wrapper()

    def close_ai_wrapper(self):
        if not self.ai_wrapper.close_PE():
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"Error closing files of PE: ",
                use_AI=True,
            )

    def close_pe_wrapper(self):
        if not self.pe_wrapper.close_PE():
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"Error closing files of PE: ",
                use_AI=True,
            )

    def _init_log(self):
        self.ai_pe_wrapper_commit_id = about_msg()

        if not (result := self.ai_wrapper._get_common_lib_commit_id())[0]:
            raise ReferenceError("AI_Wrapper")
        _, ai_common_lib_commit_id = result

        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"{RELEASE_INFO}, Commit ID PE_Wrapper: {self.ai_pe_wrapper_commit_id}, {ai_common_lib_commit_id}, started",
            use_AI=True,
        )

        self.common_lib_commit_id = ai_common_lib_commit_id.split(" ")[-1]

    def _init_wrappers(self, addInfo):
        os.makedirs(
            (
                output_ai := self.configMgr.get_config(
                    use_ai_multipath=True,
                    generation=self.scenario_generation[self.scenario],
                ).log_path.decode("utf-8")
            ),
            exist_ok=True,
        )
        self.prev_ai_epoch = GPS_Time()
        self.close_ai_wrapper()
        self.ai_wrapper.reset(
            self.configMgr,
            self.wrapper_data,
            self.rewardMgr,
            True,
            (
                self.scenario_generation[self.scenario]
                if self.num_generations > 1
                else None
            ),
        )
        self._init_log()
        self.reward_rec.initialize(self.wrapper_data.initial_epoch)
        if not self.ai_wrapper._start_processing(
            output_ai,
            self.ai_pe_wrapper_commit_id,
            self.common_lib_commit_id,
        ):
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"Error processing PE: {addInfo}",
                use_AI=True,
            )
            self.close_PE()
            raise RuntimeError

    def process_epochs(self):
        if not (result := self.ai_wrapper.process_epoch())[0]:
            _, ai_state, _ = result
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"Error processing PE: {ai_state}",
                use_AI=True,
            )
            return False

        _, ai_state, ai_pvt = result
        ai_epoch = GPS_Time(w=ai_pvt.timestamp_week, s=ai_pvt.timestamp_second)

        if self.prev_ai_epoch > ai_epoch:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"AI_Wrapper processed old epoch: {ai_epoch.calendar_column_str_d()} | Prev epoch: {self.prev_ai_epoch.calendar_column_str_d()}",
                use_AI=True,
            )

        self.prev_ai_epoch = ai_epoch

        if ai_state == "finished_wrapper":
            self.reward_rec.close()
            return True

        elif ai_state == "action_needed":
            return self.ai_wrapper.get_features_AI()

        else:
            return False

    def compute(self, predictions: np.ndarray = None):
        if predictions is not None:
            if not self.ai_wrapper.load_predictions_AI(self.prev_ai_epoch, predictions):
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.MAIN,
                    f"Error loading predictions",
                    use_AI=True,
                )
                return False, None

        if not (result := self.ai_wrapper.compute())[0]:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"Error processing PE: ",
                use_AI=True,
            )
            return False, None
        _, ai_output = result

        return True, ai_output
