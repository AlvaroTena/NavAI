import copy
import ctypes as ct
import os
import shutil
import time

import pewrapper.misc.parser_utils as parser_utils
from navutils.logger import Logger
from neptune import Run
from pewrapper.api.common_api_types import Log_Handle, LogCategoryPE
from pewrapper.managers import configuration_mgr, wrapper_data_mgr
from pewrapper.types.gps_time_wrapper import GPS_Time
from rlnav.env.wrapper import RLWrapper
from rlnav.managers import reward_mgr
from rlnav.types.reference_types import ReferenceMode, ReferenceType
from rlnav.utils.common import get_parent_scenario_name, is_scenario_subset


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
        Logger.log_message(level, Logger.Module.PE, f"{message}")


class WrapperManager:
    def __init__(
        self,
        scenarios_path,
        n_scenarios,
        priority_scen,
        scenarios_subset,
        output_path,
        rewardMgr: reward_mgr.RewardManager,
        npt_run: Run,
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

            for scen in reversed(priority_scen):
                if scen in all_scenarios:
                    all_scenarios.remove(scen)
                    all_scenarios.insert(0, scen)
                else:
                    Logger.log_message(
                        Logger.Category.WARNING,
                        Logger.Module.CONFIG,
                        f"Priority scenario '{priority_scen}' not found. Maintaining normal order.",
                    )

            first_scen, last_scen = n_scenarios
            if last_scen != -1:
                self.scenarios = all_scenarios[first_scen:last_scen]

            else:
                self.scenarios = all_scenarios[first_scen:]

            if scenarios_subset:
                updated_scenarios = []
                for scenario in self.scenarios:
                    if any([scenario in subset for subset in scenarios_subset]):
                        updated_scenarios.extend(
                            [
                                subset
                                for subset in scenarios_subset
                                if subset.startswith(scenario)
                            ]
                        )
                    else:
                        updated_scenarios.append(scenario)
                self.scenarios = updated_scenarios

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

        self.base_output_path = output_path
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

        self.pe_wrapper = RLWrapper(
            self.configMgr,
            self.wrapper_data,
            self.rewardMgr,
            use_AI=False,
        )

        self.scenario_it = iter(self.scenarios)
        self.scenario = None

    def reset_times(self):
        self._times = {}
        self._times["next_scenario"] = []
        self._times["restart_scenario"] = []

    def get_times(self):
        times = copy.deepcopy(self._times)
        self.reset_times()
        times.update({"noAI": self.pe_wrapper.get_times()})
        return times

    def next_scenario(self, parsing_rate=0):
        start = time.time()

        self.close_pe_wrapper()
        self.pe_wrapper.reset(self.configMgr, self.wrapper_data, self.rewardMgr, False)

        prev_scenario = ""

        try:
            self.scenario = next(self.scenario_it)
        except StopIteration:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.MAIN,
                f"No more scenarios in {self.scenarios_path}",
            )
            return False

        if (
            prev_scenario != ""
            and (
                is_scenario_subset(self.scenario) and is_scenario_subset(prev_scenario)
            )
            and (
                (parent_scenario_name := get_parent_scenario_name(self.scenario))
                == get_parent_scenario_name(prev_scenario)
            )
        ):
            if parent_scenario_name is not None:
                self._init_subscenario(parent_scenario_name)

            else:
                raise ValueError("Error getting parent scenario name")

        else:
            prev_scenario = self.scenario
            scenario_path = os.path.join(self.scenarios_path, self.scenario)
            parent_scenario_name = ""
            if not os.path.isdir(scenario_path) and is_scenario_subset(self.scenario):
                scenario_path = os.path.join(
                    self.scenarios_path,
                    (parent_scenario_name := get_parent_scenario_name(self.scenario)),
                )
                subset = True

            else:
                subset = False

            self._init_scenario(scenario_path, parsing_rate)
            if subset:
                self._init_subscenario(parent_scenario_name)

        self.rewardMgr.set_output_path(os.path.normpath(self.output_path))

        self._log_npt["training/scenario"] = self.scenario
        self._times["next_scenario"].append(time.time() - start)

        return True

    def _init_scenario(self, scenario_path, parsing_rate):
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

        self.output_path = os.path.join(self.base_output_path, self.scenario)

        self.configMgr.reset(self.output_path, tracing_config_file)
        self.wrapper_data.reset(initial_epoch_session, final_epoch_session)

        self.rewardMgr.load_reference(reference_file_path, ref_mode, ref_type)

        self._copy_files_from_scenario(
            session_path,
            config_file_path,
            tracing_config_file,
            self.output_path,
        )

        ################################################
        #################  PROCESSING  #################
        ################################################
        if not (result := self.configMgr.parse_config_file(config_file_path))[0]:
            _, addInfo = result
            log_msg = f"Error processing config file: {addInfo}"

            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.READER,
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
                Logger.Module.READER,
                log_msg,
            )

            self.close_PE()
            raise ValueError(log_msg)

        _, addInfo = result

        baseline_file_path = os.path.join(
            scenario_path, "INPUTS/AI/PE_Baseline.parquet"
        )

        if not os.path.isfile(baseline_file_path):
            os.makedirs(
                (
                    output_noai := self.configMgr.get_config(
                        use_ai_multipath=False
                    ).log_path.decode("utf-8")
                ),
                exist_ok=True,
            )
            if not (
                result := self.pe_wrapper.process_scenario(
                    None, None, output_noai, None
                )
            )[0]:
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.MAIN,
                    f"Error processing scenario",
                )
            _, pe_errors = result
            if pe_errors is not None:
                self._log_baseline_errors(pe_errors)

            self.rewardMgr.save_baseline(baseline_file_path)

        else:
            pe_errors = self.rewardMgr.load_baseline(baseline_file_path)
            self._log_baseline_errors(pe_errors)

    def _init_subscenario(self, parent_scenario_name):
        scenario_path = os.path.join(self.scenarios_path, parent_scenario_name)
        config_path = os.path.join(scenario_path, "CONFIG")
        session_path = os.path.join(config_path, "session.ini")
        subsession_path = os.path.join(config_path, "subsessions.ini")

        if (dir_error := not os.path.isdir(config_path)) or not os.path.isfile(
            subsession_path
        ):
            log_msg = f"Subsession file ({subsession_path}) not exists or CONFIG directory not exists in {scenario_path}"
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.CONFIG,
                log_msg,
            )
            if dir_error:
                raise NotADirectoryError(log_msg)
            else:
                raise FileNotFoundError(log_msg)

        if not (result := parser_utils.parse_session_file(session_path, verbose=False))[
            0
        ]:
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
            _,
            tracing_config_file,
            _,
            initial_epoch_session,
            final_epoch_session,
        ) = result

        subsession_epochs = parser_utils.parse_subsessions_file(
            subsession_path, self.scenario
        )
        if subsession_epochs:
            initial_epoch = subsession_epochs["InitialEpoch"]
            final_epoch = subsession_epochs["FinalEpoch"]
        else:
            initial_epoch = initial_epoch_session
            final_epoch = final_epoch_session

        self.configMgr.reset_log_path(self.output_path)
        self.wrapper_data.reset_epochs(initial_epoch, final_epoch)

        self.rewardMgr.limit_epochs(initial_epoch, final_epoch)
        pe_errors = self.rewardMgr.limit_baseline_log(initial_epoch, final_epoch)
        self._log_baseline_errors(pe_errors)

        self._copy_files_from_scenario(
            session_path,
            config_file_path,
            tracing_config_file,
            self.output_path,
            subsession_epochs,
        )

    def _copy_files_from_scenario(
        self,
        session_pe_file_path,
        wrapper_config_file_path,
        wrapper_tracing_file_path,
        output_path,
        change_session_epochs=None,
    ):
        # COPY INPUT FILES TO OUTPUT FOLDER
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        # Session Config file
        session_file_destination = os.path.join(output_path, "session.ini")
        if os.path.isfile(session_pe_file_path):
            if change_session_epochs is None or not change_session_epochs:
                shutil.copyfile(session_pe_file_path, session_file_destination)
            else:
                parser_utils.write_session_file(
                    session_pe_file_path,
                    session_file_destination,
                    change_session_epochs,
                )

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
        self.close_pe_wrapper()

    def close_pe_wrapper(self):
        if not self.pe_wrapper.close_PE():
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"Error closing files of PE: ",
            )

    def _log_baseline_errors(self, pe_errors):
        if self._log_npt.exists(f"training/train/PE_Errors"):
            del self._log_npt[f"training/train/PE_Errors"]

        if self._log_npt is not None and self._log_npt._mode != "debug":
            if all([v is not None and not v.empty for v in pe_errors.values()]):
                self._log_npt[f"training/{self.scenario}/PE"].extend(
                    {k: v.to_list() for k, v in pe_errors.items() if k != "Epoch"}
                )
                self._log_npt[f"training/train/PE_Errors"].extend(
                    {k: v.to_list() for k, v in pe_errors.items() if k != "Epoch"}
                )
