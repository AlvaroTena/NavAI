import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import pewrapper.types.constants as pe_const
import rlnav.types.constants as const
from navutils.logger import Logger
from pewrapper.managers import configuration_mgr, wrapper_data_mgr
from pewrapper.misc.version_wrapper_bin import RELEASE_INFO, about_msg
from pewrapper.types.gps_time_wrapper import GPS_Time
from rlnav.data.dataset import RLDataset
from rlnav.env.wrapper import RLWrapper
from rlnav.managers.reward_mgr import RewardManager

ELEV_THRES = 30
MIN_ELEV = 5


def get_transformers_min_max(transformers_path: str):

    min_values = {feature: np.inf for feature in const.PROCESSED_FEATURE_LIST}
    max_values = {feature: -np.inf for feature in const.PROCESSED_FEATURE_LIST}

    def process_feature(feature):
        file_path = os.path.join(transformers_path, f"transformed_{feature}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            return feature, df.min().min(), df.max().max()
        return feature, np.inf, -np.inf

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(process_feature, const.PROCESSED_FEATURE_LIST)

    for feature, fmin, fmax in results:
        min_values[feature] = min(min_values[feature], fmin)
        max_values[feature] = max(max_values[feature], fmax)

    return (
        [min_values[f] for f in const.PROCESSED_FEATURE_LIST],
        [max_values[f] for f in const.PROCESSED_FEATURE_LIST],
    )


class PE_Env(py_environment.PyEnvironment):
    def __init__(
        self,
        configMgr: configuration_mgr.ConfigurationManager,
        wrapper_data: wrapper_data_mgr.WrapperDataManager,
        rewardMgr: RewardManager,
        min_values: list,
        max_values: list,
        transformers_path: str,
        output_path: str,
        window_size=1,
        scenario: str = None,
        generation: int = None,
    ):
        self.configMgr = configMgr
        self.wrapper_data = wrapper_data
        self.rewardMgr = rewardMgr

        self.scenario = scenario
        self.gen = generation
        self.transformers_path = transformers_path
        self.output_path = output_path

        self.wrapper = None

        self._action_spec = tuple(
            array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=1, name=f"action_{i}"
            )
            for i in range(pe_const.MAX_SATS * pe_const.NUM_CHANNELS)
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(
                pe_const.MAX_SATS * pe_const.NUM_CHANNELS,
                1 + len(const.PROCESSED_FEATURE_LIST),
            ),
            dtype=np.float32,
            minimum=np.array([0.0] + min_values),
            maximum=np.array([1.0] + max_values),
            name="observation",
        )

        self.window_size = window_size

        self._state = np.zeros(self._observation_spec.shape, dtype=np.float32)
        self._episode_ended = False
        self.completed_dataset_once = False

        self.reset_times()

    def reset_times(self):
        self._times = {}
        self._times["reset_env"] = []
        self._times["step_env"] = []
        if hasattr(self, "dataset"):
            self.dataset.reset_times()
        if hasattr(self, "wrapper") and self.wrapper is not None:
            self.wrapper.reset_times()
        if hasattr(self, "rewardMgr"):
            self.rewardMgr.reset_times()

    def get_times(self):
        if hasattr(self, "dataset"):
            self._times.update(self.dataset.get_times())
        self._times.update(self.wrapper.get_times())
        self._times.update(self.rewardMgr.get_times())

        times = copy.deepcopy(self._times)
        self.reset_times()
        return times

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        start = time.time()

        os.makedirs(self.output_path, exist_ok=True)

        self.wrapper = RLWrapper(
            self.configMgr,
            self.wrapper_data,
            self.rewardMgr,
            use_AI=True,
            generation=self.gen,
        )

        self.dataset = RLDataset(
            config_signals=[
                [
                    self.configMgr.config_info_.Signal_1_GPS,
                    self.configMgr.config_info_.Signal_2_GPS,
                ],
                [
                    self.configMgr.config_info_.Signal_1_GAL,
                    self.configMgr.config_info_.Signal_2_GAL,
                ],
                [
                    self.configMgr.config_info_.Signal_1_BDS,
                    self.configMgr.config_info_.Signal_2_BDS,
                ],
            ],
            transformer_path=os.path.join(self.transformers_path, "transform_fn.pkl"),
            window_size=self.window_size,
        )

        self.prev_ai_epoch = GPS_Time()
        self._close_wrapper()

        self.wrapper.reset(
            self.configMgr,
            self.wrapper_data,
            self.rewardMgr,
            use_AI=True,
            generation=self.gen,
        )
        self._init_log()
        self.rewardMgr.set_output_path(self.output_path, self.scenario, self.gen)
        self.rewardMgr.reward_rec.initialize(
            self.wrapper.wrapper_file_data_.initial_epoch
        )

        if not self.wrapper._start_processing(
            self.output_path,
            self.commit_id,
            self.common_lib_commit_id,
        ):
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                "Error processing PE: ",
                use_AI=True,
            )
            self._close_wrapper()
            raise RuntimeError

        state = self._check_state()
        if isinstance(state, bool):
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                "Failed to reset RL environment",
            )
            raise Exception

        self._episode_ended = False
        self._times["reset_env"].append(time.time() - start)
        return ts.restart(self._state)

    def _step(self, action):
        start = time.time()
        if self._episode_ended:
            return self._reset()

        action = np.array(action, dtype=np.bool_)
        validity = self._state[:, 0]

        action = np.concatenate(
            (validity[..., np.newaxis], action[..., np.newaxis]), axis=-1
        )
        action = action.reshape(
            (
                pe_const.MAX_SATS,
                pe_const.NUM_CHANNELS,
                -1,
            )
        )

        if not (result := self._compute(action))[0]:
            return

        _, ai_output = result

        reward = self.rewardMgr.compute_reward(ai_output)

        epoch = GPS_Time(
            w=ai_output.output_PE.timestamp_week, s=ai_output.output_PE.timestamp_second
        )
        self.rewardMgr.reward_rec.record(epoch, reward)

        state = self._check_state()
        if isinstance(state, bool):
            if state:
                self._episode_ended = True
                self.completed_dataset_once = True

            else:
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.MAIN,
                    "Failed to process epochs RL environment",
                )
                raise Exception

        self._times["step_env"].append(time.time() - start)
        if self._episode_ended:
            return ts.termination(
                np.zeros(self._observation_spec.shape, dtype=np.float32), reward
            )

        else:
            return ts.transition(self._state, reward)

    def _check_state(self):
        state = self._process_epochs()

        if isinstance(state, pd.DataFrame):
            while isinstance(state, pd.DataFrame) and (
                state.empty
                or not (ref_match := self.rewardMgr.match_ref(self.prev_ai_epoch))
            ):
                if not state.empty and not ref_match:
                    self.dataset.ingest_data(state)

                self._compute()
                state = self._process_epochs()

            if isinstance(state, pd.DataFrame):
                state = self.dataset.process_data(state, self._observation_spec)

                self._state = state

        return state

    def _process_epochs(self):
        if not (result := self.wrapper.process_epoch())[0]:
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
            self.rewardMgr.reward_rec.close()
            return True

        elif ai_state == "action_needed":
            return self.wrapper.get_features_AI()

        else:
            return False

    def _compute(self, predictions: np.ndarray = None):
        if predictions is not None:
            if not self.wrapper.load_predictions_AI(self.prev_ai_epoch, predictions):
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.MAIN,
                    f"Error loading predictions",
                    use_AI=True,
                )
                return False, None

        if not (result := self.wrapper.compute())[0]:
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"Error processing PE: ",
                use_AI=True,
            )
            return False, None
        _, ai_output = result

        return True, ai_output

    def _init_log(self):
        self.commit_id = about_msg()

        if not (result := self.wrapper._get_common_lib_commit_id())[0]:
            raise ReferenceError("AI_Wrapper")
        _, common_lib_commit_id = result

        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"{RELEASE_INFO}, Commit ID PE_Wrapper: {self.commit_id}, {common_lib_commit_id}, started",
            use_AI=True,
        )

        self.common_lib_commit_id = common_lib_commit_id.split(" ")[-1]

    def _close_wrapper(self):
        if not self.wrapper.close_PE():
            Logger.log_message(
                Logger.Category.ERROR,
                Logger.Module.MAIN,
                f"Error closing files of PE: ",
                use_AI=True,
            )

    def get_logging_data(self):
        logging_data = {}
        logging_data.update(self.rewardMgr.get_log_data())

        return logging_data

    def get_ai_positions(self):
        return self.rewardMgr.get_ai_positions()

    def update_map(self):
        return self.rewardMgr.update_map()

    def get_reward_filepath(self):
        return self.rewardMgr.get_reward_filepath()

    def is_done(self):
        return self.completed_dataset_once
