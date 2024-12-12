import copy
import os
import time

import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import pewrapper.types.constants as pe_const
import rlnav.types.constants as const
from navutils.logger import Logger
from pewrapper.types import GPS_Time
from rlnav.data.dataset import RLDataset
from rlnav.managers.reward_mgr import RewardManager
from rlnav.managers.wrapper_mgr import WrapperManager

ELEV_THRES = 30
MIN_ELEV = 5


def get_transformers_min_max(transformers_path: str):
    min_values = {feature: np.inf for feature in const.PROCESSED_FEATURE_LIST}
    max_values = {feature: -np.inf for feature in const.PROCESSED_FEATURE_LIST}

    for feature in const.PROCESSED_FEATURE_LIST:
        file_path = os.path.join(transformers_path, f"transformed_{feature}.parquet")

        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)

            feature_min = df.min().min()
            feature_max = df.max().max()

            min_values[feature] = min(min_values[feature], feature_min)
            max_values[feature] = max(max_values[feature], feature_max)

    return (
        [min_values[feature] for feature in const.PROCESSED_FEATURE_LIST],
        [max_values[feature] for feature in const.PROCESSED_FEATURE_LIST],
    )


class PE_Env(py_environment.PyEnvironment):
    def __init__(
        self,
        wrapperMgr: WrapperManager,
        rewardMgr: RewardManager,
        transformers_path: str,
        window_size=1,
    ):
        self.wrapperMgr = wrapperMgr
        self.rewardMgr = rewardMgr

        self.transformers_path = transformers_path

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(pe_const.MAX_SATS * pe_const.NUM_CHANNELS,),
            dtype=np.float32,
            minimum=[0.5],
            maximum=[15.0],
            name="action",
        )
        min_values, max_values = get_transformers_min_max(
            os.path.join(transformers_path, "transforms")
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

        self.reset_times()

    def reset_times(self):
        self._times = {}
        self._times["reset_env"] = []
        self._times["step_env"] = []

    def get_times(self):
        times = copy.deepcopy(self._times)
        self.reset_times()
        if hasattr(self, "dataset"):
            times.update(self.dataset.get_times())
        return times

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        start = time.time()
        self.dataset = RLDataset(
            config_signals=[
                [
                    self.wrapperMgr.configMgr.config_info_.Signal_1_GPS,
                    self.wrapperMgr.configMgr.config_info_.Signal_2_GPS,
                ],
                [
                    self.wrapperMgr.configMgr.config_info_.Signal_1_GAL,
                    self.wrapperMgr.configMgr.config_info_.Signal_2_GAL,
                ],
                [
                    self.wrapperMgr.configMgr.config_info_.Signal_1_BDS,
                    self.wrapperMgr.configMgr.config_info_.Signal_2_BDS,
                ],
            ],
            transformer_path=os.path.join(self.transformers_path, "transform_fn.pkl"),
            window_size=self.window_size,
        )

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

        invalid_mask = np.logical_or(
            action < self._action_spec.minimum, action > self._action_spec.maximum
        )

        validity = self._state[:, 0]
        validity = np.where(invalid_mask.squeeze(), 0, validity)

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

        if not (result := self.wrapperMgr.compute(action))[0]:
            return

        _, ai_output = result

        reward = self.rewardMgr.compute_reward(ai_output)

        epoch = GPS_Time(
            w=ai_output.output_PE.timestamp_week, s=ai_output.output_PE.timestamp_second
        )
        self.wrapperMgr.reward_rec.record(epoch, reward)

        state = self._check_state()
        if isinstance(state, bool):
            if state:
                self._episode_ended = True

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
        state = self.wrapperMgr.process_epochs()

        if isinstance(state, pd.DataFrame):
            while isinstance(state, pd.DataFrame) and (
                state.empty
                or not (
                    ref_match := self.rewardMgr.match_ref(self.wrapperMgr.prev_ai_epoch)
                )
            ):
                if not state.empty and not ref_match:
                    self.dataset.ingest_data(state)

                self.wrapperMgr.compute()
                state = self.wrapperMgr.process_epochs()

            if isinstance(state, pd.DataFrame):
                state = self.dataset.process_data(state, self._observation_spec)

                self._state = state

        return state
