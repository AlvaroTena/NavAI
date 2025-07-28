import copy
import os
import time
from typing import Union

import folium
import numpy as np
import pandas as pd
import pyproj
import rlnav.types.constants as const
from geopy.point import Point
from navutils.logger import Logger
from pewrapper.api.pe_api_types import Latitude_Direction, Longitude_Direction
from pewrapper.managers import OutputStr
from pewrapper.types.gps_time_wrapper import CALENDAR_COLUMN_D_FORMAT, GPS_Time
from rlnav.managers.reference_datamanager import ReferenceDataManager, ReferenceMode
from rlnav.recorder.reward_recorder import RewardRecorder
from rlnav.types.reference_types import ReferenceMode, ReferenceType
from rlnav.types.running_metric import RunningDiffMetric

gps_start_date = pd.to_datetime(
    "1980 1 6 0 0 0.000000", format=CALENDAR_COLUMN_D_FORMAT
)


transformer_model = pyproj.Transformer.from_crs(
    {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
    {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
    always_xy=True,
)


def car2geo(x, y, z):
    lon, lat, hei = transformer_model.transform(x, y, z, radians=False)
    return lat, lon, hei


class RewardManager:
    def __init__(self):
        self.refMgr = ReferenceDataManager(
            ReferenceMode.KINEMATIC, ReferenceType.Kinematic.SPAN_FILE
        )
        self.reference_data = pd.DataFrame()
        self.base_data = pd.DataFrame()
        self.ai_data = pd.DataFrame()
        self.map_initialized = False
        self.ref_positions = pd.DataFrame(columns=["Latitude", "Longitude"])
        self.base_positions = pd.DataFrame(columns=["LAT_PROP", "LON_PROP"])
        self.ai_positions = pd.DataFrame(columns=["LAT_PROP", "LON_PROP"])
        self.map = None

        self.output_path = ""
        self.map_file = ""
        self.reset_times()

        self.reward = RunningDiffMetric(decay=0.9, shape=(3,))

        self.ai_errors_values = RunningDiffMetric(window_size=-1, shape=(3,))
        self.base_errors_values = RunningDiffMetric(window_size=-1, shape=(3,))
        self.rmse_patience = 0

        self.reward_rec = RewardRecorder("")

        self.log_data = {
            "ai_errors": [],
            "instant_rewards": [],
            "running_rewards": [],
            "cumulative_rewards": [],
        }

    def next_generation(self):
        self.ai_data = pd.DataFrame()
        self.map_initialized = False
        self.ai_positions = pd.DataFrame(columns=["LAT_PROP", "LON_PROP"])
        self.map = None

        self.reward.reset()

        self.ai_errors_values.reset()
        self.base_errors_values.reset()
        self.rmse_patience = 0

    def reset_times(self):
        self._times = {"compute_reward": []}

    def get_times(self):
        times = copy.deepcopy(self._times)
        self.reset_times()
        return times

    def set_output_path(self, output_path):
        os.makedirs(
            output_path,
            exist_ok=True,
        )
        self.output_path = output_path
        self.map_file = os.path.join(output_path, "map.html")

        self.reward_rec.reset(output_path)

    def limit_epochs(
        self,
        initial_epoch: Union[pd.Timestamp, GPS_Time],
        final_epoch: Union[pd.Timestamp, GPS_Time],
    ):
        if isinstance(initial_epoch, GPS_Time):
            initial_epoch = pd.to_datetime(
                initial_epoch.calendar_column_str_d(),
                format=CALENDAR_COLUMN_D_FORMAT,
            )
        if isinstance(final_epoch, GPS_Time):
            final_epoch = pd.to_datetime(
                final_epoch.calendar_column_str_d(),
                format=CALENDAR_COLUMN_D_FORMAT,
            )

        self.initial_epoch = initial_epoch
        self.final_epoch = final_epoch

    def load_reference(self, filename, ref_mode, ref_type):
        self.__init__()
        self.refMgr.reset(ref_mode, ref_type)
        self.refMgr.parse_ref_data(filename)

        self.reference_data = self.refMgr.reference

        ref_positions = self.reference_data[["Epoch", "Latitude", "Longitude"]]
        ref_positions.set_index("Epoch", inplace=True)
        self.ref_positions = ref_positions

    def load_baseline(self, filename):
        self.base_data = pd.read_parquet(filename)
        base_data = self.base_data[["RawEpoch", "LAT_PROP", "LON_PROP"]]
        base_data.set_index("RawEpoch", inplace=True)
        self.base_positions = base_data

        return self.log_baseline(self.base_data)

    def limit_baseline_log(
        self,
        initial_epoch: Union[pd.Timestamp, GPS_Time],
        final_epoch: Union[pd.Timestamp, GPS_Time],
    ):
        if isinstance(initial_epoch, GPS_Time):
            initial_epoch = pd.to_datetime(
                initial_epoch.calendar_column_str_d(),
                format=CALENDAR_COLUMN_D_FORMAT,
            )
        if isinstance(final_epoch, GPS_Time):
            final_epoch = pd.to_datetime(
                final_epoch.calendar_column_str_d(),
                format=CALENDAR_COLUMN_D_FORMAT,
            )

        baseline = self.base_data[
            self.base_data["RawEpoch"].between(initial_epoch, final_epoch)
        ]
        return self.log_baseline(baseline)

    def log_baseline(self, baseline):
        pe_errors = {
            "Epoch": baseline["RawEpoch"],
            "NorthError": baseline["NorthErrorProp"],
            "EastError": baseline["EastErrorProp"],
            "UpError": baseline["UpErrorProp"],
            "HorizontalError": baseline["HorizontalErrorProp"],
            "VerticalError": baseline["VerticalErrorProp"],
        }
        return pe_errors

    def save_baseline(self, filename):
        self.base_data.to_parquet(filename)

    def update_base(self, pe_output: OutputStr):
        pe_base = self._get_record_output(pe_output)
        self.base_data = pd.concat([self.base_data, pe_base], axis=0, ignore_index=True)
        self.base_data.sort_values(by=["RawEpoch"], inplace=True)

    def update_agent(self, ai_output: OutputStr):
        pe_ai = self._get_record_output(ai_output)
        self.ai_data = pd.concat([self.ai_data, pe_ai], axis=0, ignore_index=True)
        self.ai_data.sort_values(by=["RawEpoch"], inplace=True)

    def calculate_propagated_base_positions(self):
        pe_ref_df = self._match_and_extract_common_epochs(self.base_data)

        if pe_ref_df is not None:
            self.base_data = self._get_error_position_ref_span(pe_ref_df)
            base_positions = self.base_data[["RawEpoch", "LAT_PROP", "LON_PROP"]]
            base_positions.set_index("RawEpoch", inplace=True)
            self.base_positions = pd.concat([self.base_positions, base_positions])

            pe_errors = {
                "Epoch": self.base_data["RawEpoch"],
                "NorthError": self.base_data["NorthErrorProp"],
                "EastError": self.base_data["EastErrorProp"],
                "UpError": self.base_data["UpErrorProp"],
                "HorizontalError": self.base_data["HorizontalErrorProp"],
                "VerticalError": self.base_data["VerticalErrorProp"],
            }
            return pe_errors
        return None

    def compute_reward(self, ai_output: OutputStr):
        start = time.time()
        epoch = GPS_Time(
            w=ai_output.output_PE.timestamp_week, s=ai_output.output_PE.timestamp_second
        )

        try:
            ai_record = self._get_record_output(ai_output)
            ai_ref_df = self._match_and_extract_common_epochs(ai_record)
            pe_ref_df = self.base_data.loc[
                self.base_data["RawEpoch"].isin(ai_record["RawEpoch"])
            ]

            if ai_ref_df is not None and not pe_ref_df.empty:
                ai_ref_df = self._get_error_position_ref_span(ai_ref_df)
                ai_positions = ai_ref_df[["RawEpoch", "LAT_PROP", "LON_PROP"]]
                ai_positions.set_index("RawEpoch", inplace=True)
                self.ai_positions = pd.concat([self.ai_positions, ai_positions])

                # For base errors
                if isinstance(pe_ref_df["NorthErrorProp"], (pd.Series, list)):
                    north_error = (
                        pe_ref_df["NorthErrorProp"].iloc[-1]
                        if isinstance(pe_ref_df["NorthErrorProp"], pd.Series)
                        else pe_ref_df["NorthErrorProp"][-1]
                    )
                    east_error = (
                        pe_ref_df["EastErrorProp"].iloc[-1]
                        if isinstance(pe_ref_df["EastErrorProp"], pd.Series)
                        else pe_ref_df["EastErrorProp"][-1]
                    )
                    up_error = (
                        pe_ref_df["UpErrorProp"].iloc[-1]
                        if isinstance(pe_ref_df["UpErrorProp"], pd.Series)
                        else pe_ref_df["UpErrorProp"][-1]
                    )
                else:
                    north_error = pe_ref_df["NorthErrorProp"]
                    east_error = pe_ref_df["EastErrorProp"]
                    up_error = pe_ref_df["UpErrorProp"]

                base_errors = np.fabs(np.array([north_error, east_error, up_error]))

                # For AI errors
                if isinstance(ai_ref_df["NorthErrorProp"], (pd.Series, list)):
                    ai_north_error = (
                        ai_ref_df["NorthErrorProp"].iloc[-1]
                        if isinstance(ai_ref_df["NorthErrorProp"], pd.Series)
                        else ai_ref_df["NorthErrorProp"][-1]
                    )
                    ai_east_error = (
                        ai_ref_df["EastErrorProp"].iloc[-1]
                        if isinstance(ai_ref_df["EastErrorProp"], pd.Series)
                        else ai_ref_df["EastErrorProp"][-1]
                    )
                    ai_up_error = (
                        ai_ref_df["UpErrorProp"].iloc[-1]
                        if isinstance(ai_ref_df["UpErrorProp"], pd.Series)
                        else ai_ref_df["UpErrorProp"][-1]
                    )
                else:
                    ai_north_error = ai_ref_df["NorthErrorProp"]
                    ai_east_error = ai_ref_df["EastErrorProp"]
                    ai_up_error = ai_ref_df["UpErrorProp"]

                ai_errors = np.fabs(
                    np.array([ai_north_error, ai_east_error, ai_up_error])
                )

                self.log_data["ai_errors"].append(
                    {
                        "Epoch": (
                            ai_ref_df["RawEpoch"].iloc[-1]
                            if ai_ref_df["RawEpoch"].iloc[-1] != gps_start_date
                            else pe_ref_df["RawEpoch"].iloc[-1]
                        ),
                        "NorthError": ai_ref_df["NorthErrorProp"].iloc[-1],
                        "EastError": ai_ref_df["EastErrorProp"].iloc[-1],
                        "UpError": ai_ref_df["UpErrorProp"].iloc[-1],
                        "HorizontalError": ai_ref_df["HorizontalErrorProp"].iloc[-1],
                        "VerticalError": ai_ref_df["VerticalErrorProp"].iloc[-1],
                    }
                )

                self.ai_errors_values.update(ai_errors)
                self.base_errors_values.update(base_errors)

                reward = self._calculate_element_wise_log_ratio_rewards(
                    ai_errors, base_errors
                )

            elif ai_ref_df is None and not pe_ref_df.empty:
                reward = np.array((-10.0, -10.0, -10.0))

            else:
                reward = np.zeros(3)

        except Exception as e:
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.REWARD,
                f"Error in compute_reward: {e}",
            )
            reward = np.zeros(3)

        reward = np.clip(reward, -10.0, 10.0, dtype=np.float32)
        self.reward.update(reward)

        instant_reward = self.reward.get_differentiated_value()
        running_reward = self.reward.get_running_value()
        cumulative_reward = self.reward.get_cumulative_value()

        self._times["compute_reward"].append(time.time() - start)

        self.log_data["instant_rewards"].append(instant_reward)
        self.log_data["running_rewards"].append(running_reward)
        self.log_data["cumulative_rewards"].append(cumulative_reward)

        self.reward_rec.record(epoch, instant_reward, cumulative_reward)

        return instant_reward

    def get_log_data(self):
        data = self.log_data.copy()

        if "ai_errors" in data and isinstance(data["ai_errors"], list):
            flattened_ai_errors = {}
            for error_dict in data["ai_errors"]:
                for key, value in error_dict.items():
                    if key not in flattened_ai_errors:
                        flattened_ai_errors[key] = []
                    flattened_ai_errors[key].append(value)
            data["ai_errors"] = flattened_ai_errors

        self.log_data = {key: [] for key in self.log_data}
        return data

    def _get_record_output(self, data: OutputStr) -> pd.DataFrame:
        record = {}

        raw_epoch = pd.to_datetime(
            GPS_Time(
                w=data.output_PE.timestamp_week, s=data.output_PE.timestamp_second
            ).calendar_column_str_d(),
            format=CALENDAR_COLUMN_D_FORMAT,
        )
        record["RawEpoch"] = [raw_epoch]
        record["Epoch"] = [raw_epoch.round(freq="100ms")]
        pe_pos_out = data.output_PE.pe_pos_out

        record["X"] = [pe_pos_out.pos_out[0]]
        record["Y"] = [pe_pos_out.pos_out[1]]
        record["Z"] = [pe_pos_out.pos_out[2]]

        latitude = (
            pe_pos_out.latitude
            if pe_pos_out.latitude_direction == Latitude_Direction.NORTH
            else -pe_pos_out.latitude
        )
        longitude = (
            pe_pos_out.longitude
            if pe_pos_out.longitude_direction == Longitude_Direction.EAST
            else -pe_pos_out.longitude
        )

        lat_lon_hei = Point(latitude, longitude, pe_pos_out.height)
        record["LAT"] = [lat_lon_hei.latitude]
        record["LON"] = [lat_lon_hei.longitude]
        record["HEI"] = [lat_lon_hei.altitude]

        record["ALL_Fix"] = [data.output_PE.pe_solution_info.total_used_sats_fixed]
        record["ALL_Float"] = [data.output_PE.pe_solution_info.total_used_sats_float]
        record["SVG_Float"] = [
            sum(
                1
                for sat in data.output_PE.pe_solution_info.used_sats_info_gps
                if sat.status
            )
        ]
        record["SVE_Float"] = [
            sum(
                1
                for sat in data.output_PE.pe_solution_info.used_sats_info_gal
                if sat.status
            )
        ]
        record["SVC_Float"] = [
            sum(
                1
                for sat in data.output_PE.pe_solution_info.used_sats_info_bds
                if sat.status
            )
        ]
        record["ALL_One_Freq"] = [
            data.output_PE.pe_solution_info.total_used_sats_float_one_freq
        ]
        record["QUALITY"] = [data.output_PE.pe_solution_info.systemStatus]

        if self.refMgr.ref_mode != ReferenceMode.STATIC:
            record["Vx"] = [data.output_PE.pe_vel_out.vel_out[0]]
            record["Vy"] = [data.output_PE.pe_vel_out.vel_out[1]]
            record["Vz"] = [data.output_PE.pe_vel_out.vel_out[2]]

        record["Heading"] = [data.output_PE.pe_imu_odo_out.att[2]]
        record["TrackAngle"] = [data.output_PE.pe_vel_out.TrackAngle]
        record["VEL_CORE_VALIDITY"] = [
            data.output_PE.pe_solution_info.vel_core_validity
        ]

        record = pd.DataFrame(record, columns=const.RTPPP_COLUMNS)

        return record

    def _match_and_extract_common_epochs(self, df_rtppp: pd.DataFrame):
        common_epochs = pd.merge(
            df_rtppp[["Epoch"]], self.reference_data[["Epoch"]], on="Epoch", how="inner"
        )

        if common_epochs.empty:
            return None

        df_rtppp_common = df_rtppp[
            df_rtppp["Epoch"].isin(common_epochs["Epoch"])
        ].reset_index(drop=True)
        df_reference_common = self.reference_data[
            self.reference_data["Epoch"].isin(common_epochs["Epoch"])
        ].reset_index(drop=True)

        columns_to_extract = [
            "Epoch",
            "RawEpoch",
            "X",
            "Y",
            "Z",
            "LAT",
            "LON",
            "HEI",
            "Vx",
            "Vy",
            "Vz",
            "QUALITY",
            "Heading",
            "TrackAngle",
            "ALL_Fix",
            "ALL_Float",
            "SVG_Float",
            "SVE_Float",
            "SVC_Float",
            "ALL_One_Freq",
            "VEL_CORE_VALIDITY",
        ]

        ref_columns_map = {
            "X-ECEF": "X_REF",
            "Y-ECEF": "Y_REF",
            "Z-ECEF": "Z_REF",
            "Latitude": "LAT_REF",
            "Longitude": "LON_REF",
            "H-Ell": "HEI_REF",
            "VNorth": "VN_REF",
            "VEast": "VE_REF",
            "VUp": "VU_REF",
            "Heading": "Heading_REF",
            "Q": "Q_REF",
            "iFlag": "iFlag_REF",
            "AmbStatus": "AmbStatus_REF",
        }

        df_data_ref = pd.concat(
            [
                df_rtppp_common[columns_to_extract],
                df_reference_common[list(ref_columns_map.keys())].rename(
                    columns=ref_columns_map
                ),
            ],
            axis=1,
        )

        return df_data_ref

    def _get_error_position_ref_span(self, df_data_ref):
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.REWARD,
            "Calculating NEU Position Error between RTPPP and SPAN",
        )

        df_data_ref_aux = df_data_ref

        df_data_ref_aux["X_PROP"] = df_data_ref_aux.loc[:, "X"]
        df_data_ref_aux["Y_PROP"] = df_data_ref_aux.loc[:, "Y"]
        df_data_ref_aux["Z_PROP"] = df_data_ref_aux.loc[:, "Z"]

        df_data_ref_aux["LAT_PROP"] = df_data_ref_aux.loc[:, "LAT"]
        df_data_ref_aux["LON_PROP"] = df_data_ref_aux.loc[:, "LON"]
        df_data_ref_aux["HEI_PROP"] = df_data_ref_aux.loc[:, "HEI"]

        if (
            (df_data_ref_aux["LAT_PROP"] == 0.0).all()
            or (df_data_ref_aux["LON_PROP"] == 0.0).all()
            or (df_data_ref_aux["HEI_PROP"] == 0.0).all()
        ):
            Logger.log_message(
                Logger.Category.WARNING,
                Logger.Module.REWARD,
                "The RTPPP has not LAT, LON or HEI. Computing LAT, LON and HEI from XYZ",
            )
            (
                df_data_ref_aux["LAT_PROP"],
                df_data_ref_aux["LON_PROP"],
                df_data_ref_aux["HEI_PROP"],
            ) = car2geo(
                df_data_ref_aux["X_PROP"],
                df_data_ref_aux["Y_PROP"],
                df_data_ref_aux["Z_PROP"],
            )

        df_data_ref_aux["X_REFPROP"] = df_data_ref_aux.loc[:, "X_REF"]
        df_data_ref_aux["Y_REFPROP"] = df_data_ref_aux.loc[:, "Y_REF"]
        df_data_ref_aux["Z_REFPROP"] = df_data_ref_aux.loc[:, "Z_REF"]

        df_data_ref_aux["LAT_REFPROP"] = df_data_ref_aux.loc[:, "LAT_REF"]
        df_data_ref_aux["LON_REFPROP"] = df_data_ref_aux.loc[:, "LON_REF"]
        df_data_ref_aux["HEI_REFPROP"] = df_data_ref_aux.loc[:, "HEI_REF"]

        epoch_not_equal_to_raw_epoch = ~df_data_ref_aux["RawEpoch"].isin(
            df_data_ref_aux["Epoch"]
        )

        if epoch_not_equal_to_raw_epoch.any():
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.REWARD,
                "The epochs between RTPPP and SPAN are not the same, so it is necessary to propagate the pos error epochs",
            )
            df_data_ref_aux.loc[epoch_not_equal_to_raw_epoch] = (
                self._propagate_pos_df_data_ref(
                    df_data_ref_aux.loc[epoch_not_equal_to_raw_epoch]
                )
            )

        df_data_ref_aux = self._calculate_neu_hv_error(df_data_ref_aux)

        return df_data_ref_aux

    def _propagate_pos_df_data_ref(self, df_data_ref):
        delta_t = (df_data_ref["Epoch"] - df_data_ref["RawEpoch"]).dt.total_seconds()
        df_data_ref.loc[:, "X_PROP"] = df_data_ref["X"] + df_data_ref["Vx"] * delta_t
        df_data_ref.loc[:, "Y_PROP"] = df_data_ref["Y"] + df_data_ref["Vy"] * delta_t
        df_data_ref.loc[:, "Z_PROP"] = df_data_ref["Z"] + df_data_ref["Vz"] * delta_t

        (
            df_data_ref.loc[:, "LAT_PROP"],
            df_data_ref.loc[:, "LON_PROP"],
            df_data_ref.loc[:, "HEI_PROP"],
        ) = car2geo(df_data_ref["X_PROP"], df_data_ref["Y_PROP"], df_data_ref["Z_PROP"])

        return df_data_ref

    def _calculate_neu_hv_error(self, df_data_ref_aux: pd.DataFrame):
        df_data_ref_aux["NorthErrorProp"] = (
            np.deg2rad(df_data_ref_aux["LAT_PROP"] - df_data_ref_aux["LAT_REFPROP"])
        ) * np.sqrt(
            df_data_ref_aux["X_REFPROP"] ** 2
            + df_data_ref_aux["Y_REFPROP"] ** 2
            + df_data_ref_aux["Z_REFPROP"] ** 2
        )

        df_data_ref_aux["EastErrorProp"] = np.deg2rad(
            df_data_ref_aux["LON_PROP"] - df_data_ref_aux["LON_REFPROP"]
        ) * np.sqrt(
            df_data_ref_aux["X_REFPROP"] ** 2 + df_data_ref_aux["Y_REFPROP"] ** 2
        )

        df_data_ref_aux["UpErrorProp"] = (
            df_data_ref_aux["HEI_PROP"] - df_data_ref_aux["HEI_REFPROP"]
        )

        df_data_ref_aux["HorizontalErrorProp"] = np.sqrt(
            (df_data_ref_aux["NorthErrorProp"] ** 2)
            + (df_data_ref_aux["EastErrorProp"] ** 2)
        )
        df_data_ref_aux["VerticalErrorProp"] = np.fabs(df_data_ref_aux["UpErrorProp"])

        return df_data_ref_aux

    def _calculate_element_wise_log_ratio_rewards(self, ai_errors, base_errors):
        """
        Calculate element-wise log-ratio rewards between AI errors and base errors.

        Args:
            ai_errors (numpy.ndarray): Array of absolute AI error values
            base_errors (numpy.ndarray): Array of absolute base error values

        Returns:
            numpy.ndarray: Array of log-ratio rewards for each error component
        """
        # Ensure inputs are numpy arrays
        ai_errors = np.array(ai_errors)
        base_errors = np.array(base_errors)

        # Add small epsilon to avoid division by zero
        epsilon = 1e-6

        # Initialize rewards array with zeros
        rewards = np.zeros_like(ai_errors, dtype=float)

        # Calculate rewards element-wise based on the condition
        better_indices = ai_errors <= base_errors
        worse_indices = ~better_indices

        # For elements where AI performs better than or equal to base
        rewards[better_indices] = np.log(
            base_errors[better_indices] / (ai_errors[better_indices] + epsilon)
        )

        # For elements where AI performs worse than base
        rewards[worse_indices] = -np.log(
            ai_errors[worse_indices] / (base_errors[worse_indices] + epsilon)
        )

        return np.nan_to_num(rewards, nan=-10.0, posinf=10.0, neginf=-10.0)

    def finalize_evaluation(self):
        base_errors = {
            "Epoch": self.base_data["RawEpoch"],
            "NorthError": self.base_data["NorthErrorProp"],
            "EastError": self.base_data["EastErrorProp"],
            "UpError": self.base_data["UpErrorProp"],
            "HorizontalError": self.base_data["HorizontalErrorProp"],
            "VerticalError": self.base_data["VerticalErrorProp"],
        }

        ai_ref_df = self._match_and_extract_common_epochs(self.ai_data)
        if ai_ref_df is not None:
            ai_df = self._get_error_position_ref_span(ai_ref_df)
            # Guardamos posiciones IA para el mapa
            ai_pos = ai_df[["RawEpoch", "LAT_PROP", "LON_PROP"]].set_index("RawEpoch")
            self.ai_positions = pd.concat([self.ai_positions, ai_pos])
            # Errores IA
            ai_errors = {
                "Epoch": ai_df["RawEpoch"],
                "NorthError": ai_df["NorthErrorProp"],
                "EastError": ai_df["EastErrorProp"],
                "UpError": ai_df["UpErrorProp"],
                "HorizontalError": ai_df["HorizontalErrorProp"],
                "VerticalError": ai_df["VerticalErrorProp"],
            }
        else:
            ai_errors = None

        rewards = None
        if base_errors is not None and ai_errors is not None:
            df_base = self.base_data.set_index("RawEpoch")
            df_ai = ai_df.set_index("RawEpoch")
            common = df_base.index.intersection(df_ai.index)
            be = np.column_stack(
                [
                    df_base.loc[common, "NorthErrorProp"].values,
                    df_base.loc[common, "EastErrorProp"].values,
                    df_base.loc[common, "UpErrorProp"].values,
                ]
            )
            ae = np.column_stack(
                [
                    df_ai.loc[common, "NorthErrorProp"].values,
                    df_ai.loc[common, "EastErrorProp"].values,
                    df_ai.loc[common, "UpErrorProp"].values,
                ]
            )
            rr = self._calculate_element_wise_log_ratio_rewards(ae, be)

            rewards = np.clip(rr, -10.0, 10.0, dtype=np.float32)

            instants = []
            cumulatives = []

            for reward in rewards:
                self.reward.update(reward)

                instants.append(self.reward.get_differentiated_value())
                cumulatives.append(self.reward.get_cumulative_value())

            rows = [
                [
                    f"{epoch.year:4}  {epoch.month:2}  {epoch.day:2}  {epoch.hour:2}  {epoch.minute:2}  {epoch.second + epoch.microsecond/1e6:9.6f}",
                    inst,
                    cum,
                ]
                for epoch, inst, cum in zip(list(common), instants, cumulatives)
            ]
            self.reward_rec.write_batch(rows)

            rewards = {"Epoch": list(common), "Rewards": rewards}

        return {"base_errors": base_errors, "ai_errors": ai_errors, "rewards": rewards}

    def _create_map(self):
        self.map_initialized = True
        self.last_ai_position_index = 0
        self.map = folium.Map(
            location=[self.ref_positions.iloc[0, 0], self.ref_positions.iloc[0, 1]],
            zoom_start=15,
        )
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri Satellite",
            overlay=False,
            control=False,
            max_zoom=20,
        ).add_to(self.map)

        self.ref_group = folium.FeatureGroup(name="Reference Path").add_to(self.map)
        self.base_group = folium.FeatureGroup(name="Base Path").add_to(self.map)
        self.ai_group = folium.FeatureGroup(name="AI Path").add_to(self.map)

        # Create and add the reference and base paths
        ref_positions = self.ref_positions
        if hasattr(self, "initial_epoch") and hasattr(self, "final_epoch"):
            ref_positions = ref_positions[
                ref_positions.index.to_series().between(
                    self.initial_epoch, self.final_epoch
                )
            ]
        self.ref_polyline = folium.PolyLine(
            ref_positions.values.tolist(),
            color="blue",
            weight=2.5,
            opacity=1,
            tooltip="Reference Path",
        ).add_to(self.ref_group)

        base_positions = self.base_positions
        if hasattr(self, "initial_epoch") and hasattr(self, "final_epoch"):
            base_positions = base_positions[
                base_positions.index.to_series().between(
                    self.initial_epoch, self.final_epoch
                )
            ]

        self.base_polyline = folium.PolyLine(
            base_positions.values.tolist(),
            color="green",
            weight=2.5,
            opacity=1,
            tooltip="Base Path",
        ).add_to(self.base_group)

        folium.LayerControl(overlay=True).add_to(self.map)

    def update_map(self, reset=False):
        if not self.map_initialized or reset:
            self._create_map()

        new_positions = self.ai_positions.iloc[
            self.last_ai_position_index :
        ].values.tolist()

        if new_positions:
            self.map.location = new_positions[-1]

            new_ai_polyline = folium.PolyLine(
                new_positions,
                color="red",
                weight=2.5,
                opacity=1,
                tooltip="AI Path",
            )
            self.ai_group.add_child(new_ai_polyline)

            self.last_ai_position_index = len(self.ai_positions)

            self.map.save(self.map_file)

        return self.map_file

    def match_ref(self, epoch: Union[pd.Timestamp, GPS_Time]) -> bool:
        if isinstance(epoch, GPS_Time):
            raw_epoch = pd.to_datetime(
                epoch.calendar_column_str_d(),
                format=CALENDAR_COLUMN_D_FORMAT,
            )
        else:
            raw_epoch = epoch

        epoch = raw_epoch.round(freq="100ms")

        return self.reference_data["Epoch"].isin([epoch]).any()

    def check_reconvergence(self):
        """
        Check if the current error (RMSE) exceeds 3sigma
        (three standard deviations) from their respective means, indicating
        the need for reconvergence.
        """
        # Ensure there is sufficient data to calculate mean and standard deviation
        if (
            not self.ai_errors_values.is_initialized
            or not self.base_errors_values.is_initialized
            or self.ai_errors_values.count < 2
            or self.base_errors_values.count < 2
        ):
            return False

        # Calculate mean and standard deviation of past RMSE values
        ai_mean_rmse = self.ai_errors_values.get_mean()
        ai_std_rmse = self.ai_errors_values.get_std()
        max_ai_threshold = ai_mean_rmse + (3 * ai_std_rmse)

        pe_mean_rmse = self.base_errors_values.get_mean()
        pe_std_rmse = self.base_errors_values.get_std()
        max_pe_threshold = pe_mean_rmse + (3 * pe_std_rmse)

        # Get the current RMSE
        current_ai_rmse = self.ai_errors_values.get_differentiated_value()

        if any(self.ai_errors_values.get_cumulative_value() > max_pe_threshold):
            self.rmse_patience += 1

        # Determine if reconvergence is needed based on RMSE or reward
        reconvergence_needed = (
            any(current_ai_rmse > max_ai_threshold)
            or any(current_ai_rmse > max_pe_threshold)
            or self.rmse_patience > 10
        )

        if reconvergence_needed:
            self.rmse_patience = 0

        return reconvergence_needed

    def get_ai_positions(self):
        return self.ai_positions

    def get_reward_filepath(self):
        return self.reward_rec.file_path
