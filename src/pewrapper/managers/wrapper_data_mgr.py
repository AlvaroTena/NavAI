import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from navutils.logger import Logger
from navutils.singleton import Singleton
from pewrapper.managers import ConfigurationManager
from pewrapper.types.gps_time_wrapper import GPS_Time


class WrapperDataManager(metaclass=Singleton):
    def __init__(
        self,
        initial_epoch_constr: GPS_Time,
        final_epoch_constr: GPS_Time,
        configMgr: ConfigurationManager,
    ):
        self.configMgr_ = configMgr
        self._wrapper_file_data: pd.DataFrame = pd.DataFrame()
        self.initial_epoch = initial_epoch_constr
        self.final_epoch = final_epoch_constr

    def reset(
        self,
        initial_epoch_constr: GPS_Time,
        final_epoch_constr: GPS_Time,
    ):
        self.__init__(initial_epoch_constr, final_epoch_constr, self.configMgr_)

    def parse_wrapper_file(self, filename: str, parsing_rate: int) -> Tuple[bool, str]:
        addInfo = []
        Logger.log_message(
            Logger.Category.DEBUG, Logger.Module.MAIN, " Parsing wrapper file"
        )

        try:
            if filename.endswith(".parquet"):
                df = pd.read_parquet(filename)
                is_parquet = True

            elif filename.endswith(".txt"):
                df = pd.read_csv(
                    filename,
                    header=None,
                    comment="#",
                    sep="\r\n",
                    names=["line"],
                    engine="python",
                    dtype=str,
                )
                is_parquet = False

            else:
                raise IOError(f"Unsupported file format: {filename}")

            df, addInfo = self._parse_wrapper_checks(
                df, self.configMgr_.compute_log_, is_parquet
            )

            wrapper_lines = len(df)

            for addInfo_line in addInfo:
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.READER,
                    addInfo_line,
                )
            addInfo.clear()

            df["prev_epoch"] = pd.NaT
            rate_msg_types = ["GNSS", "COMPUTE", "COMPUTE_RESET"]
            rate_mask = df["msg_type"].isin(rate_msg_types)
            df.loc[rate_mask, "prev_epoch"] = df.loc[rate_mask, "epoch"].shift(1)
            df["prev_epoch"] = df["prev_epoch"].ffill()

            df["parse_line"] = (
                (parsing_rate == 0)
                | (~rate_mask)
                | ((df["epoch"] - df["prev_epoch"]).dt.total_seconds() >= parsing_rate)
            )
            addInfo.extend(
                df.loc[~df["parse_line"], "epoch"]
                .apply(
                    lambda x: f"Epoch skipped due to rate check {x.strftime('%Y  %m  %d  %H  %M  %S.%f')} Rate: {parsing_rate}",
                    axis=1,
                )
                .tolist()
            )
            for info in addInfo:
                Logger.log_message(
                    Logger.Category.TRACE,
                    Logger.Module.READER,
                    info,
                )
            addInfo.clear()
            df = df[df["parse_line"]]

            initial_epoch_dt = pd.to_datetime(
                self.initial_epoch.calendar_column_str_d(),
                format="%Y %m %d %H %M %S.%f",
            )
            final_epoch_dt = pd.to_datetime(
                self.final_epoch.calendar_column_str_d(), format="%Y %m %d %H %M %S.%f"
            )
            df = df[(df["epoch"] >= initial_epoch_dt) & (df["epoch"] <= final_epoch_dt)]

            imu_latency = pd.to_timedelta(-self.configMgr_.IMU_latency_, unit="s")
            odo_latency = pd.to_timedelta(-self.configMgr_.ODO_latency_, unit="s")

            df["epoch"] = np.where(
                df["msg_type"] == "IMU",
                df["epoch"] + imu_latency,
                np.where(
                    df["msg_type"] == "ODOMETER", df["epoch"] + odo_latency, df["epoch"]
                ),
            )
            df.sort_values("epoch", inplace=True)

            self._wrapper_file_data = df[["epoch", "msg_type", "msg_data"]]

            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.READER,
                f" Wrapper file processed (Number of lines: {wrapper_lines})",
            )
            result = True

        except IOError as e:
            addInfo = f"Unable to open file ({filename}): {e}"
            result = False

        return result, addInfo

    def _parse_wrapper_checks(
        self, df: pd.DataFrame, compute_log: bool, is_parquet: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        addInfo = []

        N_VALID_FIELDS = 4
        N_VALID_FIELDS_COMPUTE = 5
        N_VALID_FIELDS_COMPUTE_OTH = 6

        if not is_parquet:
            df["split_vector"] = df["line"].apply(lambda x: re.split(r"\s+", x.strip()))
            df["n_fields"] = df["split_vector"].apply(len)

            df["valid_n_fields"] = df["n_fields"] >= N_VALID_FIELDS
            df.loc[df["valid_n_fields"], "msg_type"] = df.loc[
                df["valid_n_fields"], "split_vector"
            ].apply(lambda x: x[2] if len(x) > 2 else "")

        else:
            # count epoch as 2 fields (date and time) plus 1 for message type
            df["n_fields"] = 3 + df["message"].str.split(r"\s+").str.len()
            df.rename(columns={"type": "msg_type"}, inplace=True)

        valid_n_fields_mask = (df["n_fields"] == N_VALID_FIELDS) | (
            df["n_fields"].isin([N_VALID_FIELDS_COMPUTE, N_VALID_FIELDS_COMPUTE_OTH])
            & df["msg_type"].isin(["COMPUTE", "COMPUTE_RESET"])
        )
        addInfo.extend(
            df.loc[~valid_n_fields_mask]
            .apply(
                lambda row: (
                    f"Number of fields in record incorrect: {row['n_fields']} vs {N_VALID_FIELDS} (or {N_VALID_FIELDS_COMPUTE} in COMPUTE line)",
                    row.name + 1,
                    row["line"] if not is_parquet else row["message"],
                ),
                axis=1,
                result_type="reduce",
            )
            .tolist()
        )

        df = df[valid_n_fields_mask]
        compute_mask = df["n_fields"] == N_VALID_FIELDS_COMPUTE
        compute_oth_mask = df["n_fields"] == N_VALID_FIELDS_COMPUTE_OTH

        if not is_parquet:
            df["msg_data"] = df["split_vector"].apply(lambda x: x[3])

            df.loc[compute_mask, "msg_data"] += "   " + df.loc[
                compute_mask, "split_vector"
            ].apply(lambda x: x[4])
            df.loc[compute_oth_mask, "msg_data"] += "   " + df.loc[
                compute_oth_mask, "split_vector"
            ].apply(lambda x: x[4] + "   " + x[5])

            df["msg_data"] = df["msg_data"].str.upper()
            df["msg_data"] = df["msg_data"].str.strip()

            # Extraer fecha y hora de split_vector
            df["date"] = df["split_vector"].apply(lambda x: x[0])
            df["time"] = df["split_vector"].apply(lambda x: x[1])

            # Convertir fecha y hora a datetime y manejar errores
            df["epoch"] = pd.to_datetime(
                df["date"] + " " + df["time"],
                format="%Y/%m/%d %H:%M:%S.%f",
                errors="coerce",
            )

            invalid_epoch_mask = df["epoch"].isnull()
            addInfo.extend(
                df.loc[invalid_epoch_mask, "split_vector"]
                .apply(
                    lambda row: (
                        f"Record timestamp bad formed. Date: {row['split_vector'][0]}. Time: {row['split_vector'][1]}",
                        row.name + 1,
                        row["line"],
                    ),
                    axis=1,
                    result_type="reduce",
                )
                .tolist()
            )

        else:
            df.rename(columns={"message": "msg_data"}, inplace=True)

            invalid_epoch_mask = df["epoch"].isnull()
            addInfo.extend(
                df.loc[invalid_epoch_mask, "epoch"]
                .apply(
                    lambda row: (
                        f"Record timestamp bad formed. Epoch: {row['epoch']}",
                        row.name + 1,
                        row["epoch"],
                    ),
                    axis=1,
                    result_type="reduce",
                )
                .tolist()
            )

        df = df[~invalid_epoch_mask]

        if compute_log:
            valid_msg_types = [
                "CS",
                "GNSS",
                "IMU",
                "COMPUTE",
                "COMPUTE_RESET",
                "ODOMETER",
                "CP",
            ]
        else:
            valid_msg_types = ["CS", "GNSS", "IMU", "ODOMETER", "CP"]

        invalid_msg_type_mask = ~df["msg_type"].isin(valid_msg_types)

        df = df[~invalid_msg_type_mask]

        bytes_msg_types = ~df["msg_type"].isin(
            ["IMU", "ODOMETER", "COMPUTE", "COMPUTE_RESET", "CP"]
        )
        invalid_msg_length_mask = bytes_msg_types & (df["msg_data"].apply(len) % 2 != 0)
        addInfo.extend(
            df.loc[invalid_msg_length_mask]
            .apply(
                lambda row: (
                    f"Message length ({len(row['msg_data'])}) does not match an integer number of bytes",
                    row.name + 1,
                    row["line"] if not is_parquet else row["msg_data"],
                ),
                axis=1,
                result_type="reduce",
            )
            .tolist()
        )
        df = df[~invalid_msg_length_mask]

        valid_hex_chars = set("0123456789abcdefABCDEF")
        invalid_hex_mask = bytes_msg_types & ~df.loc[bytes_msg_types, "msg_data"].apply(
            lambda x: set(str(x)).issubset(valid_hex_chars)
        )
        addInfo.extend(
            df.loc[invalid_hex_mask]
            .apply(
                lambda row: (
                    f"Message content bad formed, wrong hex format (msg: {row['msg_data']})",
                    row.name + 1,
                    row["line"] if not is_parquet else row["msg_data"],
                ),
                axis=1,
                result_type="reduce",
            )
            .tolist()
        )
        df = df[~invalid_hex_mask]

        addInfo = [
            f" Problem parsing line: {addInfo_line}. Skipping line {line_number} ({line})"
            for addInfo_line, line_number, line in addInfo
        ]

        return df[["epoch", "msg_type", "msg_data"]], addInfo

    def items(self):
        return self._wrapper_file_data.to_numpy()

    def get(
        self, key: GPS_Time, default=pd.DataFrame(columns=["msg_type", "msg_data"])
    ) -> pd.DataFrame:
        return self._wrapper_file_data.loc[
            self._wrapper_file_data["epoch"]
            == pd.to_datetime(
                key.calendar_column_str_d(), format="%Y %m %d %H %M %S.%f"
            ),
            ["msg_type", "msg_data"],
        ]

    def __iter__(self):
        return iter(self._wrapper_file_data.groupby("epoch"))
