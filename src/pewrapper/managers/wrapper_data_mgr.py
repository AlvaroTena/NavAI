import re
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from navutils.logger import Logger
from navutils.singleton import Singleton
from pewrapper.managers import ConfigurationManager
from pewrapper.managers.decoder_manager import Decoder_Manager
from pewrapper.types.gps_time_wrapper import CALENDAR_COLUMN_D_FORMAT, GPS_Time


class WrapperDataManager(metaclass=Singleton):
    def __init__(
        self,
        initial_epoch_constr: GPS_Time,
        final_epoch_constr: GPS_Time,
        configMgr: ConfigurationManager,
    ):
        self.configMgr_ = configMgr
        self._wrapper_file_data: pd.DataFrame = pd.DataFrame()
        self.decoder_manager_ = Decoder_Manager(configMgr)
        self.initial_epoch = initial_epoch_constr
        self.final_epoch = final_epoch_constr

    def reset(
        self,
        initial_epoch_constr: GPS_Time,
        final_epoch_constr: GPS_Time,
    ):
        self.__init__(initial_epoch_constr, final_epoch_constr, self.configMgr_)
        self.decoder_manager_.reset()

    def set_subset_epochs(
        self,
        initial_epoch: Optional[GPS_Time] = None,
        final_epoch: Optional[GPS_Time] = None,
    ) -> None:
        """
        Set time constraints for a subset of data.

        Args:
            initial_epoch: Optional starting time constraint
            final_epoch: Optional ending time constraint
        """
        self.subset_initial_epoch = (
            initial_epoch if initial_epoch is not None else self.initial_epoch
        )
        self.subset_final_epoch = (
            final_epoch if final_epoch is not None else self.final_epoch
        )

    def parse_wrapper_file(self, filename: str, parsing_rate: int) -> Tuple[bool, str]:
        addInfo = []
        Logger.log_message(
            Logger.Category.DEBUG, Logger.Module.MAIN, " Parsing wrapper file"
        )

        result = self.decoder_manager_.initialise()

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

            imu_latency = pd.to_timedelta(-self.configMgr_.IMU_latency_, unit="s")
            odo_latency = pd.to_timedelta(-self.configMgr_.ODO_latency_, unit="s")

            df["epoch"] = np.where(
                df["msg_type"] == "IMU",
                df["epoch"] + imu_latency,
                np.where(
                    df["msg_type"] == "ODOMETER", df["epoch"] + odo_latency, df["epoch"]
                ),
            )

            df["gnss_epoch"] = pd.NaT
            df.loc[df["msg_type"] == "GNSS", "gnss_epoch"] = self._extract_gnss_epochs(
                df.loc[df["msg_type"] == "GNSS", "msg_data"]
            )

            df.sort_values("epoch", inplace=True)
            df["gnss_epoch"] = df["gnss_epoch"].ffill()

            self._wrapper_file_data = df[
                ["epoch", "gnss_epoch", "msg_type", "msg_data"]
            ]

            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.READER,
                f" Wrapper file processed (Number of lines: {wrapper_lines})",
            )
            result &= True

        except IOError as e:
            addInfo = f"Unable to open file ({filename}): {e}"
            result &= False

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

    def _extract_gnss_epochs(self, msg_series: pd.Series) -> pd.Series:
        """Extract GNSS epochs from a series of GNSS message data.

        Processes each GNSS message in the input Series by calling the decoder manager
        to extract epoch information from each message.

        Args:
            msg_series: A pandas Series containing GNSS message data (hex strings)

        Returns:
            A pandas Series containing the extracted GNSS epochs (as pd.Timestamp objects)
        """

        def extract_epoch(msg_data: str):
            if not isinstance(msg_data, str) or pd.isna(msg_data):
                return pd.NaT
            try:
                raw_msg = bytes.fromhex(msg_data)
            except Exception:
                return pd.NaT

            result, msg_result, epoch = (
                self.decoder_manager_.extract_epoch_from_gnss_message(
                    raw_msg, len(raw_msg)
                )
            )
            if result and msg_result:
                return pd.to_datetime(
                    epoch.calendar_column_str_d(),
                    format=CALENDAR_COLUMN_D_FORMAT,
                )
            else:
                return pd.NaT

        return msg_series.apply(extract_epoch)

    def _filter_epochs(
        self, filter_subset: bool = False, ignore_subset_initial_epoch: bool = False
    ) -> pd.DataFrame:
        """
        Filter DataFrame rows based on time epochs within specified range.

        Args:
            filter_subset: Whether to use subset epoch range instead of main range
            ignore_subset_initial_epoch: Whether to use main initial epoch even when filtering by subset

        Returns:
            DataFrame containing only rows within the specified time range
        """
        # Determine initial epoch based on parameters
        initial_epoch_str = (
            self.initial_epoch.calendar_column_str_d()
            if not filter_subset or ignore_subset_initial_epoch
            else self.subset_initial_epoch.calendar_column_str_d()
        )

        # Determine final epoch based on parameters
        final_epoch_str = (
            self.final_epoch.calendar_column_str_d()
            if not filter_subset
            else self.subset_final_epoch.calendar_column_str_d()
        )

        # Convert string representations to datetime objects
        date_format = CALENDAR_COLUMN_D_FORMAT
        initial_epoch = pd.to_datetime(initial_epoch_str, format=date_format)
        final_epoch = pd.to_datetime(final_epoch_str, format=date_format)

        # Filter data based on epoch range
        mask = (self._wrapper_file_data["gnss_epoch"] >= initial_epoch) & (
            self._wrapper_file_data["gnss_epoch"] <= final_epoch
        )

        return self._wrapper_file_data[mask]

    def items(self, filter_subset: bool = False):
        return self._filter_epochs(filter_subset).to_numpy()

    def get(
        self, key: GPS_Time, default=pd.DataFrame(columns=["msg_type", "msg_data"])
    ) -> pd.DataFrame:
        df = self._filter_epochs()
        result = df.loc[
            df["gnss_epoch"]
            == pd.to_datetime(
                key.calendar_column_str_d(), format=CALENDAR_COLUMN_D_FORMAT
            ),
            ["msg_type", "msg_data"],
        ]
        return result if not result.empty else default

    def get_iterator(
        self, filter_subset: bool = False, ignore_subset_initial_epoch: bool = False
    ) -> iter:
        """
        Returns an iterator over grouped epochs from a filtered DataFrame.

        Args:
            filter_subset: Boolean flag to determine if subset filtering should be applied
            ignore_subset_initial_epoch: Boolean flag to control initial epoch handling during filtering

        Returns:
            Iterator over tuples of (epoch_number, epoch_dataframe)
        """
        df = self._filter_epochs(filter_subset, ignore_subset_initial_epoch)
        return iter(df.groupby("epoch"))

    def __iter__(self):
        df = self._filter_epochs()
        return iter(df.groupby("epoch"))
