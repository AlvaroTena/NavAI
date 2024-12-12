import csv
import os

from navutils.logger import Logger
from pewrapper.types import GPS_Time


class RewardRecorder:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.file_path = ""
        self.output_file = None
        self.csv_writer = None

    def reset(self, output_path: str):
        self.close()
        self.__init__(output_path)

    def initialize(self, epoch_file: GPS_Time):
        os.makedirs(self.output_path, exist_ok=True)
        self.file_path = os.path.join(
            self.output_path,
            f"reward_{self.get_date_filename(epoch_file)}.csv",
        )
        self.output_file = open(self.file_path, "w", newline="")
        self.csv_writer = csv.writer(self.output_file)
        header = [
            "epoch",
            "reward",
        ]
        self.csv_writer.writerow(header)
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.WRITER,
            f"Reward recording started, saving to {self.file_path}",
        )

    def record(self, epoch: GPS_Time, reward):
        if not self.output_file or self.output_file.closed:
            raise ValueError("Recorder has not been initialized or has been closed")

        row = [epoch.calendar_column_str_d(), reward]
        self.csv_writer.writerow(row)
        self.output_file.flush()

    def close(self):
        if self.output_file is not None:
            self.output_file.close()
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.WRITER,
                f"Reward recorder closed",
            )

    @staticmethod
    def get_date_filename(date: GPS_Time):
        return f"{date.year()}{date.month()}{date.day()}_{date.hour()}{date.min()}{date.day_sec()}"
