import csv
import os
from datetime import datetime
from typing import Dict

from navutils.logger import Logger


class TrainingRecorder:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.output_file = None
        self.csv_writer = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def initialize(self):
        os.makedirs(self.output_path, exist_ok=True)
        file_path = os.path.join(
            self.output_path,
            f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        self.output_file = open(file_path, "w", newline="")
        self.csv_writer = csv.writer(self.output_file)
        header = [
            "epoch",
            "loss",
            "policy_gradient_loss",
            "value_estimation_loss",
            "l2_regularization_loss",
            "entropy_regularization_loss",
            "kl_penalty_loss",
            "clip_fraction",
        ]
        self.csv_writer.writerow(header)
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.WRITER,
            f"Train metrics recording started, saving to {file_path}",
        )

    def record_metrics(self, train_metrics: Dict[str, float]):
        if not self.output_file or self.output_file.closed:
            raise ValueError("Recorder has not been initialized or has been closed")

        epoch = datetime.now().isoformat()
        row = [
            epoch,
            train_metrics["loss"],
            train_metrics["policy_gradient_loss"],
            train_metrics["value_estimation_loss"],
            train_metrics["l2_regularization_loss"],
            train_metrics["entropy_regularization_loss"],
            train_metrics["kl_penalty_loss"],
            train_metrics["clip_fraction"],
        ]
        self.csv_writer.writerow(row)
        self.output_file.flush()

    def close(self):
        if self.output_file:
            self.output_file.close()
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.WRITER,
                f"Train metrics recorder closed",
            )
