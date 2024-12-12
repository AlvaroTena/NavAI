import hashlib
import time
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

run_id = hashlib.md5(str(time.time()).encode()).hexdigest()

current_file = Path(__file__)
project_dir = current_file.parents[4]
project_dir = project_dir.resolve()

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "retries": 1,
}


dag = DAG(
    "ML-GSharp",
    default_args=default_args,
    description="An ML pipeline with DVC",
    schedule_interval=None,
)

process_data = BashOperator(
    task_id="process_data",
    bash_command=f"dvc repro process_data",
    dag=dag,
    env={"NEPTUNE_CUSTOM_RUN_ID": f"{run_id}"},
    append_env=True,
    cwd=project_dir,
)

data_reports = BashOperator(
    task_id="data_reports",
    bash_command=f"dvc repro data_reports",
    dag=dag,
    env={"NEPTUNE_CUSTOM_RUN_ID": f"{run_id}"},
    append_env=True,
    cwd=project_dir,
)

normalize_data = BashOperator(
    task_id="normalize_data",
    bash_command=f"dvc repro normalize_data",
    dag=dag,
    env={"NEPTUNE_CUSTOM_RUN_ID": f"{run_id}"},
    append_env=True,
    cwd=project_dir,
)

train_model = BashOperator(
    task_id="train_model",
    bash_command=f"dvc repro train_model",
    dag=dag,
    env={"NEPTUNE_CUSTOM_RUN_ID": f"{run_id}"},
    append_env=True,
    cwd=project_dir,
)

evaluate_model = BashOperator(
    task_id="evaluate_model",
    bash_command=f"dvc repro evaluate_model",
    dag=dag,
    env={"NEPTUNE_CUSTOM_RUN_ID": f"{run_id}"},
    append_env=True,
    cwd=project_dir,
)

predict_data = BashOperator(
    task_id="predict_data",
    bash_command=f"dvc repro predict_data",
    dag=dag,
    env={"NEPTUNE_CUSTOM_RUN_ID": f"{run_id}"},
    append_env=True,
    cwd=project_dir,
)

model_reports = BashOperator(
    task_id="model_reports",
    bash_command=f"dvc repro model_reports",
    dag=dag,
    env={"NEPTUNE_CUSTOM_RUN_ID": f"{run_id}"},
    append_env=True,
    cwd=project_dir,
)

(
    process_data
    >> data_reports
    >> normalize_data
    >> train_model
    >> evaluate_model
    >> predict_data
    >> model_reports
)
