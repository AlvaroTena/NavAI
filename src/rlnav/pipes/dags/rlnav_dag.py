from datetime import datetime

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
    "RL-GSharp",
    default_args=default_args,
    description="An RL pipeline with DVC",
    schedule_interval=None,
)

read_and_process_data = BashOperator(
    task_id="read_and_process_data",
    bash_command=f"dvc repro read_and_process_data",
    dag=dag,
    env={"NEPTUNE_CUSTOM_RUN_ID": f"{run_id}"},
    append_env=True,
    cwd=project_dir,
)

read_and_transform_data = BashOperator(
    task_id="read_and_transform_data",
    bash_command=f"cd {project_path} && dvc repro read_and_transform_data",
    dag=dag,
    env={"NEPTUNE_CUSTOM_RUN_ID": f"{run_id}"},
    append_env=True,
    cwd=project_dir,
)

read_and_process_data >> read_and_transform_data
