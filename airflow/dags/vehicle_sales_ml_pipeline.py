import os
import uuid
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.dummy_operator import DummyOperator

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# Correlation id for training job (this can be also found on MLFLow tracking)
correlation_id = uuid.uuid4()


def get_training_config():
    """Get the training configuration."""
    config_path = os.path.join(PROJECT_ROOT, "examples", "training_config.yaml")
    return config_path


def get_data_path():
    """Get the input data path."""
    return os.path.join(PROJECT_ROOT, "data", "china_vehicle_sales_data.csv")


def get_model_output_path():
    """Get the model output path."""
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"model_{correlation_id}.pkl")


with DAG(
    dag_id="vehicle_sales_ml_pipeline",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["vehicle_sales", "ml_pipeline"],
) as dag:

    start = DummyOperator(task_id="start")

    # Data preprocessing task
    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        python examples/run_train.py \
            --input_data_path {get_data_path()} \
            --config_file_path {get_training_config()} \
            --saved_model_path {get_model_output_path()}
        """,
    )

    # Model training with MLflow tracking
    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        python examples/run_train_mlflow.py \
            --input_data_path {get_data_path()} \
            --config_file_path {get_training_config()} \
            --saved_model_path {get_model_output_path()}
        """,
    )

    # Model evaluation and metadata generation
    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        python examples/run_train.py \
            --input_data_path {get_data_path()} \
            --config_file_path {get_training_config()} \
            --saved_model_path {get_model_output_path()} \
            --evaluate_only
        """,
    )

    complete = DummyOperator(task_id="complete", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define the DAG structure
    start >> preprocess_data >> train_model >> evaluate_model >> complete
