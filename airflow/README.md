# Vehicle Sales ML Pipeline with Airflow

This directory contains the Airflow DAG implementation for the Vehicle Sales ML pipeline. The pipeline automates the process of data preprocessing, model training, and evaluation.

## Setup

1. Install Airflow and dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Airflow environment:
```bash
export AIRFLOW_HOME=$(pwd)
airflow db init
```

3. Create an Airflow user:
```bash
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

4. Start the Airflow webserver:
```bash
airflow webserver --port 8080
```

5. In a new terminal, start the Airflow scheduler:
```bash
airflow scheduler
```

## Pipeline Structure

The pipeline consists of the following tasks:

1. **Data Preprocessing**: Prepares the input data for training
2. **Model Training**: Trains the XGBoost model with MLflow tracking
3. **Model Evaluation**: Evaluates the model performance and generates metadata

## Running the Pipeline

1. Access the Airflow web interface at `http://localhost:8080`
2. Log in with the credentials created above
3. Navigate to the "vehicle_sales_ml_pipeline" DAG
4. Click "Trigger DAG" to start the pipeline

## Monitoring

- Use the Airflow web interface to monitor task progress
- Check MLflow for model training metrics and artifacts
- View model metadata in the `models` directory

## Configuration

The pipeline uses the following configuration files:
- `examples/training_config.yaml`: Model training parameters
- `examples/run_train.py`: Training script
- `examples/run_train_mlflow.py`: MLflow-enabled training script

## Output

The pipeline generates:
- Trained model files in the `models` directory
- MLflow tracking data in the `mlruns` directory
- Model evaluation metrics and metadata
