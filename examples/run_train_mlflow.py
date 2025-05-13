import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import warnings
from typing import Dict, List, Optional

import pandas as pd
import xgboost as xgb
import yaml
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

from vehicle_ml import Trainer, logger
from vehicle_ml.data import DataIngester, DatetimeSplitter
from vehicle_ml.feature import add_lagging_feature
from vehicle_ml.metrics.regression import get_mae, get_rmse

os.environ["DO_DEBUG"] = "false"
warnings.filterwarnings("ignore")

# Set MLflow tracking URI (you can change this to your preferred location)
mlflow.set_tracking_uri("file:./mlruns")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=str, nargs="?", default="baseline-001")
    parser.add_argument("config_file_path", type=str, nargs="?", default="./training_config.yaml")
    parser.add_argument(
        "input_data_path", type=str, nargs="?", default="../data/sales_data.csv", help="original data path"
    )
    parser.add_argument("saved_model_path", type=str, nargs="?", default=None)
    return parser.parse_args()


def get_training_config(config_file: str) -> Dict:
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


def prepare_data(input_data_path):
    data_ingester = DataIngester(input_data_path=input_data_path)
    data = data_ingester.ingest()

    if os.getenv("DO_DEBUG", "false") == "true":
        data = data.sample(frac=0.01)
        logger.info(f"[IMPORTANT] Data - sample size: {data.shape}")

    return data


def add_features(data: pd.DataFrame, feature_columns: Optional[List[str]] = None):
    data = add_lagging_feature(
        data,
        groupby_column=["provinceId", "model"],
        value_columns=["salesVolume"],
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        feature_columns=feature_columns,
    )

    data = add_lagging_feature(
        data,
        groupby_column=["provinceId", "model"],
        value_columns=["popularity"],
        lags=[1],
        feature_columns=feature_columns,
    )

    # data = add_rolling_feature(
    #     data=data,
    #     groupby_column=["provinceId", "model"],
    #     value_columns=["salesVolume"],
    #     periods=[3, 6],
    #     agg_funs=["max", "mean", "sum", "min"],
    #     feature_columns=feature_columns
    # )

    # data = add_num_num_feature(
    #     data=data,
    #     num_features=['salesVolume_lag1', 'salesVolume_lag2'],
    #     fun_list=['diff'],
    #     feature_columns=feature_columns
    # )
    return data


def get_metrics(y_true, y_pred):
    mae = get_mae(y_true, y_pred)
    rmse = get_rmse(y_true, y_pred)
    return {"mae": mae, "rmse": rmse}


def run_train(input_data_path, saved_model_path, config, label_column_name="salesVolume"):
    # Start MLflow run
    with mlflow.start_run(run_name=os.path.basename(saved_model_path).replace(".pkl", "")):
        # Log parameters
        mlflow.log_params(config["model_params"])
        mlflow.log_params(config["fit_params"])
        mlflow.log_param("categorical_features", config["categorical_feature"])

        data = prepare_data(input_data_path=input_data_path)
        data = data.sort_values(["Date", "provinceId"])

        feature_columns = config["categorical_feature"].copy()
        data = add_features(data, feature_columns=feature_columns)
        data[config["categorical_feature"]] = data[config["categorical_feature"]].astype(str).astype("category")
        logger.info(
            f"[IMPORTANT] Feature size: {len(feature_columns)}, categorical feature: {len(config['categorical_feature'])}"
        )

        data_splitter = DatetimeSplitter(time_column="Date", test_date=["201711", "201712"])
        train, valid = data_splitter.split(data)

        x_train = train[feature_columns]
        y_train = train[label_column_name]
        x_valid = valid[feature_columns]
        y_valid = valid[label_column_name]
        logger.info(
            f"[IMPORTANT] Train size: {x_train.shape}:{y_train.shape}, Valid size: {x_valid.shape}:{y_valid.shape}"
        )

        # Log dataset info
        mlflow.log_param("train_size", len(x_train))
        mlflow.log_param("valid_size", len(x_valid))
        mlflow.log_param("feature_count", len(feature_columns))

        model = xgb.XGBRegressor(**config["model_params"])
        trainer = Trainer(model)
        trainer.train(x_train, y_train, x_valid, y_valid, fit_params=config["fit_params"])

        # Get predictions and metrics
        valid_pred = trainer.predict(x_valid)
        valid_true = valid[label_column_name]
        metrics = get_metrics(valid_true, valid_pred)
        logger.info(f"[IMPORTANT] Test metrics: {metrics}")

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Log feature importance plot
        importance_plot_path = trainer.plot_feature_importance()
        if importance_plot_path:
            mlflow.log_artifact(importance_plot_path)

        # Log model with MLflow
        signature = infer_signature(x_valid, valid_pred)
        mlflow.xgboost.log_model(
            model,
            "model",
            signature=signature,
            input_example=x_valid.head(),
            registered_model_name="vehicle_sales_predictor",
        )

        # Save validation results
        valid["pred"] = valid_pred
        valid.to_csv("valid_with_pred.csv", index=False)
        mlflow.log_artifact("valid_with_pred.csv")

        # Generate and log model metadata
        metadata = {
            "version": os.path.basename(saved_model_path).replace(".pkl", ""),
            "last_updated": datetime.now().isoformat(),
            "feature_columns": feature_columns,
            "model_path": saved_model_path,
            "metrics": metrics,
            "model_config": {
                "model_params": config.get("model_params", {}),
                "categorical_features": config.get("categorical_feature", []),
            },
        }

        metadata_path = "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        mlflow.log_artifact(metadata_path)


if __name__ == "__main__":
    args = parse_args()
    if args.saved_model_path is None:
        args.saved_model_path = f"./models/{args.exp_id}.pkl"

    logger.info(f"[IMPORTANT] Start Experiment: {args.exp_id}")
    config = get_training_config(args.config_file_path)

    run_train(input_data_path=args.input_data_path, saved_model_path=args.saved_model_path, config=config)
