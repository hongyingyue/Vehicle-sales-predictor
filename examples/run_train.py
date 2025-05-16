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

from vehicle_ml import Trainer, logger, timer
from vehicle_ml.data import DataIngester, DatetimeSplitter
from vehicle_ml.feature import (
    feature_registry,
    registry,
    add_lagging_feature,
    add_datetime_feature,
    add_rolling_feature,
    add_num_num_feature,
    add_cat_num_feature,
)
from vehicle_ml.metrics.regression import get_mae, get_rmse

os.environ["DO_DEBUG"] = "false"
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=str, nargs="?", default="baseline-001")
    parser.add_argument("config_file_path", type=str, nargs="?", default="./training_config.yaml")
    parser.add_argument(
        "input_data_path",
        type=str,
        nargs="?",
        default="../data/china_vehicle_sales_data.csv",
        help="original data path",
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
    # Add datetime features
    data = add_datetime_feature(
        data,
        time_column="Date",
        date_type_list=["year", "month", "quarter", "dayofweek"],
        feature_columns=feature_columns,
    )

    # Add lagging features for sales volume
    data = add_lagging_feature(
        data,
        groupby_column=["provinceId", "model"],
        value_columns=["salesVolume"],
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        feature_columns=feature_columns,
    )

    # # Add lagging features for popularity
    # data = add_lagging_feature(
    #     data,
    #     groupby_column=["provinceId", "model"],
    #     value_columns=["popularity"],
    #     lags=[1],
    #     feature_columns=feature_columns,
    # )

    # # Add rolling features for sales volume
    # data = add_rolling_feature(
    #     data,
    #     groupby_column=["provinceId", "model"],
    #     value_columns=["salesVolume"],
    #     periods=[3, 6],
    #     agg_funs=["mean", "std", "max", "min"],
    #     feature_columns=feature_columns
    # )

    # # Add numerical interaction features
    # data = add_num_num_feature(
    #     data,
    #     num_features=['salesVolume_lag1', 'salesVolume_lag2', 'popularity_lag1'],
    #     fun_list=['ratio', 'diff'],
    #     feature_columns=feature_columns
    # )

    # # Add categorical-numerical features
    # data = add_cat_num_feature(
    #     data,
    #     amount_feas=['salesVolume'],
    #     category_feas=['provinceId', 'model'],
    #     fun_list=['mean', 'std', 'max', 'min']
    # )

    return data


def get_metrics(y_true, y_pred):
    mae = get_mae(y_true, y_pred)
    rmse = get_rmse(y_true, y_pred)
    return {"mae": mae, "rmse": rmse}


def generate_model_metadata(
    model_path: str, feature_columns: List[str], metrics: Dict[str, float], config: Dict, exp_id: str
) -> None:
    """
    Generate model metadata file with version information and model details.

    Args:
        model_path: Path to the saved model
        feature_columns: List of feature columns used in the model
        metrics: Dictionary of model metrics
        config: Training configuration
        exp_id: Experiment ID
    """
    metadata = {
        "version": exp_id,
        "last_updated": datetime.now().isoformat(),
        "feature_columns": feature_columns,
        "model_path": model_path,
        "metrics": metrics,
        "model_config": {
            "model_params": config.get("model_params", {}),
            "categorical_features": config.get("categorical_feature", []),
        },
    }

    # Save metadata in the same directory as the model
    metadata_path = os.path.join(os.path.dirname(model_path), "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata saved to {metadata_path}")


@timer("training_pipeline")
def run_train(input_data_path, saved_model_path, config, label_column_name="salesVolume"):
    data = prepare_data(input_data_path=input_data_path)
    data = data.sort_values(["Date", "provinceId"])

    feature_columns = config["categorical_feature"].copy()
    data = add_features(data, feature_columns=feature_columns)
    # data.to_csv("temp.csv", index=False)
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
    logger.info(f"[IMPORTANT] Train size: {x_train.shape}:{y_train.shape}, Valid size: {x_valid.shape}:{y_valid.shape}")

    model = xgb.XGBRegressor(**config["model_params"])
    trainer = Trainer(model)
    trainer.train(x_train, y_train, x_valid, y_valid, fit_params=config["fit_params"])

    # Save model and generate metadata
    trainer.save_model(saved_model_path)
    trainer.plot_feature_importance()

    # Get predictions and metrics
    valid_pred = trainer.predict(x_valid)
    valid_true = valid[label_column_name]
    metrics = get_metrics(valid_true, valid_pred)
    logger.info(f"[IMPORTANT] Test metrics: {metrics}")

    # Generate model metadata
    generate_model_metadata(
        model_path=saved_model_path,
        feature_columns=feature_columns,
        metrics=metrics,
        config=config,
        exp_id=os.path.basename(saved_model_path).replace(".pkl", ""),
    )

    # Save validation results
    valid["pred"] = valid_pred
    valid.to_csv("valid_with_pred.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    if args.saved_model_path is None:
        args.saved_model_path = f"./models/{args.exp_id}.pkl"

    logger.info(f"[IMPORTANT] Start Experiment: {args.exp_id}")
    config = get_training_config(args.config_file_path)

    run_train(input_data_path=args.input_data_path, saved_model_path=args.saved_model_path, config=config)
