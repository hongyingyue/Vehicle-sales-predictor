# Examples
The data is processed from open source sales data.

## data version
```shell
uv pip install dvc
dvc init
dvc remote add -d saleremote ssh://local_server_path
```


## Train experiments
```shell
python run_train.py
```

## MLFlow
```python
from vehicle_ml.utils.mlflow_utils import get_or_create_experiment

experiment_id = get_or_create_experiment("sales_xgboost_experiment")

with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_artifact("model_weight.pkl")
    mlflow.log_artifact("feature_columns.json")
    mlflow.log_artifact("important.log")
    mlflow.set_tag("sales_xgb_pipeline")

mlflow.end_run()
```


## Server
```shell
python app.py
```

test

```shell
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "province_id": "P001",
    "model_name": "ModelA",
    "date": "202312",
    "historical_sales": [120, 135, 142, 150, 138, 145, 160, 152, 148, 155, 165, 158],
    "province": "Beijing",
    "body_type": "SUV"
  }'
```

```text
{"prediction":146.52386474609375,"confidence_interval":{"lower_bound":131.8714782714844,"upper_bound":161.17625122070314},"feature_importance":{"province":0.008583756163716316,"model":0.012458096258342266,"bodyType":0.002620626939460635,"salesVolume_lag1":0.7366300821304321,"salesVolume_lag2":0.13781027495861053,"salesVolume_lag3":0.029291000217199326,"salesVolume_lag4":0.011792780831456184,"salesVolume_lag5":0.003110651159659028,"salesVolume_lag6":0.0030429470352828503,"salesVolume_lag7":0.003805882763117552,"salesVolume_lag8":0.005683655850589275,"salesVolume_lag9":0.003304564394056797,"salesVolume_lag10":0.00472667720168829,"salesVolume_lag11":0.009785151109099388,"salesVolume_lag12":0.018264394253492355,"popularity_lag1":0.009089479222893715},"model_version":"baseline-001","prediction_timestamp":"2025-05-12T20:53:51.154189"}%
```

deploy in docker
```
docker build -t vehicle-sales-api -f Dockerfile.inference .
docker run -d -p 8000:8000 --name vehicle-api vehicle-sales-api
```


deploy in AWS sagermaker
```
```

## Feature
```python
from vehicle_ml.feature import LocalFeatureStore, FeatureDefinition

# Create a feature store instance
store = LocalFeatureStore("my_features")

# Define a feature
feature_def = FeatureDefinition(
    name="sales_lag_7d",
    description="7-day lagged sales feature",
    feature_type="numerical",
    computation_function="add_lagging_feature",
    parameters={
        "groupby_column": "store_id",
        "value_columns": ["sales"],
        "lags": [7]
    },
    created_at=datetime.now(),
    updated_at=datetime.now(),
    version="1.0.0",
    tags=["sales", "time_series"]
)

# Register the feature
store.register_feature(feature_def)

# Compute and save feature values
data = pd.DataFrame(...)  # Your input data
computed_data = store.compute_feature("sales_lag_7d", data)
store.save_feature("sales_lag_7d", computed_data)

# Retrieve feature values
feature_data = store.get_feature(
    "sales_lag_7d",
    entity_ids=["store_1", "store_2"],
    start_time=datetime(2023, 1, 1),
    end_time=datetime(2023, 12, 31)
)
```
