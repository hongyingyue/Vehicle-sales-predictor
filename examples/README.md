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
    "historical_sales": [120, 135, 142, 150, 138, 145, 160, 152, 148, 155, 165, 158]
  }'
```
