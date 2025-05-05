import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Sales Volume Prediction API",
    description="API for predicting sales volume using XGBoost time series model",
    version="1.0.0",
)


class SalesRequest(BaseModel):
    province_id: str = Field(..., description="Province ID")
    model_name: str = Field(..., description="Model name")
    date: str = Field(..., description="Date in YYYYMM format")
    historical_sales: List[float] = Field(..., description="Historical sales volumes (at least 12 months)")

    class Config:
        schema_extra = {
            "example": {
                "province_id": "P001",
                "model_name": "ModelA",
                "date": "202312",
                "historical_sales": [120, 135, 142, 150, 138, 145, 160, 152, 148, 155, 165, 158],
            }
        }


class SalesResponse(BaseModel):
    prediction: float = Field(..., description="Predicted sales volume")
    confidence_interval: Dict[str, float] = Field(..., description="Confidence interval for the prediction")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance for this prediction")


class SalesPredictor:
    def __init__(self, model_path: str = None) -> None:
        """
        Initialize the sales predictor model.

        Args:
            model_path: Path to the saved XGBoost model
        """
        self.model_path = model_path or os.getenv("MODEL_PATH", "models/xgb_sales_model.json")
        self.model = None
        self.feature_columns = None
        self.load_model()

    def load_model(self):
        """Load the XGBoost model and feature configuration"""
        try:
            # Load XGBoost model
            self.model = xgb.XGBRegressor()
            self.model.load_model(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")

            # Load feature configuration
            config_path = os.path.join(os.path.dirname(self.model_path), "feature_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                    self.feature_columns = config.get("feature_columns", [])
                    logger.info(f"Loaded {len(self.feature_columns)} feature columns")
            else:
                # Default feature columns if config not found
                self.feature_columns = [f"salesVolume_lag_{i}" for i in range(1, 13)]
                logger.warning(f"Feature config not found at {config_path}, using default feature columns")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def add_lagging_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagging features to the data.

        Args:
            data: DataFrame with sales data

        Returns:
            DataFrame with added lagging features
        """
        # Group by province_id and model_name
        for col in ["salesVolume"]:
            for lag in range(1, 13):  # Lag 1 to 12
                lag_col = f"{col}_lag_{lag}"
                data[lag_col] = data.groupby(["provinceId", "model"])[col].shift(lag)
        return data

    def prepare_input_data(self, request: SalesRequest) -> pd.DataFrame:
        """
        Prepare input data for prediction.

        Args:
            request: SalesRequest object

        Returns:
            DataFrame ready for prediction
        """
        # Create a DataFrame from historical data
        dates = pd.date_range(
            end=pd.to_datetime(request.date, format="%Y%m"),
            periods=len(request.historical_sales) + 1,  # +1 for the prediction point
            freq="M",
        )

        # Create base dataframe with historical data
        data = pd.DataFrame(
            {
                "Date": [d.strftime("%Y%m") for d in dates[:-1]],  # Exclude the prediction date
                "provinceId": request.province_id,
                "model": request.model_name,
                "salesVolume": request.historical_sales,
            }
        )

        # Add prediction point
        prediction_point = pd.DataFrame(
            {
                "Date": [dates[-1].strftime("%Y%m")],
                "provinceId": request.province_id,
                "model": request.model_name,
                "salesVolume": [np.nan],  # This is what we want to predict
            }
        )

        # Combine historical data and prediction point
        full_data = pd.concat([data, prediction_point], ignore_index=True)

        # Add features
        full_data = self.add_lagging_features(full_data)

        return full_data

    def predict(self, request: SalesRequest) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Make a sales prediction.

        Args:
            request: SalesRequest object with input data

        Returns:
            Tuple of (prediction, confidence_interval, feature_importance)
        """
        try:
            # Prepare input data
            input_data = self.prepare_input_data(request)

            # Get features for prediction (last row)
            X_pred = input_data.iloc[-1:][self.feature_columns]

            # Check for missing values
            if X_pred.isna().any().any():
                missing_cols = X_pred.columns[X_pred.isna().any()].tolist()
                raise ValueError(f"Missing values in prediction features: {missing_cols}")

            # Make prediction
            prediction = float(self.model.predict(X_pred)[0])

            # Calculate feature importance
            feature_importance = {}
            if hasattr(self.model, "feature_importances_"):
                for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
                    feature_importance[feature] = float(importance)

            # Simple confidence interval estimation (could be more sophisticated)
            confidence_interval = {
                "lower_bound": max(0, prediction * 0.9),  # Assuming 10% lower bound
                "upper_bound": prediction * 1.1,  # Assuming 10% upper bound
            }

            return prediction, confidence_interval, feature_importance

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Initialize model
model = SalesPredictor()


def get_model():
    """Dependency to get model instance"""
    return model


@app.post("/predict", response_model=SalesResponse)
def predict(request: SalesRequest, model=Depends(get_model)):
    """
    Predict sales volume for the next time step.

    Args:
        request: SalesRequest object with input data
        model: SalesPredictor instance (injected)

    Returns:
        SalesResponse with prediction and additional information
    """
    prediction, confidence_interval, feature_importance = model.predict(request)

    return SalesResponse(
        prediction=prediction, confidence_interval=confidence_interval, feature_importance=feature_importance
    )


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    """Return model information"""
    return {
        "model_path": model.model_path,
        "feature_columns": model.feature_columns,
        "model_loaded": model.model is not None,
    }


if __name__ == "__main__":
    import uvicorn

    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
