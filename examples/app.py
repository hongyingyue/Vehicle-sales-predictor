import json
import logging
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with proper metadata
app = FastAPI(
    title="Vehicle Sales Prediction API",
    description="API for predicting vehicle sales volume using XGBoost time series model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SalesRequest(BaseModel):
    province_id: str = Field(..., description="Province ID", min_length=1, max_length=10)
    model_name: str = Field(..., description="Model name", min_length=1, max_length=50)
    date: str = Field(..., description="Date in YYYYMM format")
    historical_sales: List[float] = Field(..., description="Historical sales volumes (at least 12 months)")
    province: str = Field(..., description="Province name")
    body_type: str = Field(..., description="Vehicle body type")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, "%Y%m")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYYMM format")

    @field_validator("historical_sales")
    @classmethod
    def validate_historical_sales(cls, v):
        if len(v) < 12:
            raise ValueError("At least 12 months of historical sales data is required")
        if any(s < 0 for s in v):
            raise ValueError("Sales values cannot be negative")
        return v

    class Config:
        schema_extra = {
            "example": {
                "province_id": "P001",
                "model_name": "ModelA",
                "date": "202312",
                "historical_sales": [120, 135, 142, 150, 138, 145, 160, 152, 148, 155, 165, 158],
                "province": "Beijing",
                "body_type": "SUV",
            }
        }


class SalesResponse(BaseModel):
    prediction: float = Field(..., description="Predicted sales volume")
    confidence_interval: Dict[str, float] = Field(..., description="Confidence interval for the prediction")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance for this prediction")
    model_version: str = Field(..., description="Version of the model used for prediction")
    prediction_timestamp: str = Field(..., description="Timestamp of the prediction")


class ModelMetadata(BaseModel):
    version: str
    last_updated: str
    feature_columns: List[str]
    model_path: str


class SalesPredictor:
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the sales predictor model.

        Args:
            model_path: Path to the saved XGBoost model
        """
        self.model_path = model_path or os.getenv("MODEL_PATH", "models/xgb_sales_model.pkl")
        self.model = None
        self.feature_columns = None
        self.model_metadata = None
        self.load_model()

    def load_model(self):
        """Load the XGBoost model and feature configuration"""
        try:
            # Load model metadata
            metadata_path = os.path.join(os.path.dirname(self.model_path), "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)
                    self.feature_columns = self.model_metadata.get("feature_columns", [])
                    logger.info(f"Loaded model metadata: version {self.model_metadata.get('version')}")
            else:
                raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

            # Load XGBoost model
            model_path = self.model_metadata.get("model_path", None)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")

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
        # Add lagging features for salesVolume (lags 1-12)
        for col in ["salesVolume"]:
            for lag in range(1, 13):  # Lag 1 to 12
                lag_col = f"{col}_lag{lag}"
                data[lag_col] = data.groupby(["provinceId", "model"])[col].shift(lag)

        # Add lagging features for popularity (lag 1)
        for col in ["popularity"]:
            for lag in range(1, 2):  # Only lag 1 for popularity
                lag_col = f"{col}_lag{lag}"
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
        try:
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
                    "popularity": [0] * len(request.historical_sales),  # Add popularity column with default value
                    "province": request.province,
                    "bodyType": request.body_type,
                }
            )

            # Add prediction point
            prediction_point = pd.DataFrame(
                {
                    "Date": [dates[-1].strftime("%Y%m")],
                    "provinceId": request.province_id,
                    "model": request.model_name,
                    "salesVolume": [np.nan],  # This is what we want to predict
                    "popularity": [0],  # Add popularity for prediction point
                    "province": request.province,
                    "bodyType": request.body_type,
                }
            )

            # Combine historical data and prediction point
            full_data = pd.concat([data, prediction_point], ignore_index=True)

            # Add features
            full_data = self.add_lagging_features(full_data)

            # Convert categorical features to the correct type
            categorical_features = ["province", "model", "bodyType"]
            for feature in categorical_features:
                if feature in full_data.columns:
                    full_data[feature] = full_data[feature].astype("category")

            return full_data

        except Exception as e:
            logger.error(f"Error preparing input data: {str(e)}")
            raise ValueError(f"Error preparing input data: {str(e)}")

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

            # Calculate confidence interval using model's prediction intervals if available
            # Otherwise use a simple heuristic
            if hasattr(self.model, "predict_proba"):
                # If model supports prediction intervals, use them
                pred_std = np.std(self.model.predict_proba(X_pred))
                confidence_interval = {
                    "lower_bound": max(0, prediction - 1.96 * pred_std),
                    "upper_bound": prediction + 1.96 * pred_std,
                }
            else:
                # Simple confidence interval estimation
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
async def predict(request: SalesRequest, background_tasks: BackgroundTasks, model: SalesPredictor = Depends(get_model)):
    """
    Predict sales volume for the next time step.

    Args:
        request: SalesRequest object with input data
        background_tasks: FastAPI background tasks
        model: SalesPredictor instance (injected)

    Returns:
        SalesResponse with prediction and additional information
    """
    try:
        prediction, confidence_interval, feature_importance = model.predict(request)

        # Log prediction in background
        background_tasks.add_task(
            logger.info,
            f"Prediction made for province_id={request.province_id}, "
            f"model_name={request.model_name}, date={request.date}",
        )

        return SalesResponse(
            prediction=prediction,
            confidence_interval=confidence_interval,
            feature_importance=feature_importance,
            model_version=model.model_metadata.get("version", "unknown"),
            prediction_timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error processing prediction request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        if model.model is None:
            raise RuntimeError("Model not loaded")

        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": model.model_metadata.get("version", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Return model information"""
    try:
        return {
            "model_path": model.model_path,
            "feature_columns": model.feature_columns,
            "model_loaded": model.model is not None,
            "model_version": model.model_metadata.get("version", "unknown"),
            "last_updated": model.model_metadata.get("last_updated", "unknown"),
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Start server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info", workers=4)
