"""Local file system-based feature store implementation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from .store import FeatureDefinition, FeatureStore

logger = logging.getLogger(__name__)


class LocalFeatureStore(FeatureStore):
    """Feature store implementation using local file system storage."""

    def __init__(self, name: str, base_path: str = "feature_store"):
        self.base_path = Path(base_path)
        super().__init__(name)

    def _initialize_store(self) -> None:
        """Initialize the feature store directory structure."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.base_path / "metadata"
        self.data_path = self.base_path / "data"
        self.metadata_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)

        # Load existing feature definitions
        for metadata_file in self.metadata_path.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    feature_def = FeatureDefinition(**json.load(f))
                    self.feature_definitions[feature_def.name] = feature_def
            except Exception as e:
                logger.error(f"Failed to load feature definition from {metadata_file}: {e}")

    def register_feature(self, feature_def: FeatureDefinition) -> None:
        """Register a new feature definition."""
        # Update timestamps
        now = datetime.now()
        if feature_def.name in self.feature_definitions:
            feature_def.created_at = self.feature_definitions[feature_def.name].created_at
        else:
            feature_def.created_at = now
        feature_def.updated_at = now

        # Save metadata
        metadata_file = self.metadata_path / f"{feature_def.name}.json"
        with open(metadata_file, "w") as f:
            json.dump(feature_def.dict(), f, indent=2, default=str)

        self.feature_definitions[feature_def.name] = feature_def
        logger.info(f"Registered feature: {feature_def.name}")

    def get_feature(
        self,
        feature_name: str,
        entity_ids: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve feature values for given entities and time range."""
        if feature_name not in self.feature_definitions:
            raise ValueError(f"Feature {feature_name} not found")

        data_file = self.data_path / f"{feature_name}.parquet"
        if not data_file.exists():
            raise FileNotFoundError(f"Feature data for {feature_name} not found")

        df = pd.read_parquet(data_file)

        # Filter by entity IDs
        if entity_ids:
            df = df[df.index.isin(entity_ids)]

        # Filter by time range
        if start_time or end_time:
            time_col = self.feature_definitions[feature_name].parameters.get("time_column")
            if time_col and time_col in df.columns:
                if start_time:
                    df = df[df[time_col] >= start_time]
                if end_time:
                    df = df[df[time_col] <= end_time]

        return df

    def compute_feature(self, feature_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """Compute feature values for given data."""
        if feature_name not in self.feature_definitions:
            raise ValueError(f"Feature {feature_name} not found")

        feature_def = self.feature_definitions[feature_name]
        # Here you would implement the actual feature computation logic
        # based on the feature definition's computation_function and parameters
        # For now, we'll just return the input data
        return data

    def save_feature(self, feature_name: str, data: pd.DataFrame) -> None:
        """Save computed feature values to storage."""
        if feature_name not in self.feature_definitions:
            raise ValueError(f"Feature {feature_name} not found")

        data_file = self.data_path / f"{feature_name}.parquet"
        data.to_parquet(data_file)
        logger.info(f"Saved feature data for {feature_name}")
