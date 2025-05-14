"""Feature store implementation for managing and serving features in production.

This module provides a feature store implementation that handles feature computation,
storage, and serving in a production environment.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FeatureDefinition(BaseModel):
    """Definition of a feature including its metadata and computation logic."""

    name: str
    description: str
    feature_type: str  # e.g., "numerical", "categorical", "datetime"
    computation_function: str
    parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: str
    tags: List[str] = []


class FeatureStore(ABC):
    """Abstract base class for feature store implementations."""

    def __init__(self, name: str):
        self.name = name
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self._initialize_store()

    @abstractmethod
    def _initialize_store(self) -> None:
        """Initialize the feature store."""
        pass

    @abstractmethod
    def register_feature(self, feature_def: FeatureDefinition) -> None:
        """Register a new feature definition."""
        pass

    @abstractmethod
    def get_feature(
        self,
        feature_name: str,
        entity_ids: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Retrieve feature values for given entities and time range."""
        pass

    @abstractmethod
    def compute_feature(self, feature_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """Compute feature values for given data."""
        pass

    @abstractmethod
    def save_feature(self, feature_name: str, data: pd.DataFrame) -> None:
        """Save computed feature values to storage."""
        pass

    def list_features(self) -> List[FeatureDefinition]:
        """List all registered features."""
        return list(self.feature_definitions.values())

    def get_feature_definition(self, feature_name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name."""
        return self.feature_definitions.get(feature_name)

    def delete_feature(self, feature_name: str) -> None:
        """Delete a feature definition and its data."""
        if feature_name in self.feature_definitions:
            del self.feature_definitions[feature_name]
            logger.info(f"Deleted feature: {feature_name}")
