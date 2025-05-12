import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import timer
from .logger import logger

logger = logging.getLogger(__name__)


class Trainer:
    """A flexible trainer for GBDT methods (XGBoost, LightGBM, CatBoost).

    This trainer provides a unified interface for training and evaluating GBDT models
    with support for cross-validation, early stopping, and model persistence.

    Attributes:
        model: The GBDT model instance (XGBoost, LightGBM, or CatBoost)
        model_type: Type of the model (e.g., "LGBMClassifier", "XGBClassifier")
        best_iteration: Best iteration number from training
        feature_names: Names of features used in training
        metrics: Dictionary to store evaluation metrics

    Example:
        >>> from lightgbm import LGBMClassifier
        >>> model = Trainer(LGBMClassifier())
        >>> model.train(X_train, y_train, X_valid, y_valid)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None, random_state: int = 42):
        """Initialize the trainer.

        Args:
            model: GBDT model instance (XGBoost, LightGBM, or CatBoost)
            feature_names: Optional list of feature names
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.model_type = type(model).__name__
        self.feature_names = feature_names
        self.random_state = random_state
        self.best_iteration = None
        self.metrics = {}

        if not any(self.model_type.startswith(prefix) for prefix in ["LGBM", "XGB", "CatBoost"]):
            raise ValueError("Model must be an instance of XGBoost, LightGBM, or CatBoost")

    @timer("trainer train")
    def train(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        x_valid: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_valid: Optional[Union[np.ndarray, pd.Series]] = None,
        categorical_feature: Optional[List[str]] = None,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Train the model with optional validation set.

        Args:
            x_train: Training features
            y_train: Training labels
            x_valid: Optional validation features
            y_valid: Optional validation labels
            categorical_feature: List of categorical feature names
            fit_params: Additional parameters for model fitting

        Returns:
            Dictionary containing training metrics
        """
        self.input_shape = x_train.shape
        fit_params = fit_params or {}

        logger.info(f"Training {self.model_type} model...")

        try:
            if self.model_type.startswith("LGBM"):
                self._train_lightgbm(x_train, y_train, x_valid, y_valid, categorical_feature, fit_params)
            elif self.model_type.startswith("XGB"):
                self._train_xgboost(x_train, y_train, x_valid, y_valid, categorical_feature, fit_params)
            elif self.model_type.startswith("CatBoost"):
                self._train_catboost(x_train, y_train, x_valid, y_valid, categorical_feature, fit_params)

            logger.info(f"Training completed. Best iteration: {self.best_iteration}")

            return self.metrics

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def _train_lightgbm(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        x_valid: Optional[Union[np.ndarray, pd.DataFrame]],
        y_valid: Optional[Union[np.ndarray, pd.Series]],
        categorical_feature: Optional[List[str]],
        fit_params: Dict[str, Any],
    ):
        """Train LightGBM model."""
        eval_sets = [(x_train, y_train)]
        if x_valid is not None:
            eval_sets.append((x_valid, y_valid))

        self.model.fit(
            x_train,
            y_train,
            eval_set=eval_sets,
            categorical_feature=categorical_feature,
            **fit_params,
        )
        self.best_iteration = self.model.best_iteration_

    def _train_xgboost(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        x_valid: Optional[Union[np.ndarray, pd.DataFrame]],
        y_valid: Optional[Union[np.ndarray, pd.Series]],
        categorical_feature: Optional[List[str]],
        fit_params: Dict[str, Any],
    ):
        """Train XGBoost model."""
        eval_sets = [(x_train, y_train)]
        if x_valid is not None:
            eval_sets.append((x_valid, y_valid))

        self.model.fit(
            x_train,
            y_train,
            eval_set=eval_sets,
            **fit_params,
        )
        self.best_iteration = self.model.best_iteration

    def _train_catboost(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        x_valid: Optional[Union[np.ndarray, pd.DataFrame]],
        y_valid: Optional[Union[np.ndarray, pd.Series]],
        categorical_feature: Optional[List[str]],
        fit_params: Dict[str, Any],
    ):
        """Train CatBoost model."""
        from catboost import Pool

        train_data = Pool(data=x_train, label=y_train, cat_features=categorical_feature)
        if x_valid is not None:
            valid_data = Pool(data=x_valid, label=y_valid, cat_features=categorical_feature)
        else:
            valid_data = train_data

        self.model.fit(train_data, eval_set=valid_data, **fit_params)
        self.best_iteration = self.model.get_best_iteration()

    def cross_validate(
        self,
        x: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        n_splits: int = 5,
        stratified: bool = True,
        categorical_feature: Optional[List[str]] = None,
        fit_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation.

        Args:
            x: Features
            y: Labels
            n_splits: Number of cross-validation folds
            stratified: Whether to use stratified K-fold
            categorical_feature: List of categorical feature names
            fit_params: Additional parameters for model fitting

        Returns:
            Dictionary containing cross-validation metrics
        """
        if stratified:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        cv_metrics = {"train_scores": [], "valid_scores": []}

        for fold, (train_idx, valid_idx) in enumerate(kf.split(x, y), 1):
            logger.info(f"Training fold {fold}/{n_splits}")

            x_train, x_valid = x[train_idx], x[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            fold_metrics = self.train(
                x_train,
                y_train,
                x_valid,
                y_valid,
                categorical_feature,
                fit_params,
            )

            cv_metrics["train_scores"].append(fold_metrics.get("train_score", 0))
            cv_metrics["valid_scores"].append(fold_metrics.get("valid_score", 0))

        logger.info(f"Cross-validation results:")
        logger.info(
            f"Train scores: {np.mean(cv_metrics['train_scores']):.4f} ± {np.std(cv_metrics['train_scores']):.4f}"
        )
        logger.info(
            f"Valid scores: {np.mean(cv_metrics['valid_scores']):.4f} ± {np.std(cv_metrics['valid_scores']):.4f}"
        )

        return cv_metrics

    def predict(
        self, x_test: Union[np.ndarray, pd.DataFrame], method: str = "predict", num_iteration: Optional[int] = None
    ) -> np.ndarray:
        """Make predictions.

        Args:
            x_test: Test features
            method: Prediction method ("predict", "predict_proba", or "predict_proba_positive")
            num_iteration: Number of iterations to use for prediction

        Returns:
            Model predictions
        """
        if method == "predict":
            if num_iteration:
                return self.model.predict(x_test, num_iteration=num_iteration)
            return self.model.predict(x_test)
        elif method == "predict_proba":
            if num_iteration:
                return self.model.predict_proba(x_test, num_iteration=num_iteration)
            return self.model.predict_proba(x_test)
        elif method == "predict_proba_positive":
            if num_iteration:
                return self.model.predict_proba(x_test, num_iteration=num_iteration)[:, 1]
            return self.model.predict_proba(x_test)[:, 1]
        else:
            raise ValueError(f"Unsupported predict method: {method}")

    def evaluate(
        self, x: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            x: Features
            y: True labels
            metrics: List of metrics to compute

        Returns:
            Dictionary containing evaluation metrics
        """
        metrics = metrics or ["auc", "accuracy"]
        predictions = self.predict(x)

        results = {}
        for metric in metrics:
            if metric == "auc":
                results["auc"] = roc_auc_score(y, predictions)
            elif metric == "accuracy":
                results["accuracy"] = accuracy_score(y, predictions)
            elif metric == "mse":
                results["mse"] = mean_squared_error(y, predictions)
            elif metric == "r2":
                results["r2"] = r2_score(y, predictions)

        return results

    def get_feature_importance(self, importance_type: str = "gain", top_n: Optional[int] = None) -> Dict[str, float]:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance ("gain" or "split")
            top_n: Number of top features to return

        Returns:
            Dictionary mapping feature names to importance scores
        """

        if self.model_type.startswith("LGBM"):
            importance = self.model.feature_importances_(importance_type=importance_type)
        elif self.model_type.startswith("XGB"):
            importance = self.model.get_booster().get_score(importance_type=importance_type)
        elif self.model_type.startswith("CatBoost"):
            importance = self.model.get_feature_importance()
        else:
            raise ValueError("Unsupported model type for feature importance")

        # Convert to dictionary if not already
        if isinstance(importance, np.ndarray):
            if self.feature_names is None:
                raise ValueError("Feature names not set during initialization")
            importance = dict(zip(self.feature_names, importance))

        # Sort and get top N if specified
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        if top_n is not None:
            sorted_importance = dict(list(sorted_importance.items())[:top_n])

        return sorted_importance

    def plot_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature importance.

        Args:
            importance_type: Type of importance ("gain" or "split")
            top_n: Number of top features to plot
            figsize: Figure size
            save_path: Path to save the plot
        """
        importance = self.get_feature_importance(importance_type, top_n)

        plt.figure(figsize=figsize)
        sns.barplot(x=list(importance.values()), y=list(importance.keys()), palette="viridis")
        plt.title(f"Feature Importance ({importance_type})")
        plt.xlabel("Importance Score")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_model(self, path: Union[str, Path]) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> "Trainer":
        """Load model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Trainer instance with loaded model
        """
        model = joblib.load(path)
        return cls(model)

    def get_model(self) -> Any:
        """Get the underlying model instance."""
        return self.model

    def get_best_iteration(self) -> Optional[int]:
        """Get the best iteration number from training."""
        return self.best_iteration
