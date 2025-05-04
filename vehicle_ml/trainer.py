import logging
from typing import Dict, Optional

import joblib
import numpy as np

from .utils import timer

logger = logging.getLogger(__name__)


class Trainer:
    """A trainer for GBDT methods
    # how to use it:
    model = Trainer(CatBoostClassifier(**CAT_PARAMS))
    model.train(x_train, y_train, x_valid, y_valid, fit_params={})
    """

    def __init__(self, model):
        self.model = model
        self.model_type = type(model).__name__

    @timer("trainer train")
    def train(
        self,
        x_train,
        y_train,
        x_valid=None,
        y_valid=None,
        categorical_feature=None,
        fit_params=None,
    ):
        self.input_shape = x_train.shape

        # LightGBM model
        if self.model_type[:4] == "LGBM":
            if x_valid is not None:
                self.model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_train, y_train), (x_valid, y_valid)],
                    categorical_feature=categorical_feature,
                    **fit_params,
                )
            else:
                self.model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_train, y_train)],
                    categorical_feature=categorical_feature,
                    **fit_params,
                )

            self.best_iteration = self.model.best_iteration_
        elif self.model_type[:3] == "XGB":
            if x_valid is not None:
                self.model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], **fit_params)
            else:
                self.model.fit(x_train, y_train, eval_set=[(x_train, y_train)], **fit_params)
            self.best_iteration = self.model.best_iteration
        elif self.model_type[:8] == "CatBoost":
            from catboost import Pool

            train_data = Pool(data=x_train, label=y_train, cat_features=categorical_feature)
            if x_valid is not None:
                valid_data = Pool(data=x_valid, label=y_valid, cat_features=categorical_feature)
            else:
                valid_data = train_data

            self.model.fit(X=train_data, eval_set=valid_data, **fit_params)
            self.best_iteration = self.model.get_best_iteration()
        else:
            raise ValueError("model type should be in xgboost, lightgbm or catboost")

    def predict(self, x_test, method="predict", num_iteration=None):
        if method == "predict":
            if num_iteration:
                return self.model.predict(x_test, num_iteration=num_iteration)
            else:
                return self.model.predict(x_test)
        elif method == "predict_proba":
            if num_iteration is not None:
                return self.model.predict_proba(x_test, num_iteration=num_iteration)
            else:
                return self.model.predict_proba(x_test)
        elif method == "predict_proba_positive":
            if num_iteration is not None:
                return self.model.predict_proba(x_test, num_iteration=num_iteration)[:, 1]
            else:
                return self.model.predict_proba(x_test)[:, 1]
        else:
            raise ValueError(f"unsupported predict method of {method}")

    def create_feature_importance(self, use_method: str = "gain", importance_params: Dict = None) -> np.ndarray:
        """Create feature importance array.

        Args:
            use_method: Method to calculate feature importance (default: "auto")
            importance_params: Additional parameters for importance calculation (optional)

        Returns:
            Feature importance array
        """
        importance_params = importance_params or {}

        if use_method in ["gain", "split"]:
            if self.model_type[:4] == "LGBM":
                importance = self.model.feature_importances_(importance_type=use_method)
                return importance
            elif self.model_type[:3] == "XGB":
                booster = self.model.get_booster()
                gain_importance = booster.get_score(importance_type="gain")
                split_importance = booster.get_score(importance_type="weight")
                return gain_importance
            else:
                logger.warning(f"{self.model_type} does not have feature_importances_ attribute")
                return np.array([])

        # Additional importance methods could be implemented here
        logger.warning(f"Importance method {use_method} not implemented, using default")
        return np.array([])

    def plot_feature_importance(self, saved_fig_path: str = "feature_importance", top_n: Optional[int] = None) -> None:
        """Plot feature importance.

        Args:
            columns: Feature names (optional)
            figsize: Figure size as (width, height) (optional)
            color_map: Matplotlib colormap name (default: "winter")
            top_n: Number of top features to plot (optional)
        """
        import matplotlib.pyplot as plt

        importance = self.create_feature_importance()
        sorted_importance = sorted(importance.items(), key=lambda item: item[1], reverse=True)

        if top_n is not None and top_n > 0:
            top_features = sorted_importance[:top_n]
        else:
            top_features = sorted_importance

        features, gains = zip(*top_features)

        plt.figure(figsize=(10, 6))
        plt.barh(features, gains, color="skyblue")
        plt.xlabel("Gain")
        plt.title("Feature Importance")
        plt.savefig(saved_fig_path)
        # plt.gca().invert_yaxis()
        # plt.tight_layout()
        # plt.show()

    def get_model(self):
        return self.model

    def save_model(self, model_dir):
        joblib.dump(self.model, model_dir)
        logger.info(f"Saved model in {model_dir}")
        return

    def get_best_iteration(self):
        return self.best_iteration


class OptunaTrainer:
    def __init__(self) -> None:
        pass
