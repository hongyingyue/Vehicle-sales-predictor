import logging

from mlflow import MLflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get an existing MLflow experiment by name, or create it if it doesn't exist.

    Parameters:
        experiment_name (str): Name of the MLflow experiment.

    Returns:
        str: The experiment ID.
    """
    client = MLflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            logger.info(f"Found existing experiment '{experiment_name}' (ID: {experiment.experiment_id})")
            return experiment.experiment_id
        else:
            experiment_id = client.create_experiment(experiment_name)
            logger.info(f"Created new experiment '{experiment_name}' (ID: {experiment_id})")
            return experiment_id
    except MlflowException as e:
        logger.error(f"Failed to get or create experiment '{experiment_name}': {e}")
        raise
