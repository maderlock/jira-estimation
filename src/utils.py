"""Shared utilities and helper functions."""
import logging
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary containing R², MAE, and RMSE metrics
    """
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


# Constants
EMBEDDING_MODELS = {
    "ada": "text-embedding-ada-002",
    "gpt4": "gpt-4",  # For future use
}

MAX_TOKENS = {
    "text-embedding-ada-002": 8191,
    "gpt-4": 8192,  # For future use
}
