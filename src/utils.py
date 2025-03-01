"""Shared utilities and helper functions."""
import logging
import os
from pathlib import Path
from typing import Dict, NamedTuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class JiraConfig(NamedTuple):
    """JIRA configuration settings."""
    url: str
    email: str
    api_token: str


class ModelConfig(NamedTuple):
    """Model configuration settings."""
    test_size: float
    cv_splits: int
    epochs: int
    batch_size: int
    learning_rate: float


def get_required_env(name: str) -> str:
    """Get a required environment variable."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is required")
    return value


def get_jira_config() -> JiraConfig:
    """Get JIRA configuration from environment variables."""
    return JiraConfig(
        url=get_required_env("JIRA_URL"),
        email=get_required_env("JIRA_EMAIL"),
        api_token=get_required_env("JIRA_API_TOKEN"),
    )


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment."""
    return get_required_env("OPENAI_API_KEY")


def get_model_config() -> ModelConfig:
    """Get model configuration from environment variables."""
    return ModelConfig(
        test_size=float(os.getenv("DEFAULT_TEST_SIZE", "0.2")),
        cv_splits=int(os.getenv("DEFAULT_CV_SPLITS", "5")),
        random_seed=int(os.getenv("DEFAULT_RANDOM_SEED", "42")),
        epochs=int(os.getenv("DEFAULT_EPOCHS", "100")),
        batch_size=int(os.getenv("DEFAULT_BATCH_SIZE", "32")),
        learning_rate=float(os.getenv("DEFAULT_LEARNING_RATE", "0.001")),
    )


def setup_logging(level: str = None) -> None:
    """Set up logging configuration."""
    level = level or os.getenv("LOG_LEVEL", "INFO")
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


# Data storage
DATA_DIR = os.getenv("DATA_DIR", "data")

# OpenAI configuration
EMBEDDING_MODELS = {
    "ada": "text-embedding-ada-002",
    "gpt4": "gpt-4",  # For future use
}

MAX_TOKENS = {
    "text-embedding-ada-002": 8191,
    "gpt4": 8192,  # For future use
}

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ada")
