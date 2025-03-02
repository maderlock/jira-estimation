"""Shared utilities and helper functions."""
import logging
import os
from pathlib import Path
from typing import Dict, NamedTuple
from dotenv import load_dotenv

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants
DATA_DIR = "data"  # Default value, can be overridden by env

# OpenAI Models
EMBEDDING_MODELS = {
    "ada": "text-embedding-ada-002",
    "davinci": "text-embedding-3-large",
}

MAX_TOKENS = {
    "text-embedding-ada-002": 8191,
    "text-embedding-3-large": 8191,
}

DEFAULT_EMBEDDING_MODEL = "ada"  # Default value, can be overridden by env


def load_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()
    global DATA_DIR, DEFAULT_EMBEDDING_MODEL
    DATA_DIR = os.getenv("DATA_DIR", DATA_DIR)
    DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

class JiraConfig(NamedTuple):
    """JIRA configuration settings."""
    url: str
    email: str
    api_token: str


class ModelConfig(NamedTuple):
    """Model configuration settings."""
    model_type: str
    test_size: float
    cv_splits: int
    random_seed: int
    epochs: int
    batch_size: int
    learning_rate: float


def get_required_env(name: str) -> str:
    """Get a required environment variable."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is required")
    return value


def get_jira_credentials() -> Dict[str, str]:
    """Get JIRA credentials from environment variables."""
    return {
        'url': os.getenv('JIRA_URL'),
        'email': os.getenv('JIRA_EMAIL'),
        'token': os.getenv('JIRA_API_TOKEN'),
    }


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment."""
    return os.getenv('OPENAI_API_KEY')


def get_model_config() -> ModelConfig:
    """Get model configuration from environment variables."""
    return ModelConfig(
        model_type=os.getenv("MODEL_TYPE", "linear"),
        test_size=float(os.getenv("DEFAULT_TEST_SIZE", "0.2")),
        cv_splits=int(os.getenv("DEFAULT_CV_SPLITS", "5")),
        random_seed=int(os.getenv("DEFAULT_RANDOM_SEED", "42")),
        epochs=int(os.getenv("DEFAULT_EPOCHS", "100")),
        batch_size=int(os.getenv("DEFAULT_BATCH_SIZE", "32")),
        learning_rate=float(os.getenv("DEFAULT_LEARNING_RATE", "0.001")),
    )


def setup_logging(level: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Optional override for log level. If not provided,
              uses LOG_LEVEL from environment (defaults to INFO)
              
    Returns:
        Root logger instance
    """
    # Get log level from environment if not overridden
    log_level = level or os.getenv('LOG_LEVEL', 'INFO')
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get and return the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    return logger


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
