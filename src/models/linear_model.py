"""Linear regression model implementation."""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

from utils import calculate_metrics, get_model_config


class LinearEstimator:
    """Linear regression model for time estimation."""

    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict] = None):
        """
        Initialize the linear model.
        
        Args:
            logger: Optional logger instance
            config: Optional model configuration
        """
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.config = config or get_model_config()
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initialized LinearEstimator")
        self.logger.debug(f"Model configuration: {self.config}")

    def prepare_data(
        self, X: np.ndarray, y: np.ndarray, test_size: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.

        Args:
            X: Feature matrix (embeddings)
            y: Target values (time_spent)
            test_size: Proportion of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size if test_size is not None else self.config.test_size
        self.logger.info(f"Preparing data with test_size={test_size}")
        self.logger.debug(f"Input shapes - X: {X.shape}, y: {y.shape}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.random_seed
        )
        
        self.logger.debug(f"Train shapes - X: {self.X_train.shape}, y: {self.y_train.shape}")
        self.logger.debug(f"Test shapes - X: {self.X_test.shape}, y: {self.y_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None,
        use_cv: bool = False,
        n_splits: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train the model and return metrics.

        Args:
            X: Feature matrix (embeddings)
            y: Target values (time_spent)
            test_size: Proportion of data for testing
            use_cv: Whether to use cross-validation
            n_splits: Number of splits for cross-validation

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Training LinearEstimator")
        self.logger.debug(f"Parameters - use_cv: {use_cv}, n_splits: {n_splits}")
        
        if use_cv:
            n_splits = n_splits if n_splits is not None else self.config.cv_splits
            self.logger.info(f"Using {n_splits}-fold cross-validation")
            return self._train_with_cv(X, y, n_splits)
        
        # Prepare train/test split if not done already
        if self.X_train is None:
            self.prepare_data(X, y, test_size)
        
        self.logger.info("Training model on train set")
        self.model.fit(self.X_train, self.y_train)
        
        y_pred = self.model.predict(self.X_test)
        metrics = calculate_metrics(self.y_test, y_pred)
        self.logger.info(f"Model performance: {metrics}")
        return metrics

    def _train_with_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int) -> Dict[str, float]:
        """Train and evaluate using cross-validation."""
        self.logger.debug("Starting cross-validation")
        
        # Define scoring metrics
        scoring = {
            'r2': 'r2',
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_seed),
            scoring=scoring,
            return_train_score=True
        )
        
        # Convert results to positive values where needed
        metrics = {
            'cv_r2_mean': cv_results['test_r2'].mean(),
            'cv_r2_std': cv_results['test_r2'].std(),
            'cv_mae_mean': -cv_results['test_mae'].mean(),
            'cv_mae_std': cv_results['test_mae'].std(),
            'cv_rmse_mean': -cv_results['test_rmse'].mean(),
            'cv_rmse_std': cv_results['test_rmse'].std(),
        }
        
        self.logger.debug(f"Cross-validation metrics: {metrics}")
        
        # Train final model on full dataset
        self.logger.info("Training final model on full dataset")
        self.model.fit(X, y)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        self.logger.debug(f"Making predictions for {len(X)} samples")
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """Save the trained model."""
        self.logger.info(f"Saving model to {path}")
        import joblib
        joblib.dump(self.model, path)
