"""Linear regression model implementation."""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

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
        
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)
            
        # Validate that we have valid Y values
        if len(y) == 0:
            raise ValueError("No training examples provided")
        if np.any(y <= 0):
            raise ValueError("Found non-positive time values in training data. All time values must be positive.")
            
        self.logger.debug(f"Y values before split: {y.tolist()}")
        self.logger.debug(f"Y stats before split - min: {y.min()}, max: {y.max()}, mean: {y.mean():.2f}")
        self.logger.debug(f"Input shapes - X: {X.shape}, y: {y.shape}")
        
        indices = np.arange(len(X))
        self.X_train, self.X_test, self.y_train, self.y_test, self._train_indices, self._test_indices = train_test_split(
            X, y, indices, test_size=test_size, random_state=self.config.random_seed
        )
        
        # Fit and transform the training data
        self.X_train = self.scaler.fit_transform(self.X_train)
        # Transform test data using training statistics
        self.X_test = self.scaler.transform(self.X_test)
        
        self.logger.debug(f"Y values after split - train: {self.y_train.tolist()}, test: {self.y_test.tolist()}")
        self.logger.debug(f"Y stats after split - train: min: {self.y_train.min()}, max: {self.y_train.max()}, mean: {self.y_train.mean():.2f}, test: min: {self.y_test.min()}, max: {self.y_test.max()}, mean: {self.y_test.mean():.2f}")
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
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return self.model.predict(self.scaler.transform(X))

    def show_examples(self, titles: List[str], descriptions: List[str], y_true: np.ndarray, y_pred: np.ndarray, n_examples: int = 3) -> None:
        """
        Show best, median and worst predictions.
        
        Args:
            titles: List of ticket titles
            descriptions: List of ticket descriptions
            y_true: True values
            y_pred: Predicted values
            n_examples: Number of examples to show (default 3: best, median, worst)
        """
        # Get absolute errors
        errors = np.abs(y_true - y_pred)
        
        # Get indices for best, median and worst predictions
        best_idx = np.argmin(errors)
        worst_idx = np.argmax(errors)
        median_idx = np.argsort(errors)[len(errors)//2]
        
        example_indices = [best_idx, median_idx, worst_idx]
        labels = ["Best", "Median", "Worst"]
        
        self.logger.info("\nExample Predictions:")
        for idx, label in zip(example_indices, labels):
            # Convert hours to hours and minutes
            true_hours = int(y_true[idx])
            true_mins = int((y_true[idx] - true_hours) * 60)
            pred_hours = int(y_pred[idx])
            pred_mins = int((y_pred[idx] - pred_hours) * 60)
            
            # Get first few lines of description
            desc_preview = "\n".join(descriptions[idx].split("\n")[:3])
            
            self.logger.info(f"\n{label} Prediction:")
            self.logger.info(f"Title: {titles[idx]}")
            self.logger.info(f"Description (first 3 lines):\n{desc_preview}")
            self.logger.info(f"True time: {true_hours}h{true_mins:02d}m")
            self.logger.info(f"Predicted: {pred_hours}h{pred_mins:02d}m")
            self.logger.info(f"Error: {errors[idx]:.2f} hours")

    def save(self, path: Path) -> None:
        """Save the trained model."""
        self.logger.info(f"Saving model to {path}")
        import joblib
        joblib.dump(self.model, path)

    def get_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the train/test split data.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, train_indices, test_indices)
        """
        if not hasattr(self, '_train_indices') or not hasattr(self, '_test_indices'):
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.X_train, self.X_test, self.y_train, self.y_test, self._train_indices, self._test_indices
