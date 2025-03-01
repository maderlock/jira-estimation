"""Linear regression model implementation."""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split

from src.utils import calculate_metrics


class LinearEstimator:
    """Linear regression model for time estimation."""

    def __init__(self):
        """Initialize the model."""
        self.model = LinearRegression()
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def prepare_data(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.

        Args:
            X: Feature matrix (embeddings)
            y: Target values (duration_hours)
            test_size: Proportion of data to use for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        use_cv: bool = False,
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """
        Train the model and return metrics.

        Args:
            X: Feature matrix (embeddings)
            y: Target values (duration_hours)
            test_size: Proportion of data to use for testing
            use_cv: Whether to use cross-validation
            n_splits: Number of splits for cross-validation

        Returns:
            Dictionary containing evaluation metrics
        """
        if use_cv:
            return self._train_with_cv(X, y, n_splits)
        
        # Prepare train/test split if not done already
        if self.X_train is None:
            self.prepare_data(X, y, test_size)
        
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        return calculate_metrics(self.y_test, y_pred)

    def _train_with_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int) -> Dict[str, float]:
        """Train and evaluate using cross-validation."""
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
            cv=KFold(n_splits=n_splits, shuffle=True, random_state=42),
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
        
        # Train final model on full dataset
        self.model.fit(X, y)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        """Save the trained model."""
        import joblib
        joblib.dump(self.model, path)
