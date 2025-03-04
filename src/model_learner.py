"""Model learner implementation."""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

from utils import calculate_metrics, get_model_config

class ModelLearner:
    def __init__(
            self,
            model: RegressorMixin,
            logger: Optional[logging.Logger] = None,
            config: Optional[Dict] = None
        ):
        """
        Initialize the learner model.
        
        Args:
            model: Scikit-learn regression model
            logger: Optional logger instance
            config: Optional model configuration
        """
        self.model = model
        self.scaler = StandardScaler()
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.config = config or get_model_config()
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initialized ModelLearner")
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

        # Log original time distribution
        self.logger.info(
            f"Original time distribution (hours) - "
            f"min: {y.min():.2f}, max: {y.max():.2f}, "
            f"mean: {y.mean():.2f}, median: {np.median(y):.2f}"
        )

        self.logger.debug(f"Input shapes - X: {X.shape}, y: {y.shape}")

        # Filter invalid times
        valid_mask = (y > 0) & (y <= 1000)  # Times must be positive and <= 1000 hours
        if not np.any(valid_mask):
            raise ValueError("No valid training examples after filtering")
            
        X = X[valid_mask]
        y = y[valid_mask]

        indices = np.arange(len(X))
        self.X_train, self.X_test, self.y_train, self.y_test, self._train_indices, self._test_indices = train_test_split(
            X, y, indices, test_size=test_size, random_state=self.config.random_seed
        )

        # Fit and transform the training data
        self.X_train = self.scaler.fit_transform(self.X_train)
        # Transform test data using training statistics
        self.X_test = self.scaler.transform(self.X_test)
        
        self.logger.debug(
            f"Train time stats - "
            f"min: {self.y_train.min():.2f}, max: {self.y_train.max():.2f}, "
            f"mean: {self.y_train.mean():.2f}, median: {np.median(self.y_train):.2f}"
        )
        self.logger.debug(
            f"Test time stats - "
            f"min: {self.y_test.min():.2f}, max: {self.y_test.max():.2f}, "
            f"mean: {self.y_test.mean():.2f}, median: {np.median(self.y_test):.2f}"
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
        self.logger.info("Training model")
        self.logger.debug(f"Parameters - use_cv: {use_cv}, n_splits: {n_splits}")

        if use_cv:
            n_splits = n_splits if n_splits is not None else self.config.cv_splits
            self.logger.info(f"Using {n_splits}-fold cross-validation")
            return self._train_with_cv(X, y, n_splits)

        return self._train_with_single_split(X, y, test_size)

    def _train_with_single_split(
        self, X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None
    ) -> Dict[str, float]:
        """Train and evaluate using a single train/test split.

        Args:
            X: Feature matrix (embeddings)
            y: Target values (time_spent)
            test_size: Proportion of data for testing

        Returns:
            Dictionary containing evaluation metrics
        """
        # Prepare train/test split if not done already
        if self.X_train is None:
            self.prepare_data(X, y, test_size)

        self.logger.info("Training model on train set")
        self.model.fit(self.X_train, self.y_train)

        self.logger.info("Evaluating model on test set")
        y_pred = self.model.predict(self.X_test)
        metrics = calculate_metrics(self.y_test, y_pred)
        self.logger.info(f"Model performance: {metrics}")
        return metrics

    def _train_with_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int) -> Dict[str, float]:
        """Train and evaluate using cross-validation."""
        self.logger.debug("Starting cross-validation")

        # First prepare data to validate and filter times
        self.prepare_data(X, y, test_size=0.2)  # test_size doesn't matter here, just for validation
        
        # Get the filtered data
        X = np.vstack((self.X_train, self.X_test))  # Combine back train/test after scaling
        y = np.concatenate((self.y_train, self.y_test))
        
        # Define scoring metrics
        scoring = {
            'r2': 'r2',
            'mae': 'neg_mean_absolute_error',
            'rmse': 'neg_root_mean_squared_error'
        }

        # Create a pipeline that includes scaling
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.model)
        ])

        # Perform cross-validation with scaling in each fold
        cv_results = cross_validate(
            pipeline,
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
        pipeline.fit(X, y)
        self.model = pipeline.named_steps['model']  # Keep the trained model
        self.scaler = pipeline.named_steps['scaler']  # Keep the final scaler
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return self.model.predict(self.scaler.transform(X))

    def show_examples(
        self,
        titles: List[str],
        descriptions: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_examples: int = 5
    ) -> None:
        """
        Show example predictions.

        Args:
            titles: List of ticket titles
            descriptions: List of ticket descriptions
            y_true: True time values (in hours)
            y_pred: Predicted time values (in hours)
            num_examples: Number of examples to show
        """
        if len(y_true) == 0:
            self.logger.warning("No examples to show")
            return
            
        num_examples = min(num_examples, len(y_true))
        indices = np.random.choice(len(y_true), num_examples, replace=False)
        
        def format_time(hours: float) -> str:
            """Format time in hours to a readable string."""
            if hours < 1:
                return f"{int(hours * 60)}m"
            else:
                h = int(hours)
                m = int((hours - h) * 60)
                return f"{h}h{m:02d}m"
        
        for i in indices:
            title = titles[i] if i < len(titles) else "N/A"
            description = descriptions[i] if i < len(descriptions) else "N/A"
            true_time = y_true[i]
            pred_time = y_pred[i]
            error = abs(true_time - pred_time)
            error_percent = (error / true_time) * 100 if true_time > 0 else float('inf')
            
            self.logger.info(f"\nTicket: {title}")
            self.logger.info(f"Description: {description[:200]}...")
            self.logger.info(f"True time: {format_time(true_time)} ({true_time:.2f} hours)")
            self.logger.info(f"Predicted time: {format_time(pred_time)} ({pred_time:.2f} hours)")
            self.logger.info(f"Error: {format_time(error)} ({error:.2f} hours, {error_percent:.1f}%)")

    def save(self, path: Path) -> None:
        """Save the trained model and scaler to disk.
        
        Args:
            path: Path where to save the model and scaler
        """
        import joblib
        self.logger.info(f"Saving model and scaler to {path}")
        # Save both model and scaler to maintain preprocessing parameters
        joblib.dump((self.model, self.scaler), path)

    def load(self, path: Path) -> None:
        """Load a trained model and scaler from disk.
        
        Args:
            path: Path to the saved model and scaler
        """
        import joblib
        self.logger.info(f"Loading model and scaler from {path}")
        # Load both model and scaler to ensure consistent preprocessing
        self.model, self.scaler = joblib.load(path)

    def get_train_test_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the train/test split data and indices.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, train_indices, test_indices)
        """
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Data has not been prepared yet. Call prepare_data first.")
            
        return self.X_train, self.X_test, self.y_train, self.y_test, self._train_indices, self._test_indices
