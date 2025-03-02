"""Module for training and evaluating models."""
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from src.config import MODELS_DIR


class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for regression."""

    def __init__(self, input_size: int):
        """Initialize the network."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layers(x)


class ModelTrainer:
    """Class to handle model training and evaluation."""

    def __init__(self, model_type: str = "linear"):
        """
        Initialize the trainer.

        Args:
            model_type: Type of model to use ('linear' or 'neural').
        """
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.

        Args:
            X: Feature matrix (embeddings)
            y: Target values (time_spent)
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
            y: Target values (time_spent)
            test_size: Proportion of data to use for testing
            use_cv: Whether to use cross-validation (only for linear model)
            n_splits: Number of splits for cross-validation

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model_type == "linear" and use_cv:
            return self._train_linear_cv(X, y, n_splits)
        
        # Prepare train/test split if not done already
        if self.X_train is None:
            self.prepare_data(X, y, test_size)
        
        if self.model_type == "linear":
            return self._train_linear()
        else:
            return self._train_neural()

    def _train_linear_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int) -> Dict[str, float]:
        """Train and evaluate linear regression model using cross-validation."""
        self.model = LinearRegression()
        
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

    def _train_linear(self) -> Dict[str, float]:
        """Train and evaluate linear regression model."""
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        y_pred = self.model.predict(self.X_test)
        return self._calculate_metrics(self.y_test, y_pred)

    def _train_neural(self) -> Dict[str, float]:
        """Train and evaluate neural network model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = SimpleNeuralNetwork(self.X_train.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train).to(device)
        y_train_tensor = torch.FloatTensor(self.y_train).reshape(-1, 1).to(device)
        X_test_tensor = torch.FloatTensor(self.X_test).to(device)

        # Training loop
        epochs = 100
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test_tensor).cpu().numpy().flatten()
            
        return self._calculate_metrics(self.y_test, y_pred)

    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        return {
            "r2_score": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model_type == "linear":
            return self.model.predict(X)
        else:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                return self.model(X_tensor).numpy().flatten()

    def save_model(self, name: str) -> None:
        """Save the trained model."""
        model_path = Path(MODELS_DIR) / f"{name}.pt"
        if self.model_type == "linear":
            torch.save(self.model, model_path)
        else:
            torch.save(self.model.state_dict(), model_path)

    def load_model(self, name: str) -> None:
        """Load a trained model."""
        model_path = Path(MODELS_DIR) / f"{name}.pt"
        if self.model_type == "linear":
            self.model = torch.load(model_path)
        else:
            self.model = SimpleNeuralNetwork(768)  # Assuming OpenAI embedding size
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
