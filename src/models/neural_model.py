"""Neural network model implementation."""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.utils import calculate_metrics


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


class NeuralEstimator:
    """Neural network model for time estimation."""

    def __init__(self):
        """Initialize the model."""
        self.model: Optional[SimpleNeuralNetwork] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict[str, float]:
        """
        Train the model and return metrics.

        Args:
            X: Feature matrix (embeddings)
            y: Target values (duration_hours)
            test_size: Proportion of data to use for testing
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer

        Returns:
            Dictionary containing evaluation metrics
        """
        # Prepare train/test split if not done already
        if self.X_train is None:
            self.prepare_data(X, y, test_size)

        # Initialize model if not done already
        if self.model is None:
            self.model = SimpleNeuralNetwork(self.X_train.shape[1]).to(self.device)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train).reshape(-1, 1).to(self.device)
        X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(self.y_test).reshape(-1, 1).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i + batch_size]
                batch_y = y_train_tensor[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test_tensor).cpu().numpy().flatten()
            
        return calculate_metrics(self.y_test, y_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.model(X_tensor).cpu().numpy().flatten()

    def save(self, path: Path) -> None:
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)
