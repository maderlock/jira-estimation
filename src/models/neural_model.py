"""Neural network model implementation."""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import calculate_metrics, get_model_config


class TimeEstimatorNN(nn.Module):
    """Neural network architecture for time estimation."""

    def __init__(self, input_size: int):
        """Initialize the network architecture."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        logger.debug(f"Created neural network with input size: {input_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class NeuralEstimator:
    """Neural network based estimator for time prediction."""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Dict = get_model_config(),
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the model.
        
        Args:
            device: Device to use for training (cuda or cpu)
            config: Model configuration
            logger: Optional logger instance
        """
        self.device = device
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.model: Optional[TimeEstimatorNN] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.logger.info(f"Initialized NeuralEstimator using device: {self.device}")
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

    def _create_data_loader(
        self, X: np.ndarray, y: np.ndarray, batch_size: Optional[int] = None
    ) -> DataLoader:
        """Create a PyTorch DataLoader from numpy arrays."""
        batch_size = batch_size if batch_size is not None else self.config.batch_size
        self.logger.debug(f"Creating DataLoader with batch_size={batch_size}")
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Train the model and return metrics.

        Args:
            X: Feature matrix (embeddings)
            y: Target values (time_spent)
            test_size: Proportion of data for testing
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Training NeuralEstimator")
        
        # Set default values from config
        epochs = epochs if epochs is not None else self.config.epochs
        learning_rate = learning_rate if learning_rate is not None else self.config.learning_rate
        self.logger.debug(f"Training parameters - epochs: {epochs}, learning_rate: {learning_rate}")
        
        # Prepare train/test split if not done already
        if self.X_train is None:
            self.prepare_data(X, y, test_size)

        # Initialize model if not done already
        if self.model is None:
            self.model = TimeEstimatorNN(input_size=X.shape[1]).to(self.device)
            self.logger.info(f"Created new model with input size: {X.shape[1]}")

        # Create data loaders
        train_loader = self._create_data_loader(self.X_train, self.y_train, batch_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        self.logger.info("Starting training loop")
        for epoch in tqdm(range(epochs), desc="Training"):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            if (epoch + 1) % 10 == 0:
                self.logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Evaluation
        self.logger.info("Evaluating model")
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
            y_pred = self.model(X_test_tensor).cpu().numpy()

        metrics = calculate_metrics(self.y_test, y_pred)
        self.logger.info(f"Model performance: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        self.logger.debug(f"Making predictions for {len(X)} samples")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def save(self, path: Path) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        self.logger.info(f"Saving model to {path}")
        torch.save(self.model.state_dict(), path)
