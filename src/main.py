"""Main script for JIRA ticket time estimation."""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_fetcher import JiraDataFetcher
from src.models.linear_model import LinearEstimator
from src.models.neural_model import NeuralEstimator
from src.text_processor import TextProcessor
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="JIRA ticket time estimation")
    # Data fetching arguments
    parser.add_argument("--project-keys", nargs="+", help="JIRA project keys")
    parser.add_argument("--exclude-labels", nargs="+", help="Labels to exclude")
    parser.add_argument("--max-results", type=int, default=1000, help="Maximum number of tickets")
    parser.add_argument("--include-subtasks", action="store_true", help="Include subtasks")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached JIRA data")
    parser.add_argument("--no-cache-update", action="store_true", help="Don't update cache with new tickets")
    
    # Model arguments
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--model-type", choices=["linear", "neural"], default="linear")
    parser.add_argument("--use-cv", action="store_true", help="Use cross-validation (linear only)")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (neural only)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (neural only)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate (neural only)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    setup_logging(args.log_level)
    
    # Fetch data
    fetcher = JiraDataFetcher()
    df = fetcher.fetch_completed_issues(
        project_keys=args.project_keys,
        exclude_labels=args.exclude_labels,
        max_results=args.max_results,
        include_subtasks=args.include_subtasks,
        use_cache=not args.no_cache,
        update_cache=not args.no_cache_update,
    )
    
    if df.empty:
        logging.error("No data fetched. Exiting.")
        return
        
    logging.info(f"Fetched {len(df)} tickets")
    
    # Process text and generate embeddings
    processor = TextProcessor()
    embeddings = processor.process_batch([
        f"{row.summary}\n{row.description}" for _, row in df.iterrows()
    ])
    
    # Prepare target variable
    y = df.duration_hours.values
    
    # Train model
    if args.model_type == "linear":
        model = LinearEstimator()
        metrics = model.train(
            embeddings,
            y,
            test_size=args.test_size,
            use_cv=args.use_cv,
            n_splits=args.cv_splits,
        )
    else:
        model = NeuralEstimator()
        metrics = model.train(
            embeddings,
            y,
            test_size=args.test_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    
    # Log results
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Save model
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / f"{args.model_type}_model"
    model_path = model_path.with_suffix(".pkl" if args.model_type == "linear" else ".pt")
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
