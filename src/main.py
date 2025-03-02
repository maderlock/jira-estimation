"""Main script for JIRA ticket time estimation."""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_fetcher import JiraDataFetcher
from models.linear_model import LinearEstimator
from models.neural_model import NeuralEstimator
from text_processor import TextProcessor, OpenAIQuotaExceededError
from utils import get_model_config, setup_logging, load_environment


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    model_config = get_model_config()
    
    parser = argparse.ArgumentParser(description="JIRA ticket time estimation")
    # Data fetching arguments
    parser.add_argument("--project-keys", nargs="+", help="JIRA project keys")
    parser.add_argument("--exclude-labels", nargs="+", help="Labels to exclude")
    parser.add_argument("--max-results", type=int, default=1000, help="Maximum number of tickets")
    parser.add_argument("--include-subtasks", action="store_true", help="Include subtasks")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached JIRA data")
    parser.add_argument("--no-cache-update", action="store_true", help="Don't update cache with new tickets")
    parser.add_argument("--force-update", action="store_true", help="Force full update of cached data")
    
    # Model arguments
    parser.add_argument("--test-size", type=float, default=model_config.test_size, help="Test set size")
    parser.add_argument("--model-type", choices=["linear", "neural"], default="linear")
    parser.add_argument("--use-cv", action="store_true", help="Use cross-validation (linear only)")
    parser.add_argument("--cv-splits", type=int, default=model_config.cv_splits, help="Number of CV splits")
    parser.add_argument("--epochs", type=int, default=model_config.epochs, help="Training epochs (neural only)")
    parser.add_argument("--batch-size", type=int, default=model_config.batch_size, help="Batch size (neural only)")
    parser.add_argument("--learning-rate", type=float, default=model_config.learning_rate, help="Learning rate (neural only)")
    parser.add_argument("--log-level", help="Override default log level from environment")
    return parser.parse_args()


def main():
    """Main function."""
    # Load environment variables first
    load_environment()
    
    args = parse_args()
    setup_logging(args.log_level)  # Will use LOG_LEVEL from env if not overridden
    logger = logging.getLogger(__name__)
    
    logger.info("Starting JIRA ticket time estimation")
    logger.debug(f"Arguments: {args}")

    try:
        # Initialize components
        logger.info("Initializing components")
        data_fetcher = JiraDataFetcher()
        text_processor = TextProcessor()
        model = LinearEstimator() if args.model_type == "linear" else NeuralEstimator()

        # Fetch and process data
        logger.info("Fetching JIRA tickets")
        df = data_fetcher.fetch_tickets(
            project_keys=args.project_keys,
            max_results=args.max_results,
            exclude_labels=args.exclude_labels,
            include_subtasks=args.include_subtasks,
            use_cache=not args.no_cache,
            update_cache=not args.no_cache_update,
            force_update=args.force_update,
        )
        
        if df.empty:
            logger.error("No tickets found matching criteria")
            return 1

        logger.info("Processing ticket text")
        # Combine summary and description for embedding
        texts = df["summary"] + " " + df["description"].fillna("")
        try:
            # Create metadata for caching
            metadata = [
                {"key": row.key, "summary": row.summary}
                for _, row in df.iterrows()
            ]
            X = text_processor.get_embeddings(texts, metadata=metadata)
        except OpenAIQuotaExceededError as e:
            logger.error(str(e))
            logger.error("Unable to continue without embeddings. Please check your OpenAI API quota.")
            return 1
            
        y = df["time_spent"].values

        # Train model
        logger.info(f"Training {args.model_type} model")
        metrics = model.train(
            X,
            y,
            test_size=args.test_size,
            use_cv=args.use_cv,
            n_splits=args.cv_splits,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

        # Save results
        logger.info("Saving results")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        model_file = results_dir / f"{args.model_type}_model"
        model_file = model_file.with_suffix(".pkl" if args.model_type == "linear" else ".pt")
        model.save(model_file)
        logger.info(f"Model saved to {model_file}")

        # Print metrics
        logger.info("Final model performance:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return 0
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
