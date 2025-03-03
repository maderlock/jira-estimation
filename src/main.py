"""Main entry point for JIRA ticket time estimation."""
import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data_fetching import JiraDataFetcher, DataCache
from models.linear_model import LinearEstimator
from models.neural_model import NeuralEstimator
from text_processing import TextProcessor
from text_processing.constants import DEFAULT_EMBEDDING_MODEL
from utils import get_model_config, setup_logging, load_environment, get_jira_credentials, get_openai_api_key


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
    parser.add_argument("--force-update", action="store_true", help="Force full update of cached data")
    # Model arguments
    parser.add_argument("--test-size", type=float, default=model_config.test_size, help="Test set size")
    parser.add_argument("--model-type", choices=["linear", "neural"], default=model_config.model_type,
                      help="Model type (linear or neural)")
    parser.add_argument("--use-cv", action="store_true", help="Use cross-validation (linear only)")
    parser.add_argument("--cv-splits", type=int, default=model_config.cv_splits, help="Number of CV splits")
    parser.add_argument("--epochs", type=int, default=model_config.epochs, help="Training epochs (neural only)")
    parser.add_argument("--batch-size", type=int, default=model_config.batch_size, help="Batch size (neural only)")
    parser.add_argument("--learning-rate", type=float, default=model_config.learning_rate, help="Learning rate (neural only)")
    parser.add_argument("--log-level", help="Override default log level from environment")
    parser.add_argument("--embedding-model", help="Embedding model to use")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size (neural only)")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of layers (neural only)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (neural only)")
    parser.add_argument("--data-dir", type=str, default="data",
                      help="Directory for data storage (cache, embeddings, results)")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the JIRA ticket time estimation.

    Args:
        args: Command line arguments
    """
    # Load environment variables
    load_environment()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Setup data directories
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize caches and processors
    text_processor = TextProcessor(
        openai_api_key=get_openai_api_key(),
        cache_dir=str(data_dir),
        model=args.embedding_model or DEFAULT_EMBEDDING_MODEL,
        logger=logger.getChild("text_processor")
    )
    
    data_cache = DataCache(
        cache_dir=str(data_dir),
        logger=logger.getChild("data_cache")
    )
    
    # Get JIRA credentials
    jira_creds = get_jira_credentials()
    
    # Initialize data fetcher
    data_fetcher = JiraDataFetcher(
        jira_url=jira_creds['url'],
        jira_email=jira_creds['email'],
        jira_token=jira_creds['token'],
        text_processor=text_processor,
        cache_dir=str(data_dir),
        logger=logger.getChild("data_fetcher")
    )

    logger.info("Starting JIRA ticket time estimation")
    logger.debug(f"Arguments: {args}")

    try:
        # Fetch and process data
        logger.info("Fetching JIRA tickets")
        df = data_fetcher.fetch_tickets(
            project_keys=args.project_keys,
            max_results=args.max_results,
            exclude_labels=args.exclude_labels,
            include_subtasks=args.include_subtasks,
            use_cache=not args.no_cache,
            force_update=args.force_update,
        )
        
        if len(df) == 0:
            logger.error("No tickets found matching criteria")
            return 1

        logger.info("Processing ticket text")
        # Get descriptions and summaries
        descriptions = df["description"].fillna("").astype(str)
        summaries = df["summary"].fillna("").astype(str)
        
        # Log some stats about the text data
        desc_lengths = descriptions.str.len()
        logger.info(f"Description length stats - min: {desc_lengths.min()}, max: {desc_lengths.max()}, mean: {desc_lengths.mean():.1f}")
        logger.info(f"Empty descriptions: {(desc_lengths == 0).sum()}/{len(df)}")
        
        try:
            # Create metadata for caching
            metadata = [
                {"key": row.key, "summary": row.summary}
                for _, row in df.iterrows()
            ]
            X = text_processor.process_batch(
                texts=descriptions.tolist(),
                queries=summaries.tolist(),  # Use summaries as queries
                metadata=metadata,
                show_progress=True
            )
        except Exception as e:
            # Output error message and traceback
            logger.error(str(e))
            logger.error(traceback.format_exc())
            logger.error("Unable to continue without embeddings.")
            return 1
            
        y = df["time_spent"].values
        logger.debug(f"Time values before model: {y.tolist()}")
        logger.debug(f"Time stats before model - min: {y.min()}, max: {y.max()}, mean: {y.mean():.2f}")

        # Train model
        logger.info(f"Training {args.model_type} model")
        if args.model_type == "linear":
            model = LinearEstimator(logger=logger.getChild("linear_model"))
        else:
            model = NeuralEstimator(
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                logger=logger.getChild("neural_model")
            )
        
        metrics = model.train(
            X,
            y,
            test_size=args.test_size,
            use_cv=args.use_cv,
            n_splits=args.cv_splits if args.model_type == "linear" else None,
            **({"epochs": args.epochs, "batch_size": args.batch_size} if args.model_type == "neural" else {})
        )
        
        # Get test predictions for examples
        X_train, X_test, y_train, y_test, train_indices, test_indices = model.get_train_test_data()

        # Save results
        logger.info("Saving results")
        results_dir = Path(args.data_dir) / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        
        model_file = results_dir / f"{args.model_type}_model"
        model_file = model_file.with_suffix(".pkl" if args.model_type == "linear" else ".pt")
        
        logger.info(f"Saving model to {model_file}")
        model.save(model_file)
        
        # Show example predictions
        logger.info("\nExample Predictions from Test Set:")
        model.show_examples(
            titles=[df.iloc[i]["summary"] for i in test_indices],
            descriptions=[df.iloc[i]["description"] for i in test_indices],
            y_true=y_test,
            y_pred=model.predict(X_test)
        )

        # Print metrics
        logger.info("Final model performance:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return 0
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(args))
