"""Main script for training and evaluating the JIRA estimation model."""
import argparse
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.data_fetcher import JiraDataFetcher
from src.model import ModelTrainer
from src.text_processor import TextProcessor


def fetch_and_process_data(
    max_results: int = 2000,
    use_cache: bool = True,
    update_cache: bool = True,
    projects: list[str] = None,
    exclude_labels: list[str] = None,
) -> None:
    """
    Fetch and process JIRA data.
    
    Args:
        max_results: Maximum number of tickets to fetch
        use_cache: Whether to use cached JIRA data
        update_cache: Whether to update cache with new tickets
        projects: List of project keys to include
        exclude_labels: List of labels to exclude
    """
    print("Fetching JIRA data...")
    fetcher = JiraDataFetcher()
    df = fetcher.fetch_completed_issues(
        max_results=max_results,
        projects=projects,
        exclude_labels=exclude_labels,
        use_cache=use_cache,
        update_cache=update_cache,
    )
    
    print("Processing text and generating embeddings...")
    processor = TextProcessor()
    embeddings = processor.process_and_embed(df)
    
    # Save processed data
    data_path = Path(DATA_DIR)
    df.to_pickle(data_path / "processed_tickets.pkl")
    pd.DataFrame(embeddings).to_pickle(data_path / "embeddings.pkl")
    print(f"Processed {len(df)} tickets")


def train_model(
    model_type: str,
    test_size: float = 0.2,
    use_cv: bool = False,
    n_splits: int = 5,
) -> None:
    """
    Train and evaluate the model.
    
    Args:
        model_type: Type of model to train ('linear' or 'neural')
        test_size: Proportion of data to use for testing
        use_cv: Whether to use cross-validation (only for linear model)
        n_splits: Number of splits for cross-validation
    """
    # Load processed data
    data_path = Path(DATA_DIR)
    df = pd.read_pickle(data_path / "processed_tickets.pkl")
    embeddings = pd.read_pickle(data_path / "embeddings.pkl").values
    
    print(f"Training {model_type} model...")
    trainer = ModelTrainer(model_type)
    
    if use_cv:
        print(f"Using {n_splits}-fold cross-validation...")
    else:
        print(f"Using {int((1-test_size)*100)}/{int(test_size*100)} train/test split...")
    
    metrics = trainer.train(
        embeddings,
        df["time_spent_hours"].values,
        test_size=test_size,
        use_cv=use_cv,
        n_splits=n_splits,
    )
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Save the model
    trainer.save_model(f"jira_estimator_{model_type}")
    print(f"\nModel saved as jira_estimator_{model_type}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="JIRA Estimation Model Training")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch and process new data from JIRA",
    )
    parser.add_argument(
        "--train",
        choices=["linear", "neural"],
        help="Train a new model (specify type)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=2000,
        help="Maximum number of JIRA issues to fetch",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached JIRA data",
    )
    parser.add_argument(
        "--no-cache-update",
        action="store_true",
        help="Don't update the cache with new tickets",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        help="List of project keys to include",
    )
    parser.add_argument(
        "--exclude-labels",
        nargs="+",
        help="List of labels to exclude",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Use cross-validation (only for linear model)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of splits for cross-validation (default: 5)",
    )
    
    args = parser.parse_args()
    
    if args.fetch:
        fetch_and_process_data(
            max_results=args.max_results,
            use_cache=not args.no_cache,
            update_cache=not args.no_cache_update,
            projects=args.projects,
            exclude_labels=args.exclude_labels,
        )
    
    if args.train:
        train_model(
            args.train,
            test_size=args.test_size,
            use_cv=args.use_cv,
            n_splits=args.cv_splits,
        )


if __name__ == "__main__":
    main()
