"""Main entry point for JIRA ticket time estimation."""
import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Tuple, List, Dict
from logging import Logger

import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import RegressorMixin

from data_fetching import TicketFetcher, JiraDataFetcher
from model_learner import ModelLearner
from text_processing import AbstractTextProcessor, AITextProcessor
from text_processing.constants import DEFAULT_EMBEDDING_MODEL
from utils import get_model_config, setup_logging, load_environment, get_jira_credentials, get_openai_api_key

def log_start_stop(func: Callable) -> Callable:
    def wrapper(self, *args, **kwargs) -> Any:
        self.logger.info(f"Starting {func.__name__}")
        self.logger.debug(f"{func.__name__} arguments: {args}, {kwargs}")
        ret_value = func(self, *args, **kwargs)
        self.logger.info(f"Finished {func.__name__}")
        return ret_value
    return wrapper

class JiraAI:
    data_fetcher: TicketFetcher
    text_processor: AbstractTextProcessor
    model_learner: ModelLearner
    logger: Logger

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command line arguments."""
        # Load environment variables
        load_environment()
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
        parser.add_argument("--model-type", default=model_config.model_type, choices=["linear", "random_forest", "neural"],
                        help="Model type (linear, random_forest or neural)")
        parser.add_argument("--use-cv", action="store_true", help="Use cross-validation")
        parser.add_argument("--cv-splits", type=int, default=model_config.cv_splits, help="Number of CV splits")
        parser.add_argument("--random-seed", type=int, default=model_config.random_seed, help="Random seed")
        parser.add_argument("--n-estimators", type=int, default=model_config.n_estimators,
                        help="Number of estimators for random forest (overrides environment variable)")
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
        parser.add_argument("--max-depth", type=int, default=model_config.max_depth,
                        help="Maximum tree depth for random forest (default: %(default)s)")
        parser.add_argument("--min-samples-split", type=int, default=model_config.min_samples_split,
                        help="Minimum samples required to split a node (default: %(default)s)")
        parser.add_argument("--min-samples-leaf", type=int, default=model_config.min_samples_leaf,
                        help="Minimum samples required at a leaf node (default: %(default)s)")
        parser.add_argument("--max-features", type=str, default=model_config.max_features,
                        help="Number of features to consider (default: %(default)s)")
        parser.add_argument("--bootstrap", type=bool, default=model_config.bootstrap,
                        help="Whether to use bootstrap sampling (default: %(default)s)")
        return parser.parse_args()

    def _setup_utilities(self) -> None:
        self.logger = setup_logging(self.args.log_level)
        
        # Setup data directories
        self.data_dir: str = Path(self.args.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @log_start_stop
    def _create_services(self) -> None:
        self.text_processor = AITextProcessor(
            openai_api_key=get_openai_api_key(),
            cache_dir=self.data_dir,
            model=self.args.embedding_model or DEFAULT_EMBEDDING_MODEL,
            logger=self.logger.getChild("text_processor")
        )

        # Initialize data fetcher
        jira_creds = get_jira_credentials()
        self.data_fetcher = JiraDataFetcher(
            jira_url=jira_creds['url'],
            jira_email=jira_creds['email'],
            jira_token=jira_creds['token'],
            text_processor=self.text_processor,
            cache_dir=self.data_dir,
            logger=self.logger.getChild("data_fetcher")
        )

    @log_start_stop
    def _fetch_tickets(self) -> DataFrame:
        df = self.data_fetcher.fetch_tickets(
            project_keys=self.args.project_keys,
            max_results=self.args.max_results,
            exclude_labels=self.args.exclude_labels,
            include_subtasks=self.args.include_subtasks,
            use_cache=not self.args.no_cache,
            force_update=self.args.force_update,
        )
        
        if len(df) == 0:
            raise Exception("No tickets found matching criteria")

        return df

    @log_start_stop
    def _transform_data(self, df: DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Process fields - ensure they are strings and non-empty
        proc_fields: Dict[str, List[str]] = {}
        for col in ["description", "summary", "key"]:
            proc_fields[col] = df[col].fillna("").astype(str)
        
        # TODO: Move metadata into text processor
        metadata = [
            {"key": key, "summary": summary}
            for key, summary in zip(proc_fields["key"], proc_fields["summary"])
        ]
        X = self.text_processor.process_batch(
            texts=proc_fields["description"].tolist(),
            queries=proc_fields["summary"].tolist(),  # Use summaries as queries
            metadata=metadata,
            show_progress=True
        )
        y = df["time_spent"].values

        return (X, y)

    def _output_initial_stats(self, df: DataFrame) -> None:
        """
        Output initial statistics about the data.
        """
        self.logger.info(f"Number of tickets: {len(df)}")

    @log_start_stop
    def _make_model(self) -> RegressorMixin:
        #TODO: Use factory pattern to build this
        if self.args.model_type == "linear":
            model = LinearRegression()
        elif self.args.model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=self.args.n_estimators,
                max_depth=self.args.max_depth,
                min_samples_split=self.args.min_samples_split,
                min_samples_leaf=self.args.min_samples_leaf,
                max_features=self.args.max_features,
                bootstrap=self.args.bootstrap,
                random_state=self.args.random_seed
            )
        else:
            model = MLPRegressor()
        return model

    @log_start_stop
    def _make_model_learner(self, model: RegressorMixin) -> ModelLearner:
        """
        Make a model learner based on the provided model
        """
        model_learner = ModelLearner(
            model=model,
            logger=self.logger.getChild("model_learner")
        )
        return model_learner

    @log_start_stop
    def _train(self, model_learner: ModelLearner, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the model
        """
        metrics = model_learner.train(
            X,
            y,
            test_size=self.args.test_size,
            use_cv=self.args.use_cv,
            n_splits=self.args.cv_splits,
            **({"epochs": self.args.epochs, "batch_size": self.args.batch_size} if self.args.model_type == "neural" else {})
        )
        return metrics

    @log_start_stop
    def _evaluate(self, model_learner: ModelLearner) -> None:
        # Get test predictions for examples
        _, X_test, _, y_test, _, test_indices = model_learner.get_train_test_data()
        # Show example predictions
        logger.info("\nExample Predictions from Test Set:")
        model_learner.show_examples(
            titles=[df.iloc[i]["summary"] for i in test_indices],
            descriptions=[df.iloc[i]["description"] for i in test_indices],
            y_true=y_test,
            y_pred=model_learner.predict(X_test)
        )
        # Print metrics
        logger.info("Final model performance:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    @log_start_stop
    def _save_model(self) -> None:
        """
        Save the model
        """
        results_dir = self.data_dir / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        
        #TODO: Delegate what we need to to the other models
        model_file_name = results_dir / f"{args.model_type}_model"
        model_file_name = model_file_name.with_suffix(".pkl" if args.model_type == "linear" else ".pt")
        
        logger.info(f"Saving model to {model_file_name}")
        model_learner.save(model_file_name)

    def execute(self) -> int:
        """
        Main function to run the JIRA ticket time estimation.
        """

        self._setup_utilities()

        self.logger.info("Starting JIRA ticket time estimation")
        self.logger.debug("Arguments",self.args)

        self._create_services();

        try:
            # Request tickets
            df: DataFrame = self._fetch_tickets()

            # Log stats around inputs
            self._output_initial_stats(df)    

            # Transform data (e.g., feature engineering)
            (X,y) = self._transform_data(df)

            # Train model (including any cross-validation)
            model: RegressorMixin = self._make_model()
            model_learner = self._make_model_learner(model)
            self._train(model_learner, X, y)

            # Evaluate model
            self._evaluate(model_learner)

            # Save model
            self._save_model()
        except Exception as e:
            self.logger.error(f"Failed to execute: {str(e)}")
            self.logger.error(traceback.format_exc())
            return 1

        return 0

if __name__ == "__main__":
    args = JiraAI.parse_args()
    main_class = JiraAI(args)
    return_code = main_class.execute()
    sys.exit(return_code)
