#!/usr/bin/env python3
"""
Script to tune hyperparameters for random forest models using Optuna.
"""

import argparse
import json
import os
import subprocess
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances
    import numpy as np
except ImportError:
    print("Required dependencies are not installed. Please install all dependencies:")
    print("  1. Create and activate your virtual environment")
    print("  2. Run: pip install -r requirements.txt")
    sys.exit(1)

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_project_root

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tune random forest hyperparameters with Optuna")
    parser.add_argument("--project-keys", nargs="+", required=True,
                      help="JIRA project keys to fetch tickets from")
    parser.add_argument("--max-results", type=int,
                      help="Maximum number of tickets to fetch per project")
    parser.add_argument("--cv-splits", type=int, default=3,
                      help="Number of CV splits for each test")
    parser.add_argument("--n-trials", type=int, default=20,
                      help="Number of Optuna trials to run")
    parser.add_argument("--output-dir", type=str, default="tuning_results",
                      help="Directory to store results")
    parser.add_argument("--study-name", type=str, default=None,
                      help="Name for the Optuna study (default: auto-generated)")
    parser.add_argument("--include-subtasks", action="store_true",
                      help="Include subtasks in the data")
    parser.add_argument("--no-cache", action="store_true",
                      help="Don't use cached JIRA data")
    parser.add_argument("--storage", type=str, default=None,
                      help="Optuna storage URL (default: in-memory)")
    parser.add_argument("--pass-log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Log level to pass to the main script")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      default="INFO", help="Log level for the tuning script (default: INFO)")
    return parser.parse_args()

def extract_metrics_from_output(output: str) -> Dict[str, float]:
    """
    Extract metrics from the output of main.py.
    
    Args:
        output: String output from main.py
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    try:
        for line in output.split('\n'):
            # Look for lines containing metrics
            for metric in ['cv_rmse_mean', 'cv_mae_mean', 'cv_r2_mean', 'rmse', 'mae', 'r2']:
                if f"{metric}:" in line:
                    # Extract the metric value from the log line
                    # Format example: "2025-03-06 16:12:57 - root - INFO - cv_rmse_mean: 5.7126"
                    value_str = line.split(f"{metric}:")[1].strip()
                    metrics[metric] = float(value_str)
    except Exception as e:
        logger.error(f"Error parsing metrics from output: {e}")
    
    return metrics

def run_model(params: Dict[str, Any], project_keys: List[str], 
              cv_splits: int, max_results: Optional[int] = None,
              include_subtasks: bool = False, no_cache: bool = False,
              pass_log_level: Optional[str] = None) -> float:
    """
    Run main.py with specified parameters and return mean squared error.
    """
    # Get the path to main.py relative to the project root
    main_script = os.path.join(get_project_root(), "src", "main.py")
    
    # Build command
    cmd = [
        sys.executable,
        main_script,
        "--model-type", "random_forest",
        "--use-cv",
        "--cv-splits", str(cv_splits),
        "--project-keys"
    ] + project_keys
    
    # Add all model parameters
    for param_name, param_value in params.items():
        if param_value is not None:
            cmd.extend([f"--{param_name.replace('_', '-')}", str(param_value)])
    
    # Add optional data parameters
    if max_results:
        cmd.extend(["--max-results", str(max_results)])
    if include_subtasks:
        cmd.append("--include-subtasks")
    if no_cache:
        cmd.append("--no-cache")
    if pass_log_level:
        cmd.extend(["--log-level", pass_log_level])
    
    # Run the command and capture output
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Error running model: {result.stderr}")
        return float('inf')  # Return infinity for failed runs

    # Parse the output to get metrics
    metrics = extract_metrics_from_output(result.stderr)
    
    # First try to get cv_rmse_mean (preferred metric)
    if 'cv_rmse_mean' in metrics:
        rmse = metrics['cv_rmse_mean']
        logger.info(f"Trial result: RMSE={rmse:.4f}")
        return rmse
    
    # Fallback to regular RMSE if cv_rmse_mean is not available
    if 'rmse' in metrics:
        rmse = metrics['rmse']
        logger.info(f"Trial result: RMSE={rmse:.4f} (non-CV)")
        return rmse
    
    # If no RMSE metrics found, return infinity
    logger.warning("Could not find RMSE in output, returning infinity")
    return float('inf')

def objective(trial, args):
    """
    Optuna objective function for hyperparameter optimization.
    
    Returns the RMSE (Root Mean Squared Error) from cross-validation,
    which Optuna will try to minimize.
    """
    # Define model parameters to tune
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 300),
        "max_depth": trial.suggest_int("max_depth", -1, 50),  # -1 means None (unlimited)
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
    }
    
    logger.info(f"Trial {trial.number}: Testing with parameters: {params}")
    
    # Handle max_depth=None
    if params["max_depth"] <= 0:
        params["max_depth"] = None
    
    return run_model(
        params=params,
        project_keys=args.project_keys,
        cv_splits=args.cv_splits,
        max_results=args.max_results,
        include_subtasks=args.include_subtasks,
        no_cache=args.no_cache,
        pass_log_level=args.pass_log_level
    )

def save_study_results(study, output_dir):
    """Save study results to files."""
    # Make output_dir path absolute relative to project root
    output_dir = os.path.join(get_project_root(), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save study statistics
    stats_file = os.path.join(output_dir, f'optuna_stats_{timestamp}.json')
    with open(stats_file, 'w') as f:
        stats = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
            "datetime": timestamp
        }
        json.dump(stats, f, indent=2)
    
    logger.info(f"Study statistics saved to: {stats_file}")
    
    # Save all trials data
    trials_file = os.path.join(output_dir, f'optuna_trials_{timestamp}.csv')
    trial_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            data = {"number": trial.number, "value": trial.value, **trial.params}
            trial_data.append(data)
    
    if trial_data:
        import csv
        with open(trials_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trial_data[0].keys())
            writer.writeheader()
            writer.writerows(trial_data)
        logger.info(f"Trial data saved to: {trials_file}")
    
    # Try to save visualizations if plotly is available
    try:
        # Save optimization history plot
        history_file = os.path.join(output_dir, f'optuna_history_{timestamp}.html')
        plot_optimization_history(study).write_html(history_file)
        logger.info(f"Optimization history plot saved to: {history_file}")
        
        # Save parameter importance plot
        importance_file = os.path.join(output_dir, f'optuna_importance_{timestamp}.html')
        plot_param_importances(study).write_html(importance_file)
        logger.info(f"Parameter importance plot saved to: {importance_file}")
    except Exception as e:
        logger.warning(f"Could not save visualization plots: {e}")

def main():
    """Run the hyperparameter tuning."""
    args = parse_args()
    
    # Configure logging based on command-line argument
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    
    logger.info("Starting random forest hyperparameter tuning with Optuna")
    logger.info(f"Project keys: {args.project_keys}")
    logger.info(f"CV splits: {args.cv_splits}")
    logger.info(f"Number of trials: {args.n_trials}")
    
    # Generate study name if not provided
    study_name = args.study_name or f"rf_tuning_{'_'.join(args.project_keys)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create the study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # We want to minimize RMSE
        storage=args.storage,
        load_if_exists=True
    )
    
    # Run the optimization
    study.optimize(
        lambda trial: objective(trial, args), 
        n_trials=args.n_trials
    )
    
    # Print results
    logger.info("\nOptimization completed!")
    logger.info(f"Best trial: #{study.best_trial.number}")
    logger.info(f"Best RMSE: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for param, value in study.best_params.items():
        logger.info(f"  {param}: {value}")
    
    # Save results
    save_study_results(study, args.output_dir)
    
    # Print suggested command
    logger.info("\nSuggested command:")
    main_script = os.path.join("src", "main.py")
    cmd = f"python {main_script} --project-keys {' '.join(args.project_keys)} --model-type random_forest"
    
    # Add optimized parameters
    for param, value in study.best_params.items():
        if value is not None:
            cmd += f" --{param.replace('_', '-')} {value}"
    
    logger.info(cmd)

if __name__ == '__main__':
    main()
