#!/usr/bin/env python3
"""
Script to tune the n_estimators hyperparameter for random forest models.
Uses a simple iterative splitting algorithm to find optimal value.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_project_root

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tune random forest n_estimators")
    parser.add_argument("--project-keys", nargs="+", required=True,
                      help="JIRA project keys to fetch tickets from")
    parser.add_argument("--max-results", type=int,
                      help="Maximum number of tickets to fetch per project")
    parser.add_argument("--min-estimators", type=int, default=10,
                      help="Minimum number of estimators to try")
    parser.add_argument("--max-estimators", type=int, default=200,
                      help="Maximum number of estimators to try")
    parser.add_argument("--cv-splits", type=int, default=3,
                      help="Number of CV splits for each test")
    parser.add_argument("--output-dir", type=str, default="tuning_results",
                      help="Directory to store results")
    return parser.parse_args()

def run_test(n_estimators: int, project_keys: List[str], cv_splits: int,
             max_results: int = None) -> Tuple[float, float, float]:
    """Run main.py with specified parameters and return MSE."""
    # Get the path to main.py relative to the project root
    main_script = os.path.join(get_project_root(), "src", "main.py")
    
    cmd = [
        sys.executable,
        main_script,
        "--model-type", "random_forest",
        "--n-estimators", str(n_estimators),
        "--use-cv",
        "--include-subtasks",
        "--cv-splits", str(cv_splits),
        "--project-keys"
    ] + project_keys

    if max_results:
        cmd.extend(["--max-results", str(max_results)])

    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running test with n_estimators={n_estimators}:")
        print(result.stderr)
        return None, None, None

    # Parse the output to get MSE
    try:
        # Look for JSON output in the stdout
        lines = result.stdout.split('\n')
        for line in lines:
            if line.startswith('{') and '"mse"' in line:
                metrics = json.loads(line)
                return (
                    metrics.get('mse', float('inf')),
                    metrics.get('mae', float('inf')),
                    metrics.get('r2', float('-inf'))
                )
    except Exception as e:
        print(f"Error parsing output for n_estimators={n_estimators}: {e}")
        return None, None, None

def binary_search_optimal_estimators(min_est: int, max_est: int, project_keys: List[str],
                                   cv_splits: int, max_results: int = None) -> List[dict]:
    """Use binary search to find optimal n_estimators value."""
    results = []
    
    # Initial test points: min, max, and midpoint
    test_points = sorted(list(set([
        min_est,
        max_est,
        (min_est + max_est) // 2
    ])))
    
    while len(test_points) > 0:
        n_est = test_points.pop(0)
        print(f"\nTesting n_estimators={n_est}...")
        
        mse, mae, r2 = run_test(n_est, project_keys, cv_splits, max_results)
        if mse is not None:
            results.append({
                'n_estimators': n_est,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'timestamp': datetime.now().isoformat()
            })
            print(f"Results: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # Add midpoints between current point and neighbors
        neighbors = sorted([r['n_estimators'] for r in results])
        idx = neighbors.index(n_est)
        
        if idx > 0:
            mid_left = (neighbors[idx-1] + n_est) // 2
            if (mid_left not in [r['n_estimators'] for r in results] and
                mid_left not in test_points and
                mid_left > min_est):
                test_points.append(mid_left)
        
        if idx < len(neighbors)-1:
            mid_right = (neighbors[idx+1] + n_est) // 2
            if (mid_right not in [r['n_estimators'] for r in results] and
                mid_right not in test_points and
                mid_right < max_est):
                test_points.append(mid_right)
    
    return results

def save_results(results: List[dict], output_dir: str):
    """Save results to CSV file."""
    # Make output_dir path absolute relative to project root
    output_dir = os.path.join(get_project_root(), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'rf_tuning_results_{timestamp}.csv')
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_estimators', 'mse', 'mae', 'r2', 'timestamp'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")
    
    # Find best result
    best_result = min(results, key=lambda x: x['mse'])
    print("\nBest configuration:")
    print(f"n_estimators: {best_result['n_estimators']}")
    print(f"MSE: {best_result['mse']:.4f}")
    print(f"MAE: {best_result['mae']:.4f}")
    print(f"R²: {best_result['r2']:.4f}")
    
    # Print suggested command
    project_keys_str = ' '.join(args.project_keys)
    main_script = os.path.join("src", "main.py")
    print("\nSuggested command:")
    print(f"python {main_script} --project-keys {project_keys_str} --model-type random_forest "
          f"--n-estimators {best_result['n_estimators']}")

if __name__ == '__main__':
    args = parse_args()
    
    print("Starting random forest hyperparameter tuning...")
    print(f"Project keys: {args.project_keys}")
    print(f"CV splits: {args.cv_splits}")
    print(f"Testing range: {args.min_estimators} to {args.max_estimators} estimators")
    
    results = binary_search_optimal_estimators(
        args.min_estimators,
        args.max_estimators,
        args.project_keys,
        args.cv_splits,
        args.max_results
    )
    
    save_results(results, args.output_dir)
