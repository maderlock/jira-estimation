# JIRA Ticket Time Estimation: Quick Reference

This document provides a quick reference for parameters, configurations, and common commands for the JIRA Ticket Time Estimation system.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JIRA_URL` | Your JIRA instance URL | *Required* |
| `JIRA_EMAIL` | Your JIRA email | *Required* |
| `JIRA_API_TOKEN` | Your JIRA API token | *Required* |
| `OPENAI_API_KEY` | Your OpenAI API key | *Required* |
| `DATA_DIR` | Directory for storing data | "data" |
| `LOG_LEVEL` | Logging level | "INFO" |
| `EMBEDDING_MODEL` | OpenAI embedding model | "ada" |
| `DEFAULT_TEST_SIZE` | Train/test split ratio | 0.2 |
| `DEFAULT_CV_SPLITS` | Number of cross-validation folds | 5 |
| `DEFAULT_RANDOM_SEED` | Random seed for reproducibility | 42 |
| `DEFAULT_EPOCHS` | Training epochs for neural network | 100 |
| `DEFAULT_BATCH_SIZE` | Batch size for neural network | 32 |
| `DEFAULT_LEARNING_RATE` | Learning rate for neural network | 0.001 |
| `DEFAULT_N_ESTIMATORS` | Number of trees in the forest | 100 |

## Command-Line Parameters

### Basic Usage

```bash
python src/main.py --project-keys PROJECT1 PROJECT2
```

### Data Fetching Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--project-keys` | List of JIRA project keys to include | *Required* |
| `--exclude-labels` | List of labels to exclude | None |
| `--max-results` | Maximum number of tickets to fetch | 1000 |
| `--include-subtasks` | Include subtasks in data fetch | False |
| `--no-cache` | Don't use cached JIRA data | False |
| `--no-cache-update` | Don't update cache with new tickets | False |
| `--force-update` | Force full update of cached data | False |

### Model Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-type` | Model type (`linear`, `random_forest`, or `neural`) | From env |
| `--test-size` | Proportion of data for testing | From env |
| `--use-cv` | Use cross-validation (linear model only) | False |
| `--cv-splits` | Number of CV splits | From env |
| `--random-seed` | Random seed for reproducibility | From env |

### Random Forest Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-estimators` | Number of trees in the forest | From env |

### Neural Network Specific

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Training epochs | From env |
| `--batch-size` | Batch size | From env |
| `--learning-rate` | Learning rate | From env |
| `--hidden-size` | Hidden layer size | 128 |
| `--num-layers` | Number of hidden layers | 1 |
| `--dropout` | Dropout rate | 0.2 |

## Common Commands

### Basic Training

```bash
# Train a linear model
python src/main.py --project-keys PROJECT1 --model-type linear

# Train a random forest model
python src/main.py --project-keys PROJECT1 --model-type random_forest

# Train a neural network
python src/main.py --project-keys PROJECT1 --model-type neural
```

### Cache Control

```bash
# Force complete refresh of cache
python src/main.py --project-keys PROJECT1 --force-update

# Use cached data only, no updates
python src/main.py --project-keys PROJECT1 --no-cache-update

# Fetch fresh but don't save to cache
python src/main.py --project-keys PROJECT1 --no-cache --no-cache-update
```

### Advanced Options

```bash
# Linear model with cross-validation
python src/main.py --project-keys PROJECT1 --model-type linear --use-cv --cv-splits 5

# Random forest with custom estimators
python src/main.py --project-keys PROJECT1 --model-type random_forest --n-estimators 200

# Neural network with custom architecture
python src/main.py --project-keys PROJECT1 --model-type neural --epochs 200 --hidden-size 256 --num-layers 2
```

### Hyperparameter Tuning

```bash
# Tune random forest hyperparameters
python src/tune_random_forest.py --project-keys PROJECT1
```

## Output Files

| File | Description | Location |
|------|-------------|----------|
| JIRA Cache | Cached JIRA ticket data | `data/jira_cache/` |
| Embedding Cache | Cached text embeddings | `data/embedding_cache/` |
| Linear Models | Saved linear regression models | `models/*.pkl` |
| Random Forest Models | Saved random forest models | `models/*.pkl` |
| Neural Network Models | Saved neural network models | `models/*.pt` |
| Logs | Application logs | Based on logging configuration |

## Performance Metrics

The system evaluates models using several metrics:
- R² (coefficient of determination): Higher is better, measures explained variance
- MAE (Mean Absolute Error): Lower is better, average absolute prediction error
- RMSE (Root Mean Square Error): Lower is better, penalizes large errors more heavily
