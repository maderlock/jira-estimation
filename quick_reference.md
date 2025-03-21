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
python -m src.main --project-keys PROJECT1 PROJECT2
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
| `--cv-splits` | Number of cross-validation splits | From env |
| `--random-seed` | Random seed for reproducibility | From env |

### Linear Model Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--fit-intercept` | Whether to calculate the intercept | True |
| `--normalize` | Whether to normalize features | False |

### Random Forest Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-estimators` | Number of trees in the forest | From env |
| `--max-depth` | Maximum depth of the trees | None |
| `--min-samples-split` | Minimum samples required to split a node | 2 |
| `--min-samples-leaf` | Minimum samples required at a leaf node | 1 |
| `--max-features` | Number of features to consider for best split | "auto" |
| `--bootstrap` | Whether to use bootstrap samples | True |

### Neural Network Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--hidden-layer-sizes` | Size of hidden layers | (100,) |
| `--activation` | Activation function | "relu" |
| `--solver` | Weight optimization solver | "adam" |
| `--alpha` | L2 penalty parameter | 0.0001 |
| `--batch-size` | Size of minibatches | From env |
| `--learning-rate` | Learning rate schedule | "constant" |
| `--learning-rate-init` | Initial learning rate | From env |
| `--max-iter` | Maximum number of iterations (epochs) | From env |

### Output Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output-dir` | Directory for output files | "results" |
| `--save-model` | Save the trained model | False |
| `--log-level` | Logging level | From env |

## Random Forest Tuning Script

The system includes a separate script for tuning Random Forest hyperparameters:

```bash
python -m scripts.tune_random_forest --project-keys PROJECT1 PROJECT2 --n-trials 100
```

### Tuning Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--n-trials` | Number of optimization trials | 50 |
| `--study-name` | Name for the Optuna study | "rf_tuning" |
| `--storage` | Database URL for study storage | None |
| `--include-subtasks` | Include subtasks in data fetch | False |
| `--no-cache` | Don't use cached JIRA data | False |
| `--log-level` | Logging level for tuning script | "INFO" |
| `--pass-log-level` | Log level to pass to main script | None |

## Common Examples

### Basic Training and Evaluation

```bash
python -m src.main --project-keys DEV SUPPORT --model-type random_forest
```

### Using Cross-Validation

```bash
python -m src.main --project-keys DEV SUPPORT --model-type random_forest --cv-splits 10
```

### Training with Custom Random Forest Parameters

```bash
python -m src.main --project-keys DEV SUPPORT --model-type random_forest --n-estimators 200 --max-depth 10 --min-samples-split 5
```

### Tuning Random Forest Hyperparameters

```bash
python -m scripts.tune_random_forest --project-keys DEV SUPPORT --n-trials 100 --study-name "dev_support_tuning"
```

### Saving a Trained Model

```bash
python -m src.main --project-keys DEV SUPPORT --model-type random_forest --save-model
```
