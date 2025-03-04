# JIRA Ticket Time Estimation

Machine learning system to predict JIRA ticket completion times using embeddings and regression techniques.

## Features

- Automatic time estimation for JIRA tickets using ML
- Support for both linear regression, random forest, and neural network models
- Text embedding generation using OpenAI's API
- Efficient data caching and incremental updates
- Cross-validation support for model evaluation
- Configurable data filtering and processing

## Project Structure

```
src/
├── models/
│   ├── linear_model.py    # Linear regression implementation
│   ├── random_forest_model.py    # Random forest implementation
│   └── neural_model.py    # Neural network implementation
├── data_fetcher.py        # JIRA data fetching and caching
├── text_processor.py      # Text processing and embeddings
├── utils.py              # Shared utilities and constants
└── main.py               # CLI and orchestration
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

### Environment Variables

Required:
- JIRA Configuration:
  * `JIRA_URL`: Your JIRA instance URL
  * `JIRA_EMAIL`: Your JIRA email
  * `JIRA_API_TOKEN`: Your JIRA API token
- OpenAI Configuration:
  * `OPENAI_API_KEY`: Your OpenAI API key

Optional:
- Data Storage:
  * `DATA_DIR`: Directory for storing data (default: "data")
- Logging:
  * `LOG_LEVEL`: Logging level (default: "INFO")
- Model Configuration:
  * `EMBEDDING_MODEL`: OpenAI embedding model (default: "ada")
- Model Parameters:
  * `DEFAULT_TEST_SIZE`: Train/test split ratio (default: 0.2)
  * `DEFAULT_CV_SPLITS`: Number of cross-validation folds (default: 5)
  * `DEFAULT_RANDOM_SEED`: Random seed for reproducibility (default: 42)
  * `DEFAULT_EPOCHS`: Training epochs for neural network (default: 100)
  * `DEFAULT_BATCH_SIZE`: Batch size for neural network (default: 32)
  * `DEFAULT_LEARNING_RATE`: Learning rate for neural network (default: 0.001)
  * `DEFAULT_N_ESTIMATORS`: Number of trees in the forest (default: 100)

## Usage

The main script provides various options for data fetching and model training:

### Basic Usage

Train a linear model with default settings:
```bash
python src/main.py --project-keys PROJECT1 PROJECT2
```

### Data Fetching Options

- `--project-keys`: List of JIRA project keys to include
- `--exclude-labels`: List of labels to exclude
- `--max-results`: Maximum number of tickets to fetch (default: 1000)
- `--include-subtasks`: Include subtasks in data fetch
- `--no-cache`: Don't use cached JIRA data
- `--no-cache-update`: Don't update cache with new tickets
- `--force-update`: Force full update of cached data, ignoring timestamps

### Cache Control Examples

```bash
# Force complete refresh of cache
python src/main.py --project-keys PROJECT1 --force-update

# Use cached data only, no updates
python src/main.py --project-keys PROJECT1 --no-cache-update

# Fetch fresh but don't save to cache
python src/main.py --project-keys PROJECT1 --no-cache --no-cache-update

# Normal operation: use cache and fetch updates
python src/main.py --project-keys PROJECT1
```

### Model Options

- `--model-type`: Choose model type (`linear`, `random_forest`, or `neural`, default: from env)
- `--test-size`: Proportion of data for testing (default: from env)
- `--use-cv`: Use cross-validation (linear model only)
- `--cv-splits`: Number of CV splits (default: from env)
- `--random-seed`: Random seed for reproducibility (default: from env)

Random Forest Specific:
- `--n-estimators`: Number of trees in the forest (default: from env)

Neural Network Specific:
- `--epochs`: Training epochs (default: from env)
- `--batch-size`: Batch size (default: from env)
- `--learning-rate`: Learning rate (default: from env)
- `--hidden-size`: Hidden layer size (default: 128)
- `--num-layers`: Number of hidden layers (default: 1)
- `--dropout`: Dropout rate (default: 0.2)

### Examples

Train a linear model with cross-validation:
```bash
python src/main.py --project-keys PROJECT1 --model-type linear --use-cv --cv-splits 5
```

Train a random forest model:
```bash
python src/main.py --project-keys PROJECT1 --model-type random_forest --n-estimators 100
```

Train a neural network:
```bash
python src/main.py --project-keys PROJECT1 --model-type neural --epochs 200 --hidden-size 256
```

Fetch fresh data without using cache:
```bash
python src/main.py --project-keys PROJECT1 --no-cache
```

### Hyperparameter Tuning

For random forest models, you can use the tuning script to find optimal hyperparameters:

```bash
python src/tune_random_forest.py --project-keys PROJECT1
```

This will suggest the optimal number of estimators to use with the main script.

## Model Performance

The system evaluates models using several metrics:
- R² (coefficient of determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)

For linear models with cross-validation, it also provides standard deviations of these metrics across folds.

## Development

- Models are saved in the `models/` directory with appropriate extensions:
  - Linear models: `.pkl`
  - Random forest models: `.pkl`
  - Neural networks: `.pt`
- JIRA data cache is stored in `data/jira_cache/`
- Logging level can be controlled via environment or `--log-level`
- All model parameters can be configured via environment variables for reproducibility
