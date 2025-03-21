# JIRA Ticket Time Estimation: Implementation Details

This document provides technical details about the implementation of the JIRA Ticket Time Estimation system.

## Project Structure

```
src/
├── data_fetching/
│   ├── __init__.py        # Package exports
│   ├── cache.py           # Data caching implementation
│   └── data_fetcher.py    # JIRA data fetching with TicketFetcher protocol
├── text_processing/
│   ├── __init__.py        # Package exports
│   ├── constants.py       # Constants for embedding models
│   ├── embedding_cache.py # Caching for embeddings
│   ├── exceptions.py      # Custom exceptions
│   └── text_processor.py  # Text processing with AbstractTextProcessor and AITextProcessor
├── __init__.py            # Root package exports
├── main.py                # CLI and JiraAI orchestration class
├── model_learner.py       # Model training and evaluation
└── utils.py               # Shared utilities and constants
```

## Core Components

### Data Fetching (`data_fetching` package)

The data fetching package is responsible for:
- Connecting to the JIRA API using provided credentials
- Fetching ticket data based on specified project keys and filters
- Implementing caching to avoid redundant API calls
- Providing incremental updates to the cache

Key implementation details:
- `TicketFetcher` Protocol defines the interface for ticket fetching
- `JiraDataFetcher` implements the protocol for JIRA integration
- Uses the `jira` Python package for API interaction
- `DataCache` implements a timestamp-based caching system
- Supports filtering by project, label, and ticket type
- Handles pagination for large result sets

### Text Processing (`text_processing` package)

The text processing package handles:
- Cleaning and normalizing text data from tickets
- Generating embeddings using OpenAI's API
- Combining multiple text fields into a single feature vector

Technical details:
- `AbstractTextProcessor` defines the interface for text processing
- `AITextProcessor` implements the interface using OpenAI's API
- Uses OpenAI's embedding models (default: "ada")
- `EmbeddingCache` provides caching of embeddings to reduce API costs
- Handles text normalization and formatting removal
- Combines ticket descriptions and summaries with attention mechanism

### Main Orchestration (`main.py`)

The `JiraAI` class orchestrates the entire process:
- Parses command-line arguments
- Initializes components (data fetcher, text processor, model learner)
- Manages the workflow from data fetching to model training and evaluation
- Handles logging and error reporting

Technical details:
- Uses the `log_start_stop` decorator for method execution logging
- Modular private methods for each step of the process
- Supports different model types through a factory-like approach
- Handles configuration through command-line arguments and environment variables

### Model Learning (`model_learner.py`)

The `ModelLearner` class handles:
- Training models on the processed data
- Evaluating model performance
- Cross-validation for all model types
- Saving trained models for later use

Technical details:
- Supports multiple scikit-learn regression models
- Implements train/test splitting with configurable ratios
- Provides metrics calculation (RMSE, MAE, R²)
- Supports cross-validation with configurable number of splits for all model types

## Model Types

The system supports three types of regression models:

### Linear Regression
- Uses scikit-learn's `LinearRegression` implementation
- Simple and interpretable model for basic prediction
- Fastest to train and evaluate

### Random Forest Regression
- Uses scikit-learn's `RandomForestRegressor`
- Ensemble method that can capture non-linear relationships
- Configurable hyperparameters:
  - `n_estimators`: Number of trees in the forest
  - `max_depth`: Maximum depth of the trees
  - `min_samples_split`: Minimum samples required to split a node
  - `min_samples_leaf`: Minimum samples required at a leaf node
  - `max_features`: Number of features to consider for best split
  - `bootstrap`: Whether to use bootstrap samples

### Neural Network Regression
- Uses scikit-learn's `MLPRegressor`
- Multi-layer perceptron for complex pattern recognition
- Configurable hyperparameters:
  - `hidden_layer_sizes`: Size of hidden layers
  - `activation`: Activation function
  - `solver`: Weight optimization solver
  - `alpha`: L2 penalty parameter
  - `batch_size`: Size of minibatches
  - `learning_rate`: Learning rate schedule
  - `max_iter`: Maximum number of iterations (epochs)

## Cross-Validation

Cross-validation is implemented for all model types to provide more robust performance evaluation:
- Uses scikit-learn's `KFold` for splitting data
- Configurable number of splits via the `--cv-splits` parameter
- Reports mean and standard deviation of performance metrics across folds
- Helps identify overfitting and provides more reliable performance estimates

## Random Forest Tuning

The system includes a separate script for tuning Random Forest hyperparameters using Optuna:
- Supports tuning multiple hyperparameters simultaneously
- Uses Bayesian optimization to efficiently search the parameter space
- Optimizes for RMSE from cross-validation
- Provides visualization of the optimization process
- Saves results in multiple formats (JSON, CSV, HTML plots)

Key components of the tuning script:

1. **create_study**:
   - Initializes an Optuna study with specified parameters
   - Supports persistence via SQLite database

2. **objective**:
   - Defines the objective function for optimization
   - Runs the model with candidate parameters
   - Returns RMSE for Optuna to minimize

3. **run_model**:
   - Executes the main script with specified parameters
   - Extracts RMSE from the output logs
   - Handles errors and timeouts

4. **optimize**:
   - Runs the optimization process for a specified number of trials
   - Implements pruning for inefficient parameter combinations

5. **save_study_results**:
   - Exports statistics to JSON and trial data to CSV
   - Generates visualization plots (optimization history, parameter importance)

6. **main**:
   - Parses command-line arguments
   - Orchestrates the entire tuning process
   - Handles logging and error reporting
