# JIRA Ticket Time Estimation: Implementation Details

This document provides technical details about the implementation of the JIRA Ticket Time Estimation system.

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

## Core Components

### Data Fetcher (`data_fetcher.py`)

The data fetcher is responsible for:
- Connecting to the JIRA API using provided credentials
- Fetching ticket data based on specified project keys and filters
- Implementing caching to avoid redundant API calls
- Providing incremental updates to the cache

Key implementation details:
- Uses the `jira` Python package for API interaction
- Implements a timestamp-based caching system
- Supports filtering by project, label, and ticket type
- Handles pagination for large result sets

### Text Processor (`text_processor.py`)

The text processor handles:
- Cleaning and normalizing text data from tickets
- Generating embeddings using OpenAI's API
- Combining multiple text fields into a single feature vector

Technical details:
- Uses OpenAI's embedding models (default: "ada")
- Implements caching of embeddings to reduce API costs
- Handles text normalization (lowercasing, punctuation removal, etc.)
- Combines title, description, and other fields with appropriate weighting

### Models

#### Linear Model (`linear_model.py`)
- Uses scikit-learn's `LinearRegression` implementation
- Supports cross-validation for more robust evaluation
- Implements feature scaling for better performance

#### Random Forest Model (`random_forest_model.py`)
- Uses scikit-learn's `RandomForestRegressor`
- Supports hyperparameter tuning via the separate tuning script
- Implements feature importance analysis

#### Neural Model (`neural_model.py`)
- Implemented using PyTorch
- Configurable architecture (layers, hidden size, dropout)
- Implements early stopping to prevent overfitting
- Uses Adam optimizer with configurable learning rate

### Main Script (`main.py`)

The main script provides:
- Command-line interface using `argparse`
- Orchestration of the entire workflow
- Configuration loading from environment variables
- Logging and error handling

## Data Flow

1. `main.py` parses arguments and loads configuration
2. `data_fetcher.py` retrieves ticket data from JIRA or cache
3. `text_processor.py` converts ticket text to embeddings
4. The selected model is trained on the processed data
5. Model performance is evaluated and results are displayed
6. The trained model is saved for future use

## Caching System

The system implements two levels of caching:
1. **JIRA Data Cache**: Stores raw ticket data to minimize API calls
   - Located in `data/jira_cache/`
   - Indexed by project key
   - Includes timestamps for incremental updates
   
2. **Embedding Cache**: Stores generated embeddings to reduce OpenAI API costs
   - Located in `data/embedding_cache/`
   - Indexed by text hash
   - Includes model information for version control

## Model Persistence

Trained models are saved in the `data/results` directory with appropriate extensions:
- Linear models: `.pkl` (using pickle)
- Random forest models: `.pkl` (using pickle)
- Neural networks: `.pt` (using PyTorch's save mechanism)

Each saved model includes metadata about its training parameters and performance metrics.
