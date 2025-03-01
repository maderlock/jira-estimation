# JIRA Estimation AI

A machine learning system for estimating JIRA ticket completion times based on historical data using embeddings and regression models.

## Features

- Fetches historical JIRA ticket data including descriptions and comments
- Generates embeddings using OpenAI's API
- Supports both linear regression and neural network models
- Efficient text processing with chunking to minimize API costs
- Smart caching system for rapid experimentation
- Prepared for AWS Lambda deployment (future)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd jira-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```bash
cp .env.example .env
```
Then edit `.env` with your actual credentials and configuration.

## Usage

### Step 1: Data Collection

First, you'll need to fetch data from JIRA. The system includes smart caching to avoid unnecessary API calls.

```bash
# Fetch all completed tickets (uses cache by default)
python -m src.main --fetch

# Fetch fresh data, ignoring cache
python -m src.main --fetch --no-cache

# Fetch from specific projects
python -m src.main --fetch --projects PROJ1 PROJ2

# Exclude certain types of tickets
python -m src.main --fetch --exclude-labels invalid wontfix

# Limit the number of tickets
python -m src.main --fetch --max-results 1000
```

The fetched data is stored in:
- Raw JIRA data: `data/jira_cache/` (cached by query parameters)
- Processed data: `data/processed_tickets.pkl`
- Generated embeddings: `data/embeddings.pkl`

### Step 2: Training Models

Once you have collected your data, you can train different types of models:

```bash
# Train a linear regression model
python -m src.main --train linear

# Train a neural network model
python -m src.main --train neural
```

Models are saved in the `models/` directory with names like `jira_estimator_linear.pt`.

### Step 3: Experimentation

For running multiple experiments:

1. **Data Preparation**:
   - Use different project combinations:
     ```bash
     python -m src.main --fetch --projects PROJ1 PROJ2
     python -m src.main --train linear
     ```
   - Filter out specific tickets:
     ```bash
     python -m src.main --fetch --exclude-labels maintenance bug
     python -m src.main --train linear
     ```

2. **Model Comparison**:
   - Train both models on the same dataset:
     ```bash
     # Use --no-cache-update to ensure same data for both models
     python -m src.main --fetch --no-cache-update
     python -m src.main --train linear
     python -m src.main --train neural
     ```

3. **Data Size Impact**:
   ```bash
   # Try different dataset sizes
   python -m src.main --fetch --max-results 500
   python -m src.main --train linear
   
   python -m src.main --fetch --max-results 2000
   python -m src.main --train linear
   ```

### Caching Behavior

- Data is cached based on query parameters (projects, labels, etc.)
- Each unique combination gets its own cache file
- Cache is automatically updated with new tickets unless `--no-cache-update` is used
- Use `--no-cache` to fetch fresh data and ignore cache

## Project Structure

```
jira-ai/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── data_fetcher.py    # JIRA data fetching
│   ├── text_processor.py  # Text processing and embeddings
│   ├── model.py          # ML models
│   └── main.py           # Main script
├── data/
│   ├── jira_cache/       # Cached JIRA data
│   ├── processed_tickets.pkl  # Current experiment data
│   └── embeddings.pkl    # Current experiment embeddings
├── models/              # Trained models
├── tests/              # Test files
├── notebooks/          # Jupyter notebooks
├── requirements.txt    # Python dependencies
├── .env.example       # Example environment variables
└── README.md         # Project documentation
```

## Future Enhancements

1. AWS Lambda Deployment
   - Push trained models to S3
   - Deploy prediction endpoint as Lambda function
   - Set up API Gateway

2. Model Improvements
   - Experiment with different embedding models
   - Add support for more ML architectures
   - Implement cross-validation

3. Monitoring
   - Add logging and metrics
   - Set up model performance monitoring
   - Track embedding API usage