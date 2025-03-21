# JIRA Ticket Time Estimation

Machine learning system to predict JIRA ticket completion times using embeddings and regression techniques.

## Overview

This project provides a machine learning-based solution for automatically estimating the time required to complete JIRA tickets. By analyzing historical ticket data and leveraging text embeddings, the system can predict completion times for new tickets, helping teams with planning and resource allocation.

## Features

- Automatic time estimation for JIRA tickets using ML
- Support for both linear regression, random forest, and neural network models
- Text embedding generation using OpenAI's API
- Efficient data caching and incremental updates
- Cross-validation support for all model types
- Configurable data filtering and processing
- Hyperparameter tuning for Random Forest models using Optuna

## Documentation

For detailed information about the project, please refer to the following documentation:

- [Mental Model](mental_model.md) - Conceptual overview of the system
- [Implementation Details](implementation_details.md) - Technical implementation specifics
- [Gotchas and Warnings](gotchas.md) - Common issues and edge cases
- [Quick Reference](quick_reference.md) - Parameters, configurations, and common commands

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

3. Set up environment variables by copying the example file:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` to add your JIRA and OpenAI API credentials.

## Usage

Basic usage:
```bash
python -m src.main --project-keys PROJECT1 PROJECT2
```

For more options, see the [Quick Reference](quick_reference.md) document.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
