# JIRA Ticket Time Estimation

Machine learning system to predict JIRA ticket completion times using embeddings and regression techniques.

## Overview

This project provides a machine learning-based solution for automatically estimating the time required to complete JIRA tickets. By analyzing historical ticket data and leveraging text embeddings, the system can predict completion times for new tickets, helping teams with planning and resource allocation.

## Features

- Automatic time estimation for JIRA tickets using ML
- Support for both linear regression, random forest, and neural network models
- Text embedding generation using OpenAI's API
- Efficient data caching and incremental updates
- Cross-validation support for model evaluation
- Configurable data filtering and processing

## Documentation

For detailed information about the project, please refer to the following documentation:

- [Mental Model](mental_model.md) - Conceptual overview of the system
- [Implementation Details](implementation_details.md) - Technical implementation specifics
- [Gotchas and Warnings](gotchas.md) - Common issues and edge cases
- [Quick Reference](quick_reference.md) - Parameters, configurations, and common commands

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

See the [Quick Reference](quick_reference.md) for details on required environment variables.

## Basic Usage

Train a model with default settings:
```bash
python src/main.py --project-keys PROJECT1 PROJECT2
```

For more detailed usage examples and options, refer to the [Quick Reference](quick_reference.md) documentation.
