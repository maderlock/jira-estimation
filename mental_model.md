# JIRA Ticket Time Estimation: Mental Model

This document provides a conceptual overview of the JIRA Ticket Time Estimation system.

## Core Concept

The JIRA Ticket Time Estimation system uses machine learning to predict how long it will take to complete JIRA tickets. It does this by analyzing historical ticket data and learning patterns between ticket descriptions and their actual completion times.

## Key Components

### 1. Data Collection
The system fetches historical ticket data from JIRA, including descriptions, titles, and actual completion times. This data forms the foundation for training the prediction models.

### 2. Text Processing
Ticket text (descriptions, titles, etc.) is processed and converted into numerical embeddings using OpenAI's API. These embeddings capture the semantic meaning of the text in a format that machine learning models can understand.

### 3. Model Training
The system supports multiple types of prediction models:
- Linear Regression: A simple model that assumes a linear relationship between features and completion time
- Random Forest: An ensemble model that can capture non-linear relationships
- Neural Network: A deep learning approach for complex pattern recognition

### 4. Prediction
Once trained, the models can predict completion times for new tickets based on their text content.

## Conceptual Flow

1. Historical ticket data is fetched from JIRA
2. Text from tickets is converted to numerical embeddings
3. Models are trained on this data to recognize patterns
4. New tickets are processed in the same way
5. The trained model predicts completion time for new tickets

## Design Philosophy

The system is designed with several key principles in mind:
- **Efficiency**: Data caching and incremental updates minimize API calls
- **Flexibility**: Support for multiple model types allows for different approaches
- **Accuracy**: Cross-validation and multiple metrics ensure reliable predictions
- **Usability**: Command-line interface with sensible defaults makes the system easy to use
- **Extensibility**: Abstract interfaces allow for easy addition of new implementations

## Component Architecture

The system follows a modular architecture with clear separation of concerns:

### JiraAI (Main Orchestrator)
The central orchestrator that coordinates the entire workflow, from data fetching to model training and evaluation.

### TicketFetcher (Interface)
Defines the contract for fetching ticket data. The current implementation uses JIRA's API, but this could be replaced with other ticket systems.

### AbstractTextProcessor (Interface)
Defines the contract for processing text data. The current implementation (AITextProcessor) uses OpenAI's API for embeddings, but this could be replaced with other embedding techniques.

### ModelLearner
Handles the training and evaluation of machine learning models, with support for different algorithms and cross-validation.

## When to Use Each Model Type

- **Linear Regression**: Best for simple datasets with clear linear relationships. Fastest to train and easiest to interpret.
- **Random Forest**: Good for capturing non-linear relationships and handling outliers. More robust than linear regression.
- **Neural Network**: Best for large datasets with complex patterns. Requires more data and tuning but can achieve higher accuracy.

## Hyperparameter Tuning

Machine learning models often have hyperparameters that need to be tuned for optimal performance. The system includes a dedicated tuning script for Random Forest models using Optuna, which can:

1. Automatically search for optimal hyperparameter combinations
2. Evaluate model performance using cross-validation
3. Visualize the optimization process
4. Provide insights into parameter importance

## Cross-Validation

Cross-validation is a technique used to assess how well a model will generalize to new data. The system supports cross-validation for all model types, which:

1. Divides the data into multiple folds
2. Trains and evaluates the model on different combinations of these folds
3. Provides a more robust estimate of model performance
4. Helps detect overfitting

## Caching Strategy

The system implements a two-level caching strategy:

1. **JIRA Data Cache**: Stores raw ticket data to minimize API calls
2. **Embedding Cache**: Stores generated embeddings to reduce OpenAI API costs

This approach significantly reduces the time and cost of running the system repeatedly.
