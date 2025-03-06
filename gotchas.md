# JIRA Ticket Time Estimation: Gotchas and Warnings

This document outlines common issues, edge cases, and warnings for the JIRA Ticket Time Estimation system.

## Data Quality Issues

### Missing Completion Times
- Tickets without completion times (still open or improperly closed) are automatically filtered out
- If too many tickets are filtered, model accuracy may suffer
- Solution: Use the `--max-results` parameter to fetch more tickets initially

### Outlier Tickets
- Extremely long or short completion times can skew model predictions
- The system does not automatically remove outliers
- Solution: Consider using the `--exclude-labels` parameter to filter out known anomalous tickets

### Insufficient Data
- Models require sufficient historical data to make accurate predictions
- Rule of thumb: At least 50 completed tickets per project for basic models
- Neural networks may require 200+ tickets for good performance
- Solution: Combine similar projects or use simpler models when data is limited

## API Limitations

### JIRA API Rate Limits
- JIRA Cloud imposes rate limits that may affect large data fetches
- The system implements exponential backoff, but may still fail with very large requests
- Solution: Use incremental updates and caching to minimize API calls

### OpenAI API Costs
- Generating embeddings incurs costs based on token count
- Large tickets with extensive descriptions can be expensive to process
- Solution: The embedding cache reduces costs for repeated runs

## Model-Specific Issues

### Linear Regression Limitations
- Assumes linear relationship between features and completion time
- May perform poorly on complex projects with non-linear patterns
- Solution: Try random forest or neural models for complex projects

### Random Forest Overfitting
- Can overfit to training data with default parameters
- Solution: Use the tuning script to find optimal hyperparameters

### Neural Network Training
- Training may be unstable with small datasets
- Learning rate may need adjustment for different projects
- Solution: Start with simpler models and only use neural networks for large datasets

## Environment Setup

### API Credentials
- Missing or incorrect API credentials will cause failures
- JIRA API tokens expire and may need to be refreshed
- Solution: Always check `.env` file if authentication errors occur

### Python Dependencies
- Some dependencies may conflict in certain environments
- Solution: Use a dedicated virtual environment for this project

## Common Error Messages

### "No completed tickets found"
- Cause: Filtered data resulted in zero tickets with completion times
- Solution: Broaden project selection or reduce filtering criteria

### "Embedding model not available"
- Cause: Specified OpenAI model doesn't exist or isn't accessible
- Solution: Check OPENAI_API_KEY and use a supported model name

### "Insufficient data for cross-validation"
- Cause: Too few samples for the requested number of CV splits
- Solution: Reduce cv-splits parameter or collect more data

## Performance Considerations

### Large Projects
- Projects with thousands of tickets may cause memory issues
- Solution: Use the `--max-results` parameter to limit data size

### Embedding Generation
- Initial embedding generation can be slow for large datasets
- Solution: This is a one-time cost; subsequent runs will use the cache
