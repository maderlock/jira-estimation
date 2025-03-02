"""Constants for text processing."""

# OpenAI embedding models and their token limits
EMBEDDING_MODELS = {
    "ada": "text-embedding-ada-002",
    # Add other models as needed
}

MAX_TOKENS = {
    "text-embedding-ada-002": 8191,
    # Add other models' limits as needed
}

DEFAULT_EMBEDDING_MODEL = "ada"
