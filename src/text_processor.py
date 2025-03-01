"""Module for processing and embedding text data."""
from typing import List

import numpy as np
import openai
import tiktoken
from pandas import DataFrame, Series

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY, TOKEN_LIMIT

openai.api_key = OPENAI_API_KEY


class TextProcessor:
    """Class to handle text processing and embedding generation."""

    def __init__(self):
        """Initialize the tokenizer."""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def process_ticket_text(self, df: DataFrame) -> Series:
        """
        Process ticket text by combining and truncating fields.

        Args:
            df: DataFrame containing ticket data.

        Returns:
            Series containing processed text.
        """
        return df.apply(
            lambda row: self._truncate_text(
                f"{row['summary']} {row['description']} {row['comments']}"
            ),
            axis=1,
        )

    def _truncate_text(self, text: str) -> str:
        """Truncate text to token limit."""
        tokens = self.tokenizer.encode(text)
        return self.tokenizer.decode(tokens[:TOKEN_LIMIT])

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings for.

        Returns:
            Array of embeddings.
        """
        # Process in batches to avoid rate limits
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = openai.Embedding.create(input=batch, model=EMBEDDING_MODEL)
            batch_embeddings = [item["embedding"] for item in response["data"]]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def process_and_embed(self, df: DataFrame) -> np.ndarray:
        """
        Process ticket text and generate embeddings.

        Args:
            df: DataFrame containing ticket data.

        Returns:
            Array of embeddings.
        """
        processed_texts = self.process_ticket_text(df)
        return self.get_embeddings(processed_texts.tolist())
