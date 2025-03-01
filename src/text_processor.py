"""Module for processing and embedding text data."""
from typing import List, Optional

import numpy as np
import openai
import tiktoken
from tqdm import tqdm

from src.utils import EMBEDDING_MODELS, MAX_TOKENS


class TextProcessor:
    """Class to handle text processing and embedding generation."""

    def __init__(self, model: str = "ada"):
        """
        Initialize the text processor.

        Args:
            model: Embedding model to use, must be one of the keys in EMBEDDING_MODELS
        """
        if model not in EMBEDDING_MODELS:
            raise ValueError(f"Model must be one of: {list(EMBEDDING_MODELS.keys())}")
            
        self.model = EMBEDDING_MODELS[model]
        self.token_limit = MAX_TOKENS[self.model]
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def process_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Process and embed a batch of texts.

        Args:
            texts: List of texts to process and embed
            batch_size: Size of batches for embedding generation
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings
        """
        # Truncate texts
        processed_texts = [self._truncate_text(text) for text in texts]
        
        # Generate embeddings in batches
        all_embeddings = []
        iterator = range(0, len(processed_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
            
        for i in iterator:
            batch = processed_texts[i:i + batch_size]
            embeddings = self._get_embeddings(batch)
            all_embeddings.extend(embeddings)
            
        return np.array(all_embeddings)

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to token limit.

        Args:
            text: Text to truncate

        Returns:
            Truncated text
        """
        if not text:
            return ""
            
        tokens = self.tokenizer.encode(text)
        return self.tokenizer.decode(tokens[:self.token_limit])

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of embeddings
        """
        response = openai.Embedding.create(
            model=self.model,
            input=texts,
        )
        return [data["embedding"] for data in response["data"]]
