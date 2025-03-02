"""Module for processing and embedding text data."""
import logging
from typing import List, Optional
import re

import numpy as np
from openai import OpenAI, APIError
import tiktoken
from tqdm import tqdm
from httpx import Timeout

from utils import (
    EMBEDDING_MODELS,
    MAX_TOKENS,
    DEFAULT_EMBEDDING_MODEL,
    get_openai_api_key,
)

logger = logging.getLogger(__name__)


class OpenAIQuotaExceededError(Exception):
    """Raised when OpenAI API quota is exceeded."""
    pass


class TextProcessor:
    """Class to handle text processing and embedding generation."""

    def __init__(self, model: str = DEFAULT_EMBEDDING_MODEL):
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
        
        # Initialize OpenAI client with custom retry settings
        self.client = OpenAI(
            api_key=get_openai_api_key(),
            max_retries=0,  # Disable retries for quota errors
            timeout=Timeout(30.0, read=300.0),  # 30s connect, 300s read
        )
        
        logger.info(f"Initialized TextProcessor with model: {model}")
        logger.debug(f"Token limit: {self.token_limit}")

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

        Raises:
            OpenAIQuotaExceededError: If the OpenAI API quota is exceeded
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
            embeddings = self.get_embeddings(batch)
            all_embeddings.extend(embeddings)
            
        return np.array(all_embeddings)

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limit."""
        if not text:
            return ""
            
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.token_limit:
            return text
        
        logger.debug(f"Truncating text from {len(tokens)} tokens to {self.token_limit}")
        return self.tokenizer.decode(tokens[:self.token_limit])

    def get_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of texts to get embeddings for
            batch_size: Number of texts to process at once

        Returns:
            Array of embeddings

        Raises:
            OpenAIQuotaExceededError: If the OpenAI API quota is exceeded
            APIError: For other OpenAI API errors
        """
        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model}")
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1} of {(len(texts)-1)//batch_size + 1}")
            
            # Truncate texts to fit token limit
            processed_texts = [self._truncate_text(text) for text in batch]
            
            try:
                response = self.client.embeddings.create(
                    input=processed_texts,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.debug(f"Successfully processed {len(batch)} texts in current batch")
            except APIError as e:
                if "insufficient_quota" in str(e):
                    logger.error("OpenAI API quota exceeded")
                    raise OpenAIQuotaExceededError("OpenAI API quota exceeded. Please check your usage and limits.") from e
                logger.error(f"OpenAI API error: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error generating embeddings: {str(e)}")
                raise
        
        logger.info("Finished generating embeddings")
        return np.array(embeddings)

    @staticmethod
    def strip_formatting(text: str) -> str:
        """
        Strip out unnecessary markdown formatting from text while preserving meaningful elements.
        
        Removes:
            - Headers (#, ##)
            - Bold (**text**)
            - Italics (_text_, *text*)
            - Table formatting (| column | value |)
            - HTML tags (<tag>text</tag>)
            
        Preserves:
            - Lists (-, *, 1.) as they show relationships
            - Newlines as they separate concepts
            - Blockquotes (>) as they indicate emphasis
            - Inline code as it may be critical in tech content
            
        Args:
            text: Text to strip formatting from
            
        Returns:
            Text with unnecessary formatting removed
        """
        if not text:
            return ""
        
        # Store code blocks to preserve them
        code_blocks = {}
        code_block_counter = 0
        
        def store_code_block(match):
            nonlocal code_block_counter
            code_block_counter += 1
            key = f"__CODE_BLOCK_{code_block_counter}__"
            code_blocks[key] = match.group(0)
            return key
            
        # Temporarily store code blocks (both ``` and indented)
        text = re.sub(r'```[\s\S]*?```', store_code_block, text)
        text = re.sub(r'(?m)^    .*$', store_code_block, text)
        
        # Remove headers using "h\d."
        text = re.sub(r'^h\d.\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold and italics
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'__(.+?)__', r'\1', text)      # __bold__
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # *italic*
        text = re.sub(r'_(.+?)_', r'\1', text)        # _italic_
        
        # Remove table formatting
        text = re.sub(r'\|.*\|', lambda m: m.group(0).replace('|', ' '), text)  # Keep content between pipes
        text = re.sub(r'[-]+[|][-]+', '', text)  # Remove table lines
        
        # Remove HTML tags but keep content
        text = re.sub(r'<[^>]+>', '', text)

        # Remove colors e.g. {color:#xxxxxx} / {color}
        text = re.sub(r'\{color[^}]*\}', '', text)

        # Strip links to leave just link title in format [link title|href]
        text = re.sub(r'\[([^|]+)\|([^\]]+)\]', r'\1', text)

        # Clean up extra whitespace while preserving meaningful newlines
        # Replace 3+ newlines with 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Restore code blocks
        for key, value in code_blocks.items():
            text = text.replace(key, value)
            
        # Final cleanup
        # Remove leading/trailing whitespace from each line while preserving indentation
        lines = [line.rstrip() for line in text.splitlines()]
        text = '\n'.join(lines)
        
        # Remove leading/trailing whitespace from entire text
        return text.strip()
