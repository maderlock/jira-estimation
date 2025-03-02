"""Module for processing and embedding text data."""
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from openai import OpenAI, APIError
import tiktoken
from tqdm import tqdm
from httpx import Timeout

from utils import (
    EMBEDDING_MODELS,
    MAX_TOKENS,
    DEFAULT_EMBEDDING_MODEL,
    get_openai_api_key,
    DATA_DIR,
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
        
        # Setup embedding cache
        self.cache_dir = Path(DATA_DIR) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        
        logger.info(f"Initialized TextProcessor with model: {model}")
        logger.debug(f"Token limit: {self.token_limit}")
        logger.debug(f"Cache directory: {self.cache_dir}")

    def process_batch(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Process and embed a batch of texts.

        Args:
            texts: List of texts to process and embed
            metadata: Optional list of metadata dicts with 'key' and 'summary' for each text
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
            batch_texts = processed_texts[i:i + batch_size]
            batch_meta = metadata[i:i + batch_size] if metadata else None
            embeddings = self.get_embeddings(batch_texts, batch_meta)
            all_embeddings.extend(embeddings)
            
        return np.array(all_embeddings)

    def _generate_document_id(self, key: str, summary: str) -> str:
        """
        Generate a stable UUID for a document based on key and summary.
        
        Args:
            key: Document key (e.g. JIRA ticket key)
            summary: Document summary/title
            
        Returns:
            UUID string
        """
        # Create a stable hash from key and summary
        content = f"{key}:{summary}"
        content_hash = hashlib.sha256(content.encode()).digest()
        # Generate a UUID using the hash as namespace
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, content))

    def _load_cache_metadata(self) -> Dict:
        """Load embedding cache metadata."""
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return {}

    def _save_cache_metadata(self, metadata: Dict) -> None:
        """Save embedding cache metadata."""
        self.metadata_file.write_text(json.dumps(metadata, indent=2))

    def _get_cached_embeddings(self, doc_ids: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Get embeddings from cache for given document IDs.
        
        Args:
            doc_ids: List of document IDs to fetch
            
        Returns:
            Tuple of (cached embeddings, indices of missing embeddings)
        """
        cached_embeddings = []
        missing_indices = []
        
        for i, doc_id in enumerate(doc_ids):
            cache_file = self.cache_dir / f"{doc_id}.parquet"
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                cached_embeddings.append(df['embedding'].values[0])
            else:
                cached_embeddings.append(None)
                missing_indices.append(i)
                
        return cached_embeddings, missing_indices

    def _save_embeddings(
        self,
        embeddings: List[np.ndarray],
        doc_ids: List[str],
        texts: List[str],
    ) -> None:
        """
        Save embeddings to cache.
        
        Args:
            embeddings: List of embedding arrays
            doc_ids: List of document IDs
            texts: List of original texts
        """
        metadata = self._load_cache_metadata()
        
        for emb, doc_id, text in zip(embeddings, doc_ids, texts):
            # Save embedding
            cache_file = self.cache_dir / f"{doc_id}.parquet"
            df = pd.DataFrame({
                'doc_id': [doc_id],
                'text': [text],
                'embedding': [emb],
                'model': [self.model],
            })
            df.to_parquet(cache_file)
            
            # Update metadata
            metadata[doc_id] = {
                'model': self.model,
                'created': datetime.now().isoformat(),
            }
            
        self._save_cache_metadata(metadata)

    def get_embeddings(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> List[np.ndarray]:
        """
        Get embeddings for a list of texts, using cache when possible.

        Args:
            texts: List of texts to get embeddings for
            metadata: Optional list of metadata dicts with 'key' and 'summary' for each text

        Returns:
            List of embedding arrays

        Raises:
            OpenAIQuotaExceededError: If the OpenAI API quota is exceeded
            APIError: For other OpenAI API errors
        """
        if not texts:
            return []
            
        # Generate document IDs if metadata provided
        doc_ids = None
        if metadata:
            doc_ids = [
                self._generate_document_id(meta['key'], meta['summary'])
                for meta in metadata
            ]
            
            # Try to get embeddings from cache
            cached_embeddings, missing_indices = self._get_cached_embeddings(doc_ids)
            
            # If all embeddings are cached, return them
            if not missing_indices:
                logger.info("All embeddings found in cache")
                return [emb for emb in cached_embeddings if emb is not None]
                
            # Get texts that need embedding
            texts_to_embed = [texts[i] for i in missing_indices]
            logger.info(f"Generating embeddings for {len(texts_to_embed)} texts")
        else:
            texts_to_embed = texts
            missing_indices = list(range(len(texts)))
            cached_embeddings = [None] * len(texts)
        
        # Process texts that need embedding
        processed_texts = [self._truncate_text(text) for text in texts_to_embed]
        
        try:
            response = self.client.embeddings.create(
                input=processed_texts,
                model=self.model
            )
            new_embeddings = [item.embedding for item in response.data]
            
            # Save new embeddings to cache if we have document IDs
            if doc_ids:
                new_doc_ids = [doc_ids[i] for i in missing_indices]
                new_texts = [texts[i] for i in missing_indices]
                self._save_embeddings(new_embeddings, new_doc_ids, new_texts)
            
            # Merge cached and new embeddings
            for new_idx, cache_idx in enumerate(missing_indices):
                cached_embeddings[cache_idx] = new_embeddings[new_idx]
                
            return [emb for emb in cached_embeddings if emb is not None]
            
        except APIError as e:
            if "insufficient_quota" in str(e):
                logger.error("OpenAI API quota exceeded")
                raise OpenAIQuotaExceededError("OpenAI API quota exceeded. Please check your usage and limits.") from e
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {str(e)}")
            raise

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limit."""
        if not text:
            return ""
            
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.token_limit:
            return text
        
        logger.debug(f"Truncating text from {len(tokens)} tokens to {self.token_limit}")
        return self.tokenizer.decode(tokens[:self.token_limit])

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
