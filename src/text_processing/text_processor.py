"""Module for processing and embedding text data."""
import logging
import re
from typing import Dict, List, Optional
import pandas as pd

import numpy as np
from openai import OpenAI, APIError
import tiktoken
from httpx import Timeout
from tqdm import tqdm

from .constants import EMBEDDING_MODELS, MAX_TOKENS, DEFAULT_EMBEDDING_MODEL
from .exceptions import OpenAIQuotaExceededError
from .embedding_cache import EmbeddingCache
from abc import abstractmethod, ABC

class AbstractTextProcessor(ABC):
    @abstractmethod
    def process_batch(
        self,
        texts: List[str],
        queries: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 100,
        chunk_overlap: int = 100,
        temperature: float = 1.0,
        show_progress: bool = True,
    ) -> np.ndarray: ...

    @staticmethod
    def strip_formatting(text: str) -> str:
        """
        Strip out unnecessary markdown formatting from text while preserving meaningful elements.
        
        Removes:
            - Headers (h1., h2.)
            - Bold (**text**)
            - Italics (_text_, *text*)
            - Table formatting (| column | value |)
            - HTML tags (<tag>text</tag>)
            - JIRA colors ({color:#xxxxxx})
            
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


class AITextProcessor(AbstractTextProcessor):
    """Class to handle text processing and embedding generation."""

    def __init__(
        self,
        openai_api_key: str,
        cache_dir: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the text processor.

        Args:
            openai_api_key: OpenAI API key for authentication
            cache_dir: Directory to store embeddings cache
            model: Embedding model to use, must be one of the keys in EMBEDDING_MODELS
            logger: Optional logger instance
        """
        if model not in EMBEDDING_MODELS:
            raise ValueError(f"Model must be one of: {list(EMBEDDING_MODELS.keys())}")
            
        self.model = EMBEDDING_MODELS[model]
        self.token_limit = MAX_TOKENS[self.model]
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize OpenAI client with custom retry settings
        self.client = OpenAI(
            api_key=openai_api_key,
            max_retries=0,  # Disable retries for quota errors
            timeout=Timeout(30.0, read=300.0),  # 30s connect, 300s read
        )
        
        # Initialize embedding cache
        self.cache = EmbeddingCache(cache_dir, model, logger=self.logger)
        
        self.logger.info(f"Initialized TextProcessor with model: {model}")
        self.logger.debug(f"Token limit: {self.token_limit}")

    def process_batch(
        self,
        texts: List[str],
        queries: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 100,
        chunk_overlap: int = 100,
        temperature: float = 1.0,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Process and embed a batch of texts using chunking and attention.

        Args:
            texts: List of texts to process and embed
            queries: List of query texts (e.g. ticket summaries) for attention
            metadata: Optional list of metadata dicts for each text
            batch_size: Size of batches for embedding generation
            chunk_overlap: Token overlap between chunks
            temperature: Temperature for attention mechanism
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings
        """
        if len(texts) != len(queries):
            raise ValueError("Number of texts and queries must match")
            
        # First get all query embeddings in one batch
        self.logger.info("Getting query embeddings")
        query_embeddings = self._get_embeddings(queries, metadata=metadata)
        
        # Process all texts to get chunks
        self.logger.info("Chunking texts")
        all_chunks = []
        chunk_indices = []  # Track which chunks belong to which text
        all_chunk_metadata = []
        
        for i, text in enumerate(texts):
            chunks = self.chunk_text(text, overlap=chunk_overlap)
            if not chunks:
                # Empty text, use query directly
                all_chunks.append(queries[i])
                chunk_indices.append([len(all_chunks) - 1])
                all_chunk_metadata.append(metadata[i] if metadata else None)
                continue
                
            start_idx = len(all_chunks)
            all_chunks.extend(chunks)
            chunk_indices.append(list(range(start_idx, len(all_chunks))))
            
            if metadata:
                # Create metadata for each chunk
                chunk_meta = [
                    {**metadata[i], 'chunk_index': j, 'total_chunks': len(chunks)}
                    for j in range(len(chunks))
                ]
                all_chunk_metadata.extend(chunk_meta)
        
        # Get embeddings for all chunks in batches
        self.logger.info(f"Getting embeddings for {len(all_chunks)} chunks")
        iterator = range(0, len(all_chunks), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing chunks")
            
        all_embeddings = []
        for i in iterator:
            batch_chunks = all_chunks[i:i + batch_size]
            batch_meta = all_chunk_metadata[i:i + batch_size] if all_chunk_metadata else None
            embeddings = self._get_embeddings(batch_chunks, metadata=batch_meta)
            all_embeddings.extend(embeddings)
            
        # Combine chunks using attention
        self.logger.info("Combining chunks with attention")
        final_embeddings = []
        for i, indices in enumerate(chunk_indices):
            chunk_embs = [all_embeddings[j] for j in indices]
            if len(chunk_embs) == 1:
                final_embeddings.append(chunk_embs[0])
            else:
                combined = self._combine_embeddings_with_attention(
                    chunk_embs,
                    query_embeddings[i],
                    temperature=temperature
                )
                final_embeddings.append(combined)
                
        return np.array(final_embeddings)

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limit."""
        if not text:
            return ""
            
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.token_limit:
            return text
            
        # Decode truncated tokens back to text
        return self.tokenizer.decode(tokens[:self.token_limit])

    def chunk_text(self, text: str, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks that fit within token limit.
        
        Args:
            text: Text to split into chunks
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.token_limit:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Get chunk that fits within token limit
            end = start + self.token_limit
            
            # Decode chunk back to text
            chunk = self.tokenizer.decode(tokens[start:end])
            chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - overlap
            
        return chunks

    def _combine_embeddings_with_attention(
        self, 
        embeddings: List[np.ndarray], 
        query_embedding: np.ndarray,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Combine multiple embeddings using attention mechanism.
        
        Args:
            embeddings: List of embeddings to combine
            query_embedding: Query embedding to calculate attention scores against
            temperature: Temperature for softmax (higher = more uniform weights)
            
        Returns:
            Combined embedding
        """
        if not embeddings:
            raise ValueError("No embeddings to combine")
        if len(embeddings) == 1:
            return embeddings[0]
            
        # Calculate cosine similarities
        similarities = []
        query_norm = np.linalg.norm(query_embedding)
        
        for emb in embeddings:
            similarity = np.dot(emb, query_embedding) / (np.linalg.norm(emb) * query_norm)
            similarities.append(similarity)
        
        # Apply temperature scaling and softmax
        scaled_similarities = np.array(similarities) / temperature
        weights = np.exp(scaled_similarities) / np.sum(np.exp(scaled_similarities))
        
        # Weighted combination
        stacked = np.stack(embeddings)
        combined = np.average(stacked, axis=0, weights=weights)
        
        self.logger.debug(f"Combined {len(embeddings)} embeddings with attention weights: {weights}")
        return combined

    def _get_embeddings(
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
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        if not texts:
            return []
            
        # Generate document IDs if metadata provided
        doc_ids = None
        if metadata:
            doc_ids = [
                self.cache._generate_document_id(meta['key'], meta['summary'])
                for meta in metadata
            ]
            
            # Try to get embeddings from cache
            cached_embeddings, missing_indices = self.cache.get_cached_embeddings(doc_ids)
            
            # If all embeddings are cached, return them
            if not missing_indices:
                self.logger.info("All embeddings found in cache")
                return [emb for emb in cached_embeddings if emb is not None]
                
            # Get texts that need embedding
            texts_to_embed = [texts[i] for i in missing_indices]
            self.logger.info(f"Generating embeddings for {len(texts_to_embed)} texts")
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
                self.cache.save_embeddings(new_embeddings, new_doc_ids, new_texts)
            
            # Merge cached and new embeddings
            for new_idx, cache_idx in enumerate(missing_indices):
                cached_embeddings[cache_idx] = new_embeddings[new_idx]
                
            return [emb for emb in cached_embeddings if emb is not None]
            
        except APIError as e:
            if "insufficient_quota" in str(e):
                self.logger.error("OpenAI API quota exceeded")
                raise OpenAIQuotaExceededError("OpenAI API quota exceeded. Please check your usage and limits.") from e
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error generating embeddings: {str(e)}")
            raise
