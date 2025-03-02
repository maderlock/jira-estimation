"""Module for caching embeddings."""
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid

import numpy as np
import pandas as pd


class EmbeddingCache:
    """Class to handle caching of text embeddings."""

    def __init__(
        self,
        cache_dir: str,
        model: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Base directory for cache storage
            model: Name of the embedding model being used
            logger: Optional logger instance
        """
        self.model = model
        self.cache_dir = Path(cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.debug(f"Initialized embedding cache in {self.cache_dir}")

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

    def _load_metadata(self) -> Dict:
        """Load embedding cache metadata."""
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return {}

    def _save_metadata(self, metadata: Dict) -> None:
        """Save embedding cache metadata."""
        self.metadata_file.write_text(json.dumps(metadata, indent=2))

    def get_cached_embeddings(self, doc_ids: List[str]) -> Tuple[List[np.ndarray], List[int]]:
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

    def save_embeddings(
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
        metadata = self._load_metadata()
        
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
            
        self._save_metadata(metadata)
