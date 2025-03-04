"""Module for caching JIRA data."""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class DataCache:
    """Class to handle caching of JIRA data."""

    def __init__(
        self,
        cache_dir: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Base directory for cache storage
            logger: Optional logger instance
        """
        self.cache_dir = Path(cache_dir) / "jira_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize metadata file if it doesn't exist
        if not self.metadata_file.exists():
            self._save_metadata({})
        
        self.logger.debug(f"Initialized data cache in {self.cache_dir}")

    def _load_metadata(self) -> Dict:
        """Load metadata from file."""
        if not self.metadata_file.exists():
            return {}
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            return {}

    def _save_metadata(self, metadata: Dict) -> None:
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")

    def get_cached_data(self, query_hash: str) -> pd.DataFrame:
        """
        Get cached data for a query.

        Args:
            query_hash: Hash of the query parameters

        Returns:
            DataFrame of cached data if found, None otherwise
        """
        cache_file = self.cache_dir / f"{query_hash}.parquet"
        if not cache_file.exists():
            return pd.DataFrame()
            
        metadata = self._load_metadata()
        if query_hash not in metadata:
            return pd.DataFrame()
            
        df = pd.read_parquet(cache_file)
        self.logger.info(f"Retrieved {len(df)} records from cache")
        return df

    def save_data(self, data: List[Dict], cache_key: str, metadata: Optional[Dict] = None) -> None:
        """Save data and metadata to cache."""
        # Save data
        df = pd.DataFrame(data) if isinstance(data, list) else data
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)
        
        # Save metadata
        if metadata:
            global_metadata = self._load_metadata()
            global_metadata[cache_key] = {
                'created': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'query_params': metadata,
                'num_records': len(df),
                'max_results': metadata.get('max_results', 1000),
            }
            self._save_metadata(global_metadata)
        
        self.logger.info(f"Cached {len(df)} records")

    def get_metadata(self, cache_key: str) -> Optional[Dict]:
        """Get metadata for cached data."""
        metadata = self._load_metadata()
        return metadata.get(cache_key)

    def load(
        self,
        cache_key: str,
        update_func: Optional[callable] = None,
        update_kwargs: Optional[Dict] = None,
        use_cache: bool = True,
        force_update: bool = False,
    ) -> pd.DataFrame:
        """
        Load data from cache, updating if necessary.

        Args:
            cache_key: Key to identify the cached data
            update_func: Function to call to update cache if needed
            update_kwargs: Keyword arguments for update function
            use_cache: Whether to use cached data at all
            force_update: Whether to force a full update of cache

        Returns:
            DataFrame containing cached data
        """
        df = pd.DataFrame()
        update_kwargs = update_kwargs or {}
        max_results = update_kwargs.get('max_results', 1000)
        
        # If cache is disabled or force update is requested, fetch fresh data
        if not use_cache or force_update:
            if not update_func:
                self.logger.warning("Cache disabled but no update function provided")
                return df
            df = update_func(**update_kwargs)
            if df is not None and len(df) > 0 and use_cache:
                self.save_data(df.to_dict('records'), cache_key, update_kwargs)
            return df if df is not None else pd.DataFrame()
        
        # Try to load from cache
        cached_data = self.get_cached_data(cache_key)
        cached_metadata = self.get_metadata(cache_key)
        cached_max_results = cached_metadata.get('max_results', 0) if cached_metadata else 0
        
        # If no cached data or insufficient results, fetch fresh
        if len(cached_data) == 0:
            self.logger.info("Cache miss, fetching fresh data")
            fresh_df = update_func(**update_kwargs)
            if fresh_df is not None and len(fresh_df) > 0:
                df = self._update_cache_minimally(pd.DataFrame(), fresh_df, cache_key, update_kwargs)
            df = df if df is not None else pd.DataFrame()
        else:
            df = cached_data
            # Only fetch fresh if we need more results and our max_results is larger
            if len(df) < max_results and max_results > cached_max_results:
                self.logger.info(f"Insufficient data in cache (cached={len(df)}, requested={max_results})")
                fresh_df = update_func(**update_kwargs)
                # Update current cache by adding in any tickets that are not currently in the cache
                df = self._update_cache_minimally(df, fresh_df, cache_key, update_kwargs)

            else:
                # Respect original cached max_results
                df = df.head(cached_max_results)
        
        self.logger.info(f"Retrieved {len(df)} records")
        return df

    def _update_cache_minimally(self, existing_cache, fresh_cache, cache_key, update_kwargs) -> pd.DataFrame:
        """Update cache with fresh data, avoiding duplicates."""
        if fresh_cache is None or len(fresh_cache) == 0:
            return existing_cache
            
        # Concatenate and remove duplicates based on key
        df = pd.concat([existing_cache, fresh_cache], ignore_index=True)
        if len(df) > 0:
            df = df.drop_duplicates(subset=['key'], keep='last')
            
        self.save_data(df.to_dict('records'), cache_key, update_kwargs)
        return df
