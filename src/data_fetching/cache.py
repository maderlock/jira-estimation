"""Module for caching JIRA data."""
import json
import logging
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
        """Load cache metadata."""
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return {}

    def _save_metadata(self, metadata: Dict) -> None:
        """Save cache metadata."""
        self.metadata_file.write_text(json.dumps(metadata, indent=2))

    def get_cached_data(self, query_hash: str) -> Optional[pd.DataFrame]:
        """
        Get cached data for a query.

        Args:
            query_hash: Hash of the query parameters

        Returns:
            DataFrame of cached data if found, None otherwise
        """
        cache_file = self.cache_dir / f"{query_hash}.parquet"
        if not cache_file.exists():
            return None
            
        metadata = self._load_metadata()
        if query_hash not in metadata:
            return None
            
        df = pd.read_parquet(cache_file)
        self.logger.info(f"Retrieved {len(df)} records from cache")
        return df

    def save_data(
        self,
        data: List[Dict],
        query_hash: str,
        query_params: Dict,
    ) -> None:
        """
        Save data to cache.

        Args:
            data: List of data dictionaries to cache
            query_hash: Hash of the query parameters
            query_params: Dictionary of query parameters
        """
        df = pd.DataFrame(data)
        cache_file = self.cache_dir / f"{query_hash}.parquet"
        df.to_parquet(cache_file)
        
        metadata = self._load_metadata()
        metadata[query_hash] = {
            'created': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'query_params': query_params,
            'num_records': len(df),
        }
        self._save_metadata(metadata)
        
        self.logger.info(f"Cached {len(df)} records")

    def load(
        self,
        cache_key: str,
        update_func: Optional[callable] = None,
        update_kwargs: Optional[Dict] = None,
        use_cache: bool = True,
        update_cache: bool = True,
        force_update: bool = False,
    ) -> pd.DataFrame:
        """
        Load data from cache, updating if necessary.

        Args:
            cache_key: Key to identify the cached data
            update_func: Function to call to update cache if needed
            update_kwargs: Keyword arguments for update function
            use_cache: Whether to use cached data at all
            update_cache: Whether to check for and fetch updates
            force_update: Whether to force a full update regardless of timestamps

        Returns:
            DataFrame containing cached data
        """
        df = pd.DataFrame()
        max_results = update_kwargs.get('max_results', 1000) if update_kwargs else 1000
        
        # If cache is disabled or force update is requested, fetch fresh data
        if not use_cache or force_update:
            if not update_func:
                self.logger.warning("Cache disabled but no update function provided")
                return df
                
            self.logger.info("Fetching fresh data" + (" (forced)" if force_update else ""))
            if update_kwargs is None:
                update_kwargs = {}
            df = update_func(**update_kwargs)
            
            if df is not None and len(df) > 0:
                self.save_data(df.to_dict('records'), cache_key, update_kwargs)
            return df
        
        # Try to load from cache
        df = self.get_cached_data(cache_key)
        if df is not None:
            self.logger.info(f"Retrieved {len(df)} records from cache")
            
            # Return cached data if updates are disabled
            if not update_cache:
                self.logger.info("Using cached data without updates")
                # Ensure we don't exceed max_results from cache
                if len(df) > max_results:
                    self.logger.debug(f"Trimming cached data to max_results={max_results}")
                    df = df.head(max_results)
                return df

            # Check for updates if update_func is provided
            if update_func:
                if update_kwargs is None:
                    update_kwargs = {}
                
                # Calculate remaining results to fetch
                remaining_results = max(0, max_results - len(df))
                
                # First, check for any new updates since last fetch
                metadata = self._load_metadata()
                last_update = metadata.get(cache_key, {}).get('created', "1970-01-01 00:00")
                
                update_kwargs_with_date = dict(update_kwargs)
                update_kwargs_with_date["updated_after"] = last_update
                update_kwargs_with_date["max_results"] = max_results
                
                new_df = update_func(**update_kwargs_with_date)
                if new_df is not None and len(new_df) > 0:
                    self.logger.info(f"Found {len(new_df)} new updated records")
                    df = pd.concat([df, new_df], ignore_index=True)
                    df = df.drop_duplicates(subset=["key"], keep="last")
                
                # If we still need more results, fetch without date filter
                remaining_results = max(0, max_results - len(df))
                if remaining_results > 0:
                    self.logger.info(f"Fetching {remaining_results} more records to reach max_results")
                    update_kwargs_no_date = dict(update_kwargs)
                    update_kwargs_no_date["max_results"] = remaining_results
                    update_kwargs_no_date.pop("updated_after", None)  # Remove date filter
                    
                    additional_df = update_func(**update_kwargs_no_date)
                    if additional_df is not None and len(additional_df) > 0:
                        self.logger.info(f"Found {len(additional_df)} additional records")
                        df = pd.concat([df, additional_df], ignore_index=True)
                        df = df.drop_duplicates(subset=["key"], keep="last")
                
                # Ensure combined results don't exceed max_results
                if len(df) > max_results:
                    self.logger.debug(f"Trimming combined data to max_results={max_results}")
                    df = df.head(max_results)
                    
                self.save_data(df.to_dict('records'), cache_key, update_kwargs)
        
        # If no cached data and updates allowed, fetch fresh
        elif update_func:
            self.logger.info("No cached data found, fetching fresh")
            if update_kwargs is None:
                update_kwargs = {}
            df = update_func(**update_kwargs)
            
            if df is not None and len(df) > 0:
                self.save_data(df.to_dict('records'), cache_key, update_kwargs)
        
        return df
