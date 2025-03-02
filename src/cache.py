"""Module for handling data caching."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from utils import DATA_DIR

logger = logging.getLogger(__name__)


class DataCache:
    """Class to handle data caching with metadata tracking."""

    def __init__(self, cache_dir: str):
        """
        Initialize the cache handler.

        Args:
            cache_dir: Subdirectory within DATA_DIR to store cache files
        """
        self.cache_dir = Path(DATA_DIR) / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        logger.debug(f"Cache directory: {self.cache_dir}")

    def get_cache_key(self, **kwargs) -> str:
        """
        Generate a cache key from keyword arguments.
        
        Args:
            **kwargs: Key-value pairs to generate cache key from
            
        Returns:
            Cache key string
        """
        components = []
        for key, value in sorted(kwargs.items()):
            if value:
                if isinstance(value, (list, tuple, set)):
                    value_str = "_".join(sorted(str(v) for v in value))
                else:
                    value_str = str(value)
                components.append(f"{key}={value_str}")
        return "_".join(components) if components else "all"

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
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        df = pd.DataFrame()
        
        # If cache is disabled or force update is requested, fetch fresh data
        if not use_cache or force_update:
            if not update_func:
                logger.warning("Cache disabled but no update function provided")
                return df
                
            logger.info("Fetching fresh data" + (" (forced)" if force_update else ""))
            if update_kwargs is None:
                update_kwargs = {}
            df = update_func(**update_kwargs)
            
            if not df.empty:
                self.save(df, cache_key)
            return df
        
        # Try to load from cache
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            df = pd.read_parquet(cache_file)
            
            # Return cached data if updates are disabled
            if not update_cache:
                logger.info("Using cached data without updates")
                return df

            # Check for updates if update_func is provided
            if update_func:
                if update_kwargs is None:
                    update_kwargs = {}
                
                # Get last update time
                metadata = self._load_metadata()
                last_update = metadata.get(cache_key, "1970-01-01 00:00")
                update_kwargs["updated_after"] = last_update
                
                # Fetch updates
                new_df = update_func(**update_kwargs)
                
                if not new_df.empty:
                    logger.info(f"Found {len(new_df)} new records")
                    df = pd.concat([df, new_df], ignore_index=True)
                    df = df.drop_duplicates(subset=["key"], keep="last")
                    self.save(df, cache_key)
                else:
                    logger.info("No new records found")
        
        # If no cached data and updates allowed, fetch fresh
        elif update_func:
            logger.info("No cached data found, fetching fresh")
            if update_kwargs is None:
                update_kwargs = {}
            df = update_func(**update_kwargs)
            
            if not df.empty:
                self.save(df, cache_key)
        
        return df

    def save(self, df: pd.DataFrame, cache_key: str) -> None:
        """
        Save data to cache and update metadata.

        Args:
            df: DataFrame to cache
            cache_key: Key to identify the cached data
        """
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        df.to_parquet(cache_file)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata[cache_key] = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._save_metadata(metadata)
        logger.debug(f"Saved {len(df)} records to {cache_file}")

    def _load_metadata(self) -> Dict[str, str]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return {}

    def _save_metadata(self, metadata: Dict[str, str]) -> None:
        """Save cache metadata."""
        self.metadata_file.write_text(json.dumps(metadata, indent=2))
