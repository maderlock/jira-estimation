"""Module for fetching and processing JIRA ticket data."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from jira import JIRA

from utils import DATA_DIR, get_jira_config

logger = logging.getLogger(__name__)


class JiraDataFetcher:
    """Class to handle JIRA data fetching and processing."""

    def __init__(self):
        """Initialize JIRA client."""
        # Get JIRA credentials from environment
        jira_config = get_jira_config()
        
        logger.info("Initializing JIRA client")
        self.jira = JIRA(
            server=jira_config.url,
            basic_auth=(jira_config.email, jira_config.api_token)
        )
        self.cache_dir = Path(DATA_DIR) / "jira_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        logger.debug(f"Cache directory: {self.cache_dir}")

    def fetch_completed_issues(
        self,
        project_keys: Optional[List[str]] = None,
        exclude_labels: Optional[List[str]] = None,
        max_results: int = 1000,
        include_subtasks: bool = True,
        use_cache: bool = True,
        update_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch completed issues from JIRA.

        Args:
            project_keys: List of project keys to include
            exclude_labels: List of labels to exclude from the search
            max_results: Maximum number of issues to fetch
            include_subtasks: Whether to include subtasks in the search
            use_cache: Whether to use cached data if available
            update_cache: Whether to update cache with new tickets

        Returns:
            DataFrame containing processed issue data
        """
        logger.info(f"Fetching completed issues for projects: {', '.join(project_keys or [])}")
        logger.debug(f"Parameters: max_results={max_results}, exclude_labels={exclude_labels}, "
                    f"include_subtasks={include_subtasks}, use_cache={use_cache}, "
                    f"update_cache={update_cache}")

        # Generate cache key based on query parameters
        cache_key = self._generate_cache_key(project_keys, exclude_labels, include_subtasks)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Try to load from cache
        if use_cache and cache_file.exists():
            cached_df = pd.read_pickle(cache_file)
            if not update_cache:
                logger.info(f"Loaded {len(cached_df)} issues from cache")
                return cached_df

            # Get last update time
            metadata = self._load_metadata()
            last_update = metadata.get(cache_key, "1970-01-01T00:00:00Z")
            
            # Update cache with new tickets
            new_df = self._fetch_issues(
                project_keys,
                exclude_labels,
                max_results,
                include_subtasks,
                updated_after=last_update,
            )
            
            if not new_df.empty:
                df = pd.concat([cached_df, new_df]).drop_duplicates(subset=["key"])
                df.to_pickle(cache_file)
                self._update_metadata(cache_key)
                logger.info(f"Updated cache with {len(new_df)} new issues")
                return df
                
            logger.info(f"No new issues found, returning cached data")
            return cached_df

        # Fetch all data if no cache or cache disabled
        df = self._fetch_issues(
            project_keys,
            exclude_labels,
            max_results,
            include_subtasks,
        )
        
        if update_cache and not df.empty:
            df.to_pickle(cache_file)
            self._update_metadata(cache_key)
            logger.info(f"Updated cache with {len(df)} issues")
            
        logger.info(f"Successfully fetched {len(df)} issues")
        return df

    def _generate_cache_key(
        self,
        project_keys: Optional[List[str]],
        exclude_labels: Optional[List[str]],
        include_subtasks: bool,
    ) -> str:
        """Generate a unique cache key based on query parameters."""
        components = []
        if project_keys:
            components.append(f"projects={'_'.join(sorted(project_keys))}")
        if exclude_labels:
            components.append(f"exclude={'_'.join(sorted(exclude_labels))}")
        if include_subtasks:
            components.append("subtasks")
        return "_".join(components) if components else "all"

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}

    def _update_metadata(self, cache_key: str) -> None:
        """Update cache metadata with current timestamp."""
        metadata = self._load_metadata()
        metadata[cache_key] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f)

    def _fetch_issues(
        self,
        project_keys: Optional[List[str]] = None,
        exclude_labels: Optional[List[str]] = None,
        max_results: int = 1000,
        include_subtasks: bool = True,
        updated_after: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch issues from JIRA API."""
        logger.info(f"Fetching issues from JIRA API")
        logger.debug(f"Parameters: project_keys={project_keys}, exclude_labels={exclude_labels}, "
                    f"max_results={max_results}, include_subtasks={include_subtasks}, "
                    f"updated_after={updated_after}")

        # Build JQL query
        conditions = ["status = Done"]
        
        if project_keys:
            conditions.append(f"project in ({','.join(project_keys)})")
        
        if not include_subtasks:
            conditions.append("type != Sub-task")
            
        if exclude_labels:
            for label in exclude_labels:
                conditions.append(f"labels != {label}")
                
        if updated_after:
            conditions.append(f"updated > '{updated_after}'")
            
        jql = " AND ".join(conditions)
        
        # Fetch issues
        issues = self.jira.search_issues(
            jql,
            maxResults=max_results,
            fields="summary,description,created,resolutiondate,timeoriginalestimate,timespent,labels",
        )
        
        if not issues:
            logger.info("No issues found")
            return pd.DataFrame()
            
        # Process issues
        data = []
        for issue in issues:
            fields = issue.fields
            data.append({
                "key": issue.key,
                "summary": fields.summary or "",
                "description": fields.description or "",
                "created": fields.created,
                "resolved": fields.resolutiondate,
                "original_estimate": fields.timeoriginalestimate or 0,
                "time_spent": fields.timespent or 0,
                "labels": ",".join(fields.labels) if fields.labels else "",
                "duration_hours": (fields.timespent or 0) / 3600,
            })
            
        logger.info(f"Fetched {len(data)} issues")
        return pd.DataFrame(data)
