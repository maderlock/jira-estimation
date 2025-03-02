"""Module for fetching and processing JIRA ticket data."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from jira import JIRA

from utils import DATA_DIR, get_jira_config

logger = logging.getLogger(__name__)


class JiraDataFetcher:
    """Class to handle JIRA data fetching and processing."""

    def __init__(self):
        """Initialize the JIRA client."""
        config = get_jira_config()
        self.jira = JIRA(
            server=config.url,
            basic_auth=(config.email, config.api_token)
        )
        logger.info("Initialized JIRA client")
        
        # Setup cache directory
        self.cache_dir = Path(DATA_DIR) / "jira_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        logger.debug(f"Cache directory: {self.cache_dir}")

    def fetch_tickets(
        self,
        project_keys: Optional[List[str]] = None,
        max_results: int = 1000,
        exclude_labels: Optional[List[str]] = None,
        include_subtasks: bool = False,
        use_cache: bool = True,
        update_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch completed JIRA tickets.

        Args:
            project_keys: List of JIRA project keys to fetch tickets from
            max_results: Maximum number of tickets to fetch
            exclude_labels: List of labels to exclude from results
            include_subtasks: Whether to include subtasks
            use_cache: Whether to use cached data
            update_cache: Whether to update cache with new tickets

        Returns:
            DataFrame containing ticket data
        """
        cache_key = self._get_cache_key(project_keys, exclude_labels, include_subtasks)
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        df = pd.DataFrame()
        
        # Try to load from cache first
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            df = pd.read_parquet(cache_file)
            
            if not update_cache:
                logger.info("Using cached data without updates")
                return df

            # Get last update time
            metadata = self._load_metadata()
            last_update = metadata.get(cache_key, "1970-01-01 00:00")
            
            # Update cache with new tickets
            new_df = self._fetch_issues(
                project_keys=project_keys,
                max_results=max_results,
                exclude_labels=exclude_labels,
                include_subtasks=include_subtasks,
                updated_after=last_update,
            )
            
            if not new_df.empty:
                logger.info(f"Found {len(new_df)} new tickets")
                df = pd.concat([df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=["key"], keep="last")
                df.to_parquet(cache_file)
                
                # Update metadata
                metadata[cache_key] = datetime.now().strftime("%Y-%m-%d %H:%M")
                self._save_metadata(metadata)
            else:
                logger.info("No new tickets found")
        else:
            # Fetch all tickets
            logger.info("Fetching all tickets")
            df = self._fetch_issues(
                project_keys=project_keys,
                max_results=max_results,
                exclude_labels=exclude_labels,
                include_subtasks=include_subtasks,
            )
            
            if not df.empty:
                df.to_parquet(cache_file)
                
                # Update metadata
                metadata = self._load_metadata()
                metadata[cache_key] = datetime.now().strftime("%Y-%m-%d %H:%M")
                self._save_metadata(metadata)
        
        return df

    def _get_cache_key(
        self,
        project_keys: Optional[List[str]],
        exclude_labels: Optional[List[str]],
        include_subtasks: bool,
    ) -> str:
        """Generate a cache key based on query parameters."""
        components = []
        
        if project_keys:
            components.append(f"projects={'_'.join(sorted(project_keys))}")
            
        if exclude_labels:
            components.append(f"exclude={'_'.join(sorted(exclude_labels))}")
            
        if include_subtasks:
            components.append("subtasks")
            
        return "_".join(components) if components else "all"

    def _load_metadata(self) -> Dict[str, str]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            return json.loads(self.metadata_file.read_text())
        return {}

    def _save_metadata(self, metadata: Dict[str, str]) -> None:
        """Save cache metadata."""
        self.metadata_file.write_text(json.dumps(metadata, indent=2))

    def _fetch_issues(
        self,
        project_keys: Optional[List[str]] = None,
        max_results: int = 1000,
        exclude_labels: Optional[List[str]] = None,
        include_subtasks: bool = False,
        updated_after: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch issues from JIRA.

        Args:
            project_keys: List of JIRA project keys
            max_results: Maximum number of issues to fetch
            exclude_labels: List of labels to exclude
            include_subtasks: Whether to include subtasks
            updated_after: Only fetch issues updated after this date

        Returns:
            DataFrame containing issue data
        """
        logger.debug(
            f"Fetching issues with: projects={project_keys}, "
            f"exclude_labels={exclude_labels}, include_subtasks={include_subtasks}, "
            f"updated_after={updated_after}")

        # Build JQL query
        conditions = ["status = Closed"]
        
        if project_keys:
            conditions.append(f"project in ({','.join(project_keys)})")
            
        if exclude_labels:
            conditions.append(f"labels not in ({','.join(exclude_labels)})")
            
        if not include_subtasks:
            conditions.append("type != Sub-task")
            
        if updated_after:
            conditions.append(f"updated > '{updated_after}'")
            
        jql = " AND ".join(conditions)

        logger.debug(f"JQL: {jql}")
        
        # Fetch issues
        issues = self.jira.search_issues(
            jql,
            maxResults=max_results,
            fields="summary,description,created,updated,resolutiondate,timeoriginalestimate,timespent",
        )
        
        if not issues:
            logger.info("No issues found")
            return pd.DataFrame()
            
        # Process issues
        data = []
        for issue in issues:
            fields = issue.fields
            
            # Calculate duration in hours
            if fields.resolutiondate and fields.created:
                start = datetime.strptime(fields.created[:19], "%Y-%m-%dT%H:%M:%S")
                end = datetime.strptime(fields.resolutiondate[:19], "%Y-%m-%dT%H:%M:%S")
                duration_hours = (end - start).total_seconds() / 3600
            else:
                duration_hours = None
                
            # Get time estimates in hours
            original_estimate = fields.timeoriginalestimate / 3600 if fields.timeoriginalestimate else None
            time_spent = fields.timespent / 3600 if fields.timespent else None
            
            data.append({
                "key": issue.key,
                "summary": fields.summary,
                "description": fields.description,
                "created": fields.created,
                "updated": fields.updated,
                "resolved": fields.resolutiondate,
                "duration_hours": duration_hours,
                "original_estimate": original_estimate,
                "time_spent": time_spent,
            })
            
        return pd.DataFrame(data)
