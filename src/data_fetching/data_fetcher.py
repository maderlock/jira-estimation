"""Module for fetching and processing JIRA ticket data."""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from jira import JIRA

from text_processing import TextProcessor
from .cache import DataCache


class JiraDataFetcher:
    """Class to handle JIRA data fetching and processing."""

    def __init__(
        self,
        jira_url: str,
        jira_email: str,
        jira_token: str,
        text_processor: TextProcessor,
        cache_dir: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the JIRA client with dependencies.
        
        Args:
            jira_url: JIRA instance URL
            jira_email: Authentication email
            jira_token: JIRA API token
            text_processor: Instance of TextProcessor
            cache_dir: Directory for cache storage
            logger: Optional logger instance
        """
        self.jira = JIRA(
            server=jira_url,
            basic_auth=(jira_email, jira_token)
        )
        self.text_processor = text_processor
        self.cache = DataCache(cache_dir, logger=logger)
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initialized JIRA client")

    def _get_cache_key(self, project_keys: List[str], **kwargs) -> str:
        """
        Generate a cache key for the given parameters.
        Only uses essential parameters that affect the data content.

        Args:
            project_keys: List of project keys to fetch tickets from
            **kwargs: Additional parameters (ignored for caching)
            
        Returns:
            Cache key string
        """
        key_parts = []
        if project_keys:
            key_parts.append(f"projects={','.join(sorted(project_keys))}")
            
        return "_".join(key_parts) if key_parts else "all_projects"

    def fetch_tickets(
        self,
        project_keys: Optional[List[str]] = None,
        max_results: int = 1000,
        use_cache: bool = True,
        exclude_labels: Optional[List[str]] = None,
        include_subtasks: bool = False,
        force_update: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch completed JIRA tickets.

        Args:
            project_keys: List of JIRA project keys to fetch tickets from
            max_results: Maximum number of tickets to fetch
            use_cache: Whether to use cached data
            exclude_labels: List of labels to exclude from results
            include_subtasks: Whether to include subtasks
            force_update: Whether to force a full update of the cache

        Returns:
            DataFrame containing ticket data
        """
        df = None
        if project_keys is None:
            project_keys = []
            
        # Generate cache key based on project keys
        cache_key = self._get_cache_key(project_keys)
        
        # Prepare fetch parameters
        fetch_kwargs = {
            "project_keys": project_keys,
            "max_results": max_results,
            "exclude_labels": exclude_labels,
            "include_subtasks": include_subtasks,
        }
        
        # Try to load from cache
        df = self.cache.load(
            cache_key,
            self._fetch_issues,
            fetch_kwargs,
            use_cache=use_cache,
            force_update=force_update
        )
        
        # Always return a DataFrame, even if empty
        if df is None:
            df = pd.DataFrame()
        
        # Ensure we don't exceed max_results
        if len(df) > max_results:
            df = df.head(max_results)
        
        return df

    def _fetch_issues(
        self,
        project_keys: Optional[List[str]] = None,
        max_results: int = 1000,
        exclude_labels: Optional[List[str]] = None,
        include_subtasks: bool = False
    ) -> pd.DataFrame:
        """
        Fetch issues from JIRA.

        Args:
            project_keys: List of JIRA project keys
            max_results: Maximum number of issues to fetch
            exclude_labels: List of labels to exclude
            include_subtasks: Whether to include subtasks

        Returns:
            DataFrame containing issue data
        """
        # Validate max_results
        if max_results <= 0:
            self.logger.warning("Invalid max_results value, using default of 1000")
            max_results = 1000
        
        self.logger.debug(
            f"Fetching issues with max_results={max_results}, "
            f"project_keys={project_keys}, exclude_labels={exclude_labels}, "
            f"include_subtasks={include_subtasks}"
        )
        
        # Build JQL query
        jql_parts = []
        if project_keys:
            jql_parts.append(f"project in ({','.join(project_keys)})")
        if exclude_labels:
            jql_parts.append(f"labels not in ({','.join(exclude_labels)})")
        if not include_subtasks:
            jql_parts.append("type != Sub-task")
        jql_parts.append("status in (Closed)")  # Only completed tickets
        jql_parts.append("timespent > 0")
        
        jql = " AND ".join(jql_parts) if jql_parts else ""
        
        # Fetch issues from JIRA
        issues = []
        try:
            # Handle pagination - JIRA API limit is 100 per request
            start_at = 0
            batch_size = min(max_results, 100)
            
            while start_at < max_results:
                self.logger.debug(f"Fetching batch starting at {start_at}")
                batch = self.jira.search_issues(
                    jql_str=jql,
                    startAt=start_at,
                    maxResults=batch_size,
                    fields=[
                        'summary', 'description', 'created', 'updated',
                        'timespent', 'timeoriginalestimate'
                    ]
                )
                
                if not batch:
                    break
                    
                issues.extend(batch)
                start_at += len(batch)
                
                # If we got fewer issues than requested, we've hit the end
                if len(batch) < batch_size:
                    break
                    
                # If we've got enough issues, stop fetching
                if len(issues) >= max_results:
                    break
                    
            self.logger.debug(f"Retrieved {len(issues)} issues from JIRA")
        except Exception as e:
            self.logger.error(f"Error fetching JIRA issues: {e}")
            return pd.DataFrame()
        
        # Process issues into records
        records = []
        for issue in issues:
            time_spent = issue.fields.timespent / 3600 if issue.fields.timespent else 0
            original_estimate = (
                issue.fields.timeoriginalestimate / 3600
                if issue.fields.timeoriginalestimate else 0
            )
            
            # Skip issues with invalid time values
            if time_spent <= 0 or time_spent > 1040:  # Max 6 months (1040 hours)
                self.logger.debug(
                    f"Skipping issue {issue.key} due to invalid time: {time_spent}"
                )
                continue
            
            records.append({
                'key': issue.key,
                'summary': self.text_processor.strip_formatting(issue.fields.summary),
                'description': self.text_processor.strip_formatting(
                    issue.fields.description or ""
                ),
                'created': issue.fields.created,
                'updated': issue.fields.updated,
                'original_estimate': original_estimate,
                'time_spent': time_spent,
            })
        
        return pd.DataFrame(records)
