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
        # Only use project keys for cache key to maximize cache reuse
        key_parts = []
        if project_keys:
            key_parts.append(f"projects={','.join(sorted(project_keys))}")
            
        return "_".join(key_parts) if key_parts else "all_projects"

    def fetch_tickets(
        self,
        project_keys: Optional[List[str]] = None,
        max_results: int = 1000,
        exclude_labels: Optional[List[str]] = None,
        include_subtasks: bool = False,
        use_cache: bool = True,
        update_cache: bool = True,
        force_update: bool = False,
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
            force_update: Whether to force a full update regardless of timestamps

        Returns:
            DataFrame containing ticket data
        """
        self.logger.info(
            f"Fetching tickets for projects={project_keys}, "
            f"max_results={max_results}, exclude_labels={exclude_labels}, "
            f"include_subtasks={include_subtasks}, use_cache={use_cache}, "
            f"update_cache={update_cache}, force_update={force_update}")

        # Get cache key using only essential parameters
        cache_key = self._get_cache_key(project_keys)
        
        # Try to load from cache
        if use_cache and not force_update:
            df = self.cache.load(
                cache_key,
                self._fetch_issues,
                {
                    "project_keys": project_keys,
                    "max_results": max_results,
                    "exclude_labels": exclude_labels,
                    "include_subtasks": include_subtasks,
                },
                use_cache=use_cache,
                update_cache=update_cache,
                force_update=force_update
            )
            if df is not None and len(df) > 0:
                return df

        # Prepare fetch parameters
        fetch_kwargs = {
            "project_keys": project_keys,
            "max_results": max_results,
            "exclude_labels": exclude_labels,
            "include_subtasks": include_subtasks,
        }
        
        # Load from cache if possible, or use this class as callback
        return self.cache.load(
            cache_key=cache_key,
            update_func=self._fetch_issues,
            update_kwargs=fetch_kwargs,
            use_cache=use_cache,
            update_cache=update_cache,
            force_update=force_update,
        )

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
        self.logger.debug(
            f"Fetching issues with: projects={project_keys}, "
            f"exclude_labels={exclude_labels}, include_subtasks={include_subtasks}, "
            f"updated_after={updated_after}")

        # Build JQL query
        conditions = ["status = Closed", "timespent > 0"]
        
        if project_keys:
            project_keys_str = ",".join(project_keys)
            conditions.append(f"project in ({project_keys_str})")
            
        if exclude_labels:
            labels_str = ",".join(exclude_labels)
            conditions.append(f"labels not in ({labels_str})")
            
        if not include_subtasks:
            conditions.append("type != Sub-task")
            
        if updated_after:
            conditions.append(f"updated > '{updated_after}'")
            
        jql = " AND ".join(conditions)
        
        self.logger.debug(f"JQL: {jql}")
        
        # Fetch issues
        issues = self.jira.search_issues(
            jql,
            maxResults=max_results,
            fields="summary,description,created,updated,timeoriginalestimate,timespent",
        )
        
        if not issues:
            self.logger.info("No issues found")
            return pd.DataFrame()
        
        # Process issues
        data = []
        for issue in issues:
            fields = issue.fields
                
            # Get time estimates in hours
            original_estimate = fields.timeoriginalestimate / 3600 if fields.timeoriginalestimate else 0
            time_spent = fields.timespent / 3600 if fields.timespent else 0
            
            # Strip out unnecessary formatting from text fields
            summary = self.text_processor.strip_formatting(getattr(fields, 'summary', ''))
            description = self.text_processor.strip_formatting(getattr(fields, 'description', ''))

            # Append data
            data.append({
                "key": issue.key,
                "summary": summary,
                "description": description,
                "created": fields.created,
                "updated": fields.updated,
                "original_estimate": original_estimate,
                "time_spent": time_spent,
            })

            self.logger.debug(f"Fetched issue {issue.key}: {data[-1]}")
            
        return pd.DataFrame(data)
