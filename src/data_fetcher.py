"""Module for fetching and processing JIRA ticket data."""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from jira import JIRA

from cache import DataCache
from utils import get_jira_config

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
        self.cache = DataCache("jira_cache")
        logger.info("Initialized JIRA client")

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
        # Generate cache key from parameters
        cache_key = self.cache.get_cache_key(
            projects=project_keys,
            exclude=exclude_labels,
            subtasks=include_subtasks
        )
        
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
