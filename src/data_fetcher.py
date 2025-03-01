"""Module for fetching and processing JIRA ticket data."""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from jira import JIRA

from src.config import DATA_DIR, JIRA_API_TOKEN, JIRA_EMAIL, JIRA_PROJECT, JIRA_URL


class JiraDataFetcher:
    """Class to handle JIRA data fetching and processing."""

    def __init__(self):
        """Initialize JIRA client."""
        self.jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
        self.cache_dir = Path(DATA_DIR) / "jira_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"

    def fetch_completed_issues(
        self,
        max_results: int = 2000,
        projects: Optional[List[str]] = None,
        exclude_labels: Optional[List[str]] = None,
        include_subtasks: bool = True,
        use_cache: bool = True,
        update_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch completed issues from JIRA.

        Args:
            max_results: Maximum number of issues to fetch.
            projects: List of project keys to include. If None, uses default project from config.
            exclude_labels: List of labels to exclude from the search.
            include_subtasks: Whether to include subtasks in the search.
            use_cache: Whether to use cached data if available.
            update_cache: Whether to update cache with new tickets.

        Returns:
            DataFrame containing processed issue data.
        """
        cache_key = self._generate_cache_key(projects, exclude_labels, include_subtasks)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if use_cache and cache_file.exists():
            cached_df = pd.read_pickle(cache_file)
            if not update_cache:
                return cached_df
            
            # Get only new tickets since last cache update
            last_update = pd.to_datetime(cached_df["created"]).max()
            new_df = self._fetch_issues(
                max_results,
                projects,
                exclude_labels,
                include_subtasks,
                updated_after=last_update,
            )
            
            if not new_df.empty:
                # Combine cached and new data, remove duplicates
                df = pd.concat([cached_df, new_df]).drop_duplicates(subset=["id"])
                df.to_pickle(cache_file)
                return df
            return cached_df

        # No cache or not using cache, fetch all data
        df = self._fetch_issues(max_results, projects, exclude_labels, include_subtasks)
        if update_cache and not df.empty:
            df.to_pickle(cache_file)
        return df

    def _fetch_issues(
        self,
        max_results: int,
        projects: Optional[List[str]],
        exclude_labels: Optional[List[str]],
        include_subtasks: bool,
        updated_after: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch issues from JIRA with given filters."""
        # Build JQL query
        project_clause = (
            f"project in ({','.join(projects)})"
            if projects
            else f"project = {JIRA_PROJECT}"
        )
        
        exclude_clause = (
            f"AND labels not in ({','.join(exclude_labels)})"
            if exclude_labels
            else ""
        )
        
        subtask_clause = "AND issuetype != Sub-task" if not include_subtasks else ""
        updated_clause = f"AND updated > '{updated_after.isoformat()}'" if updated_after else ""
        
        jql_query = f"{project_clause} AND status = Done {exclude_clause} {subtask_clause} {updated_clause} ORDER BY created DESC"
        
        issues = self.jira.search_issues(
            jql_query,
            maxResults=max_results,
            fields="summary,description,comment,created,resolutiondate,timeoriginalestimate,timespent,labels,issuetype,updated",
        )

        data = []
        for issue in issues:
            processed_issue = self._process_issue(issue)
            if processed_issue:
                data.append(processed_issue)

        df = pd.DataFrame(data)
        if not df.empty:
            self._process_timestamps(df)
        
        return df

    def _generate_cache_key(
        self,
        projects: Optional[List[str]],
        exclude_labels: Optional[List[str]],
        include_subtasks: bool,
    ) -> str:
        """Generate a unique cache key based on query parameters."""
        key_parts = []
        if projects:
            key_parts.append(f"proj_{'_'.join(sorted(projects))}")
        else:
            key_parts.append(f"proj_{JIRA_PROJECT}")
        
        if exclude_labels:
            key_parts.append(f"excl_{'_'.join(sorted(exclude_labels))}")
        
        if not include_subtasks:
            key_parts.append("no_subtasks")
        
        return "_".join(key_parts)

    def _process_issue(self, issue) -> Optional[Dict]:
        """Process a single JIRA issue."""
        fields = issue.fields
        
        # Skip if missing critical fields
        if not fields.timespent:
            return None

        comments = " ".join([c.body for c in fields.comment.comments]) if fields.comment else ""
        
        return {
            "id": issue.key,
            "summary": fields.summary or "",
            "description": fields.description or "",
            "comments": comments,
            "created": fields.created,
            "resolution_date": fields.resolutiondate,
            "original_estimate": fields.timeoriginalestimate or 0,  # in seconds
            "time_spent": fields.timespent,  # in seconds
            "issue_type": fields.issuetype.name,
            "labels": [label for label in fields.labels] if fields.labels else [],
            "updated": fields.updated,
        }

    @staticmethod
    def _process_timestamps(df: pd.DataFrame) -> None:
        """Process timestamp columns in the DataFrame."""
        df["created"] = pd.to_datetime(df["created"])
        df["resolution_date"] = pd.to_datetime(df["resolution_date"])
        df["updated"] = pd.to_datetime(df["updated"])
        
        # Convert time fields from seconds to hours
        df["original_estimate_hours"] = df["original_estimate"] / 3600
        df["time_spent_hours"] = df["time_spent"] / 3600
        
        # Calculate actual duration (resolution - created)
        df["calendar_duration_hours"] = (
            df["resolution_date"] - df["created"]
        ).dt.total_seconds() / 3600
        
        # Drop rows with missing time data
        df.dropna(subset=["time_spent_hours"], inplace=True)
