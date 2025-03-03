#!/usr/bin/env python3
"""Script to view parquet file contents."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="View parquet file contents")
    parser.add_argument("file", help="Path to parquet file")
    parser.add_argument("--head", type=int, default=5, help="Number of rows to show")
    parser.add_argument("--stats", action="store_true", help="Show statistics for numeric columns")
    args = parser.parse_args()

    # Read parquet file
    df = pd.read_parquet(args.file)
    
    print("\n=== Parquet File Info ===")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col} ({df[col].dtype})")
        
    print(f"\n=== First {args.head} Rows ===")
    print(df.head(args.head))
    
    if args.stats:
        print("\n=== Numeric Column Statistics ===")
        print(df.describe())

if __name__ == "__main__":
    main()
