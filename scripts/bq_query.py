"""
Query BigQuery directly without downloading files locally.

Usage:
    # Run a custom SQL query and print results
    python scripts/bq_query.py "SELECT * FROM Table_grisou LIMIT 10"

    # Save results to CSV
    python scripts/bq_query.py "SELECT * FROM Table_grisou LIMIT 1000" --save results.csv

    # Get table info (schema, row count)
    python scripts/bq_query.py --info Table_grisou

    # List all tables
    python scripts/bq_query.py --list

    # Get summary stats for a table
    python scripts/bq_query.py --describe Table_grisou

    # Query preprocessing results (latest run)
    python scripts/bq_query.py --latest preprocess

    # Query model history (latest run)
    python scripts/bq_query.py --latest history

    # Query predictions (latest run)
    python scripts/bq_query.py --latest predictions
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google.cloud import bigquery
from ml_logic.secrets import get_secret


def get_client():
    project = get_secret("GCP_PROJECT")
    region = get_secret("BQ_REGION")
    return bigquery.Client(project=project, location=region), project


def get_dataset():
    return get_secret("BQ_DATASET")


def run_query(sql, save_path=None):
    """Run a SQL query and return/print results."""
    client, project = get_client()
    dataset = get_dataset()

    # Allow shorthand table names without full path
    if f"{project}." not in sql and f"`{project}" not in sql:
        sql = sql.replace("FROM ", f"FROM `{project}.{dataset}`.")
        sql = sql.replace("from ", f"from `{project}.{dataset}`.")

    print(f"Running query...\n")
    df = client.query(sql).result().to_dataframe()
    print(f"Results: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print(df.to_string(max_rows=30, max_cols=15))

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nSaved to {save_path}")

    return df


def list_tables():
    """List all tables in the dataset."""
    client, project = get_client()
    dataset = get_dataset()

    tables = list(client.list_tables(f"{project}.{dataset}"))
    if not tables:
        print("No tables found.")
        return

    print(f"Tables in {project}.{dataset}:\n")
    for t in tables:
        info = client.get_table(f"{project}.{dataset}.{t.table_id}")
        print(f"  {t.table_id:<40} {info.num_rows:>12} rows  {info.num_bytes/1e6:>8.1f} MB")


def table_info(table_name):
    """Show schema and row count for a table."""
    client, project = get_client()
    dataset = get_dataset()

    table = client.get_table(f"{project}.{dataset}.{table_name}")
    print(f"Table: {table_name}")
    print(f"Rows:  {table.num_rows:,}")
    print(f"Size:  {table.num_bytes/1e6:.1f} MB")
    print(f"\nSchema:")
    for field in table.schema:
        print(f"  {field.name:<20} {field.field_type:<10} {field.mode}")


def describe_table(table_name):
    """Get summary statistics for a table (runs on BQ, no local download)."""
    client, project = get_client()
    dataset = get_dataset()
    full_ref = f"`{project}.{dataset}.{table_name}`"

    # Get numeric columns
    table = client.get_table(f"{project}.{dataset}.{table_name}")
    numeric_cols = [f.name for f in table.schema
                    if f.field_type in ("FLOAT", "FLOAT64", "INTEGER", "INT64", "NUMERIC")]

    if not numeric_cols:
        print("No numeric columns found.")
        return

    # Build stats query
    stats = []
    for col in numeric_cols:
        stats.append(f"""
        STRUCT(
            '{col}' AS col,
            COUNT({col}) AS count,
            AVG({col}) AS mean,
            STDDEV({col}) AS std,
            MIN({col}) AS min,
            MAX({col}) AS max
        )""")

    query = f"SELECT s.* FROM {full_ref}, UNNEST([{','.join(stats)}]) AS s"
    df = client.query(query).result().to_dataframe()

    print(f"Summary statistics for {table_name}:\n")
    print(df.to_string(index=False))


def latest_run(prefix):
    """Find and display the latest run for a given prefix (preprocess, history, predictions)."""
    client, project = get_client()
    dataset = get_dataset()

    tables = list(client.list_tables(f"{project}.{dataset}"))
    matching = sorted([t.table_id for t in tables if t.table_id.startswith(prefix)], reverse=True)

    if not matching:
        print(f"No tables found with prefix '{prefix}'")
        return

    latest = matching[0]
    print(f"Latest {prefix} run: {latest}\n")

    query = f"SELECT * FROM `{project}.{dataset}.{latest}` LIMIT 50"
    df = client.query(query).result().to_dataframe()
    print(df.to_string(max_rows=30, max_cols=15))
    print(f"\n({df.shape[0]} rows shown, use --save to export full table)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query BigQuery directly")
    parser.add_argument("query", nargs="?", help="SQL query to run")
    parser.add_argument("--save", help="Save results to CSV file")
    parser.add_argument("--list", action="store_true", help="List all tables")
    parser.add_argument("--info", help="Show schema/info for a table")
    parser.add_argument("--describe", help="Summary stats for a table")
    parser.add_argument("--latest", help="Show latest run (preprocess|history|predictions)")

    args = parser.parse_args()

    if args.list:
        list_tables()
    elif args.info:
        table_info(args.info)
    elif args.describe:
        describe_table(args.describe)
    elif args.latest:
        latest_run(args.latest)
    elif args.query:
        run_query(args.query, save_path=args.save)
    else:
        parser.print_help()
