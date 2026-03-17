"""Fetch all BigQuery tables locally into results/tables/"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google.cloud import bigquery
from ml_logic.secrets import get_secret


def fetch_all_tables(limit=None):
    """Download every table in the grisou dataset as CSV."""
    project = get_secret("GCP_PROJECT")
    dataset = get_secret("BQ_DATASET")
    region = get_secret("BQ_REGION")

    client = bigquery.Client(project=project, location=region)
    tables = list(client.list_tables(f"{project}.{dataset}"))

    if not tables:
        print("No tables found.")
        return

    output_dir = "results/tables"
    os.makedirs(output_dir, exist_ok=True)

    limit_msg = f" (limit: {limit} rows)" if limit else " (full)"
    print(f"Found {len(tables)} tables in {project}.{dataset}{limit_msg}\n")

    for table in tables:
        table_id = table.table_id
        full_ref = f"{project}.{dataset}.{table_id}"

        print(f"  Fetching {table_id}...", end=" ")
        try:
            query = f"SELECT * FROM `{full_ref}`"
            if limit:
                query += f" LIMIT {limit}"
            df = client.query(query).result().to_dataframe(create_bqstorage_client=True)

            path = os.path.join(output_dir, f"{table_id}.csv")
            df.to_csv(path, index=False)
            print(f"{df.shape[0]} rows -> {path}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone. All tables saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Max rows per table (default: all)")
    args = parser.parse_args()
    fetch_all_tables(limit=args.limit)
