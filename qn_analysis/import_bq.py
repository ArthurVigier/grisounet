from google.cloud import bigquery
PROJECT = "my-project"
DATASET = "taxifare_dataset"
TABLE = "processed_1k"
query = f"""
SELECT *
FROM {PROJECT}.{DATASET}.{TABLE}
LIMIT 1000
"""
client = bigquery.Client(project=gcp_project)
query_job = client.query(query)
result = query_job.result()
df = result.to_dataframe()


# load data to dataset table

SOURCE = "spheric-voyager-484810-k0.grisou.Table_grisou"

bq load --autodetect $DATASET.$TABLE $SOURCE

SELECT year, month, year, year, year, minute, minute FROM `spheric-voyager-484810-k0.grisou.Table_grisou` LIMIT 1000
