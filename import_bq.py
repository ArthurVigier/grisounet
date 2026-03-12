from google.cloud import bigquery
PROJECT = "spheric-voyager-484810-k0"
DATASET = "grisou"
TABLE = "Table_grisou"
query = f"""
SELECT *
FROM {PROJECT}.{DATASET}.{TABLE}
LIMIT 1000
"""
# modifier LIMIT selon les besoins

client = bigquery.Client(project=PROJECT)
query_job = client.query(query)
result = query_job.result()
df = result.to_dataframe()

print(df.shape)
# load data to dataset table

SOURCE = "spheric-voyager-484810-k0.grisou.Table_grisou"
df.to_csv("grisou_1000_lignes.csv", index=False)
print("Sauvegardé dans grisou_1000_lignes.csv")

#bq load  grisou.Table_grisou spheric-voyager-484810-k0.grisou.Table_grisou

#SELECT year, month, year, year, year, minute, minute FROM `spheric-voyager-484810-k0.grisou.Table_grisou` LIMIT 1000
