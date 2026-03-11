# Configuration BigQuery

Les données des capteurs sont stockées dans Google BigQuery.

## Accès

Le projet utilise le SDK Google Cloud via la librairie :

google-cloud-bigquery

## Variables d'environnement

GOOGLE_APPLICATION_CREDENTIALS

## Pipeline

Le script `import_bq.py` permet :

- la connexion à BigQuery
- l'extraction des données
- leur transformation en DataFrame pandas