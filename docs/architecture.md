# Architecture du projet

Le projet Grisounet vise à analyser les données issues de capteurs miniers afin de détecter les situations à risque liées à la concentration de méthane (CH₄).

## Architecture générale

BigQuery
↓
import_bq.py
↓
Preprocessing
↓
Model Training
↓
FastAPI
↓
Interface utilisateur

## Structure du repository

api/
API FastAPI exposant le modèle

interface/
Interface utilisateur

ml_logic/
Code du modèle et preprocessing

qn_analysis/
Notebooks d'analyse exploratoire

raw_data/
Données locales utilisées pour les tests