# Validation scientifique des seuils CH₄

## 1. Objet du document

Ce document a pour objectif d’expliciter les **seuils de travail utilisés dans le projet** autour de la concentration en méthane (CH₄), de distinguer ce qui relève :

- des **références issues du dataset** ;
- des **références métier / sécurité** associées aux capteurs ;
- des **hypothèses de travail retenues pour la modélisation**.

L’objectif n’est pas d’établir une doctrine de sécurité minière complète, mais de **rendre traçable et compréhensible la manière dont les seuils sont utilisés pour définir la cible du modèle**.

---

## 2. Contexte métier

Le dataset étudié provient d’une mine souterraine de charbon. Il combine :

- des **mesures environnementales** prises dans la mine : concentration en méthane, vitesse de l’air, température, humidité, pression, etc. ;
- des **mesures liées à l’activité du cisailleur** : vitesse, direction de déplacement, intensité électrique de plusieurs organes de la machine.

Le cisailleur extrait le charbon le long du front de taille. Son activité peut contribuer à la libération de méthane dans la zone surveillée. La surveillance continue de l’environnement permet de détecter les situations susceptibles de conduire à une alerte ou à une mise en sécurité.

Le système réel ne repose pas sur un seul capteur, mais sur un ensemble de mesures complémentaires :

- des **capteurs de méthane**, qui suivent directement la concentration en CH₄ ;
- des **anémomètres**, qui surveillent la circulation de l’air ;
- d’autres capteurs d’environnement et de procédé.

Les mesures d’anémométrie et de méthane jouent un rôle **complémentaire** dans la sécurité du site : la ventilation contribue à la dilution du méthane, et une baisse du flux d’air peut favoriser l’accumulation de gaz.

---

## 3. Capteurs méthane critiques pour le projet

Le dataset contient plusieurs capteurs de méthane. Cependant, la documentation source précise que trois capteurs sont situés dans la zone la plus exposée du longwall, c’est-à-dire l’endroit où le méthane libéré pendant l’exploitation peut le plus s’accumuler :

- `MM263`
- `MM264`
- `MM256`

Ces trois capteurs sont considérés comme **critiques pour le projet**, car ils représentent la zone opérationnelle la plus sensible du point de vue de l’accumulation de méthane liée à l’activité minière.

---

## 4. Seuils disponibles dans le dataset

D’après la description du jeu de données, les seuils associés aux capteurs de méthane sont les suivants :

### Capteurs méthane principaux du projet

| Capteur | Seuil warning (W) | Seuil alarm (A) |
|---|---:|---:|
| MM263 | 1.0 % CH₄ | 1.5 % CH₄ |
| MM264 | 1.0 % CH₄ | 1.5 % CH₄ |
| MM256 | 1.0 % CH₄ | 1.5 % CH₄ |

### Autres capteurs méthane présents dans le dataset

| Capteur | Seuil warning (W) | Seuil alarm (A) |
|---|---:|---:|
| MM252 | 1.5 % CH₄ | 2.0 % CH₄ |
| MM261 | 1.0 % CH₄ | 1.5 % CH₄ |
| MM262 | 0.6 % CH₄ | 1.0 % CH₄ |
| MM211 | 1.5 % CH₄ | 2.0 % CH₄ |

Cette hétérogénéité montre qu’il **n’existe pas un seuil unique applicable à tous les capteurs**. Les seuils dépendent du capteur, de sa position et de son rôle dans le dispositif de surveillance.

---

## 5. Compréhension métier retenue pour le projet

La compréhension métier retenue à ce stade est la suivante :

- le cisailleur extrait le charbon et se déplace selon des phases de travail et de retour ;
- son activité influence les conditions locales de production et peut contribuer à la libération de méthane ;
- les capteurs environnementaux surveillent en continu l’état de la zone ;
- lorsque certaines conditions sont atteintes, le système de sécurité peut déclencher une alerte ou une mise hors service de la machine.

Dans ce cadre, **l’objectif du projet n’est pas de prédire une explosion** ni de remplacer le système de sécurité existant.

L’objectif est de :

- **prédire à l’avance un dépassement futur de seuil méthane** sur les capteurs critiques ;
- **anticiper les situations susceptibles de provoquer une alerte, une mise en sécurité ou un arrêt de production** ;
- aider à une meilleure **régulation opérationnelle de la production**.

Le modèle doit donc être compris comme un **outil d’anticipation opérationnelle**, et non comme un système autonome de qualification du risque d’explosion.

---

## 6. Seuil de travail retenu pour la cible du modèle

La documentation du dataset précise qu’une version transformée des données a été utilisée dans une compétition de data science. Dans cette version :

- chaque exemple représente une fenêtre passée de **10 minutes** ;
- la cible indique si le **seuil warning** est atteint dans une fenêtre future ;
- pour les capteurs `MM263`, `MM264` et `MM256`, le label est positionné à `warning` si :

```text
max(MM(t181), ..., MM(t360)) ≥ 1.0