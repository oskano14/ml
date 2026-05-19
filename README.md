# Projet ML — Classification du Bien-être

## Description

Ce projet applique l'algorithme **K-Nearest Neighbors (KNN)** pour classifier des individus en 3 catégories de bien-être à partir du dataset `bienetre.csv`.

---

## Dataset — `bienetre.csv`

Le fichier contient **10 000 individus** avec des variables comme l'IMC, le stress, le sommeil, le cholestérol, l'activité physique, etc.

La colonne cible `bienetre` est un score continu. Elle est convertie en **3 classes équilibrées** via `pd.qcut` :

| Classe | Label         | Nombre de personnes |
| ------ | ------------- | ------------------- |
| 0      | Sain et Actif | 3 333 (33.3%)       |
| 1      | Modéré        | 3 333 (33.3%)       |
| 2      | À Risque      | 3 334 (33.3%)       |

---

## Script — `test.py`

### Étapes du script

1. **Chargement des données** — lecture de `bienetre.csv` et création de la variable cible en 3 classes
2. **Affichage de la distribution des classes**
3. **Comparaison des modèles KNN** pour différentes valeurs de k

### Paramètres de validation

| Paramètre            | Valeur                                |
| -------------------- | ------------------------------------- |
| Algorithme           | KNN (K-Nearest Neighbors)             |
| Valeurs de k testées | 1, 3, 5, 7, ..., 99 (nombres impairs) |
| Validation croisée   | K-Fold avec **5 folds**               |
| Split par fold       | **80% entraînement / 20% test**       |
| Normalisation        | StandardScaler (dans un Pipeline)     |

### Métriques affichées

| Métrique       | Description                                             |
| -------------- | ------------------------------------------------------- |
| **Accuracy**   | Taux de bonnes prédictions global                       |
| **Précision**  | Parmi les prédits positifs, combien sont vrais (macro)  |
| **Recall**     | Parmi les vrais positifs, combien sont détectés (macro) |
| **F1-Score**   | Moyenne harmonique de la précision et du recall (macro) |
| **Écart-type** | Stabilité de l'accuracy sur les 5 folds                 |

---

## Résultats

### Évolution selon k

| Zone              | k          | Accuracy    |
| ----------------- | ---------- | ----------- |
| Faible (instable) | k=1        | ~60.5%      |
| Progression       | k=5 → k=25 | 66% → 70%   |
| Plateau           | k≥43       | ~70.4–70.9% |
| **Meilleur k**    | **k=87**   | **~70.89%** |

### Observations

- Les **petites valeurs de k** (1, 3) sont instables et peu performantes
- L'accuracy se **stabilise à partir de k≈43** autour de 70%
- La **précision (macro)** monte plus vite que l'accuracy car les classes sont équilibrées
- Le meilleur compromis **Accuracy / F1-Score** se situe autour de **k=87**

---

## Installation

```bash
pip3 install scikit-learn pandas
```

## Lancement

```bash
python3 test.py
```
