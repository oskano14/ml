import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# 1. Chargement du dataset
df = pd.read_csv('bienetre.csv')

# Separation des features (X) et de la cible (y)
if 'target' in df.columns:
    target_col = 'target'
    y = df[target_col]
else:
    target_col = 'bienetre'
    y = pd.qcut(df[target_col], q=3, labels=[0, 1, 2]).astype(int)

X = df.drop(columns=[target_col])

# 2. Distribution des classes
class_names = {0: 'Sain et Actif', 1: 'Modéré', 2: 'À Risque'}
print("=" * 52)
print("Distribution des classes")
print("=" * 52)
counts = y.value_counts().sort_index()
for cls, count in counts.items():
    print(f"  Classe {cls} - {class_names[cls]:<15} : {count} personnes ({count/len(y)*100:.1f}%)")
print(f"  Total                          : {len(y)} personnes\n")

# 3. Validation croisée K-Fold (5 folds = 80% train / 20% test) pour KNN, k de 5 à 100
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 80)
print("KNN — Comparaison des k (K-Fold = 5 folds, 80/20)")
print("=" * 80)
print(f"{'Modèle KNN':<12} {'Accuracy':>10} {'Précision':>11} {'Recall':>10} {'F1-Score':>10} {'Écart-type':>11}")
print("-" * 56)

results = {}
for k in range(1, 100, 2):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    acc       = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy').mean()
    precision = cross_val_score(pipeline, X, y, cv=kf, scoring='precision_macro').mean()
    recall    = cross_val_score(pipeline, X, y, cv=kf, scoring='recall_macro').mean()
    f1        = cross_val_score(pipeline, X, y, cv=kf, scoring='f1_macro').mean()
    std       = cross_val_score(pipeline, X, y, cv=kf, scoring='accuracy').std()
    results[k] = acc
    print(f"  KNN (k={k:<2})  {acc:>9.2%} {precision:>10.2%} {recall:>10.2%} {f1:>10.2%} {std:>10.2%}")

# 4. Meilleur k
best_k = max(results, key=lambda k: results[k])
print("-" * 56)
print(f"\nMeilleur k : k={best_k} avec une accuracy de {results[best_k]:.2%}")