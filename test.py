import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Chargement du dataset
df = pd.read_csv('bienetre.csv')

# Separation des features (X) et de la cible (y)
if 'target' in df.columns:
	target_col = 'target'
	y = df[target_col]
else:
	target_col = 'bienetre'
	y = pd.qcut(df[target_col], q=3, labels=[2, 1, 0]).astype(int)

X = df.drop(columns=[target_col])

# 2. Division en données d'entraînement et de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Normalisation des données
scaler = StandardScaler()

# On ajuste le scaler sur le train set ET on transforme
X_train_scaled = scaler.fit_transform(X_train)

# On transforme le test set (SANS réajuster pour éviter le data leakage)
X_test_scaled = scaler.transform(X_test)


# 1. Initialisation du modèle KNN
# (n_neighbors=5 est une bonne valeur de départ, à optimiser plus tard via Grid Search)
knn = KNeighborsClassifier(n_neighbors=5)

# 2. Entraînement du modèle
knn.fit(X_train_scaled, y_train)

# 3. Prédiction sur les données de test
y_pred = knn.predict(X_test_scaled)

# Affichage des results
print("--- Score d'exactitude (Accuracy) ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}\n")

print("--- Matrice de Confusion ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Rapport de Classification ---")
print(classification_report(y_test, y_pred, target_names=['Sain et Actif (0)', 'Modéré (1)', 'À Risque (2)']))