import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- RÉPONSE À LA DÉCISION EN AMONT ---
# Choix : (b) split puis scaler.fit(X_train).
# Pourquoi : Pour éviter le "Data Leakage". Si on fit sur l'ensemble X, les moyennes 
# et écart-types du test set influencent la normalisation du train set. En production, 
# le test set simule des données futures inconnues ; le scaler ne doit donc connaître 
# que les statistiques du passé (le train set).
# ---------------------------------------

# 1. Charger le dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# 2. Split train/test (80% train+val, 20% test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Split train/val (80% de X_temp -> train, 20% de X_temp -> val)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

# 4. Normalisation (StandardScaler)
scaler = StandardScaler()

# ON FIT UNIQUEMENT SUR LE TRAIN
scaler.fit(X_train)

# ON TRANSFORME LES TROIS SETS
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# --- AFFICHAGES ET VÉRIFICATIONS ---
print(f"X_train shape : {X_train.shape}")
print(f"X_val shape   : {X_val.shape}")
print(f"X_test shape  : {X_test.shape}")

print("\n--- Stats descriptives X_train_norm ---")
print(f"Mean (par feature) : {X_train_norm.mean(axis=0).round(2)}")
print(f"Std (par feature)  : {X_train_norm.std(axis=0).round(2)}")

print(f"\nFeature names ({len(feature_names)}) : {feature_names}")

# --- TEST ADVERSARIAL : OUTLIERS ---
X_extreme = np.array([[99999, -99999, 0, 0, 0, 0, 37.0, -120.0]])
X_extreme_norm = scaler.transform(X_extreme)
print(f"\nMedInc (valeur extrême) normalisé : {X_extreme_norm[0, 0]:.2f}")