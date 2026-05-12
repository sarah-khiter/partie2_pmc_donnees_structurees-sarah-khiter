import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- RÉPONSE AU PIÈGE DE LA SORTIE ---
# Pourquoi activation='sigmoid' sur la couche de sortie rendrait le modèle inutilisable ?
# Réponse : La sigmoid écrase les valeurs entre 0 et 1. Or, nos cibles (prix des maisons) 
# vont jusqu'à 5.0 (500 000$). Avec une sigmoid, le modèle serait incapable de prédire 
# un prix supérieur à 1.0, peu importe la qualité de l'apprentissage.
# ---------------------------------------

def build_regression_model(input_dim):
    model = keras.Sequential([
        # Couche d'entrée + première couche cachée
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        # Deuxième couche cachée
        layers.Dense(32, activation='relu'),
        # Couche de sortie : 1 seul neurone, pas d'activation pour sortir n'importe quelle valeur réelle
        layers.Dense(1)
    ])
    
    # MSE pour l'optimisation (sensible aux erreurs extrêmes)
    # MAE pour l'interprétation humaine (facile à lire)
    model.compile(
        optimizer='adam', 
        loss='mse', 
        metrics=['mae']
    )
    return model

# 1. Construction et résumé
model = build_regression_model(input_dim=8)
model.summary()

# 2. Entraînement
# On utilise les données normalisées de la Phase 1
print("\nDébut de l'entraînement...")
history = model.fit(
    X_train_norm, y_train, 
    epochs=100, 
    batch_size=32,
    validation_data=(X_val_norm, y_val), 
    verbose=1
)

# 3. Évaluation finale sur le test set (le juge de paix)
test_loss, test_mae = model.evaluate(X_test_norm, y_test, verbose=0)

print(f"\n--- RÉSULTATS FINAUX ---")
print(f"MSE test : {test_loss:.4f}")
print(f"MAE test : {test_mae:.4f} (soit environ {test_mae*100000:.0f} $ d'erreur moyenne)")