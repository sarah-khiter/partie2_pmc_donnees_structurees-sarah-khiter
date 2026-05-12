import datetime
import os
import tensorflow as tf
from tensorflow import keras

# On réutilise la fonction de la Phase 2 (assure-toi qu'elle est accessible)
def build_regression_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_with_tensorboard(X_train, y_train, X_val, y_val, run_name, epochs=100):
    """Entraîne un modèle de régression avec un callback TensorBoard horodaté."""
    
    # 1. Création du chemin de log unique
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = os.path.join("logs", "fit", f"{run_name}_{timestamp}")
    
    # 2. Configuration du callback
    # histogram_freq=1 permet de voir si les poids s'activent ou s'ils "meurent"
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # 3. Construction et entraînement
    model = build_regression_model(input_dim=X_train.shape[1])
    
    print(f"\n🚀 Lancement du run : {run_name}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[tb_callback],
        verbose=0 # On laisse TensorBoard nous montrer la progression
    )
    
    print(f"✅ Run '{run_name}' terminé. Logs dans {log_dir}")
    return model, history

# --- EXÉCUTION DES DEUX SCÉNARIOS ---

# Run 1 : Données normalisées (Le "Good Path")
model_norm, history_norm = train_with_tensorboard(
    X_train_norm, y_train, X_val_norm, y_val, 
    run_name="california_norm"
)

# Run 2 : Données brutes (Le "Bad Path")
model_raw, history_raw = train_with_tensorboard(
    X_train, y_train, X_val, y_val, 
    run_name="california_raw"
)

# NOTE POUR COLAB :
%load_ext tensorboard
%tensorboard --logdir logs/fit