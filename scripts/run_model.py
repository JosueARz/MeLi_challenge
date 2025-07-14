"""
Entrena ModelPredictor, encuentra el umbral óptimo
y guarda pipeline + metadata completa.
"""

from pathlib import Path
import pandas as pd
from ml_models.model import ModelPredictor

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH   = PROJECT_DIR / "data" / "processed" / "items_with_selected_features.csv"

SAFE_COLS = [
    "base_price", "price", "is_new", "initial_quantity",
    "is_scarce_stock", "stock_ratio", "price_per_unit",
]

# ---------- carga datos ----------
df = pd.read_csv(DATA_PATH)

# ---------- entrenamiento ----------
mp = ModelPredictor(df, feature_cols=SAFE_COLS, model_dir="models")
mp.train()          # GridSearch, selección del mejor pipeline
mp.summary()

# ---------- evaluación ----------
mp.evaluate_test(threshold=0.50)
best_thr = mp.find_best_threshold()
mp.evaluate_test(threshold=best_thr)

# ---------- guardado final (pipeline + meta) ----------
mp.save_model()
