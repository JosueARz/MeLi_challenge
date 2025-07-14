from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_features(self) -> pd.DataFrame:
        """
        Crea nuevas variables basadas en columnas existentes del dataset.

        Retorna:
            pd.DataFrame: DataFrame con nuevas columnas generadas.
        """
        df = self.df

        # 1. Si tiene descuento
        df['has_discount'] = (df['base_price'] > df['price']).astype(int)

        # 2. Valor y porcentaje de descuento
        df['discount_value'] = df['base_price'] - df['price']
        df['discount_percent'] = df['discount_value'] / df['base_price'].replace(0, np.nan)

        # 3. Precio por unidad (precio / cantidad inicial)
        df['price_per_unit'] = df['price'] / df['initial_quantity'].replace(0, np.nan)

        # 4. Precio alto (binaria: mayor al P75 del precio)
        price_q3 = df['price'].quantile(0.75)
        df['is_high_priced'] = (df['price'] > price_q3).astype(int)

        # 5. Flag de venta (vendió al menos 1)
        df['sold_flag'] = (df['sold_quantity'] > 0).astype(int)

        # 6. Proporción de stock restante
        df['stock_ratio'] = df['available_quantity'] / df['initial_quantity'].replace(0, np.nan)

        # 7. Escasez de inventario (stock disponible menor o igual a 10)
        df['is_scarce_stock'] = (df['available_quantity'] <= 10).astype(int)

        self.df = df
        return df

    def select_features(self, target_col: str = "sold_flag", threshold_rf: float = 0.01):
        """
        Selecciona variables relevantes automáticamente usando F-test, Mutual Info y Random Forest.
        Se conservan features que son importantes en al menos 2 de las 3 métricas.

        Retorna:
            List[str]: lista de nombres de columnas seleccionadas
        """
        feature_cols = [
            'has_discount', 'discount_value', 'discount_percent',
            'price_per_unit', 'is_high_priced', 'stock_ratio',
            'is_scarce_stock'
        ]

        df = self.df.copy()
        X = df[feature_cols].fillna(0)
        y = df[target_col]

        # F-score
        f_scores, f_pvals = f_classif(X, y)
        f_selected = [feature_cols[i] for i, p in enumerate(f_pvals) if p < 0.05]

        # Mutual Info
        mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
        mi_selected = [feature_cols[i] for i, s in enumerate(mi_scores) if s > 0.005]

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        rf_selected = [feature_cols[i] for i, s in enumerate(rf_scores) if s > threshold_rf]

        # Contar cuántas veces aparece cada feature en los métodos
        from collections import Counter
        all_selected = f_selected + mi_selected + rf_selected
        selection_counts = Counter(all_selected)

        # Elegir las features que aparezcan en al menos 2 métodos
        final_features = [feature for feature, count in selection_counts.items() if count >= 2]
        self.selected_features = final_features
        return final_features
    
    def explain_features(self):
        pass
