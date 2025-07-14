import pandas as pd
import numpy as np


class DataAnalyzer:
    """
    Clase para realizar análisis exploratorio de datos (EDA) sobre el dataset de publicaciones,
    incluyendo análisis de nulos, inconsistencias y limpieza automatizada.
    """

    def __init__(self, filepath: str):
        """
        Inicializa el analizador cargando el dataset desde el archivo especificado.

        :param filepath: Ruta al archivo CSV del dataset.
        """
        self.filepath = filepath
        self.data = self._load_dataset()

    def _load_dataset(self) -> pd.DataFrame:
        """
        Carga el dataset desde el archivo CSV.

        :return: DataFrame con los datos cargados.
        """
        try:
            df = pd.read_csv(self.filepath, low_memory=False)
            return df
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No se encontró el archivo: {self.filepath}") from e
        except Exception as e:
            raise RuntimeError(f"Error al leer el archivo: {str(e)}") from e

    def get_data(self) -> pd.DataFrame:
        """
        Retorna el DataFrame cargado.

        :return: DataFrame original cargado desde el archivo.
        """
        return self.data

    def describe_prices_and_sales(self) -> dict:
        """
        Realiza un análisis estadístico completo de precios y cantidades vendidas,
        desagregado por variables relevantes del negocio.

        :return: Diccionario con múltiples DataFrames de resumen.
        """
        data = self.data.copy()

        numeric_cols = ['base_price', 'price', 'initial_quantity', 'sold_quantity', 'available_quantity']
        categorical_cols = ['is_new', 'warranty', 'seller_loyalty', 'shipping_is_free', 'shipping_mode', 'buying_mode', 'status']

        numeric_cols = [col for col in numeric_cols if col in data.columns]
        categorical_cols = [col for col in categorical_cols if col in data.columns]

        summaries = {}

        summaries['overall_statistics'] = data[numeric_cols].describe()

        if 'base_price' in data.columns and 'price' in data.columns:
            data['discount_value'] = data['base_price'] - data['price']
            data['discount_percent'] = data['discount_value'] / data['base_price'].replace(0, pd.NA)
            summaries['discount_summary'] = data[['discount_value', 'discount_percent']].describe()

        if 'sold_quantity' in data.columns and 'initial_quantity' in data.columns:
            data['conversion_rate'] = data['sold_quantity'] / data['initial_quantity'].replace(0, pd.NA)
            summaries['conversion_rate_summary'] = data['conversion_rate'].describe()

        for col in categorical_cols:
            if col in data.columns:
                grouped = data.groupby(col)[['price', 'sold_quantity']].agg(['mean', 'median', 'sum', 'count']).round(2)
                grouped.columns = ['_'.join(c) for c in grouped.columns]
                summaries[f'grouped_by_{col}'] = grouped.sort_values(by='sold_quantity_sum', ascending=False)

        return summaries

    def analyze_missing_and_inconsistent(self) -> dict:
        """
        Analiza y corrige valores faltantes e inconsistencias del dataset.
        Aplica reglas de limpieza definidas previamente y reporta los cambios.

        :return: Diccionario con:
            - missing_values: resumen de nulos antes de limpiar
            - inconsistencies: resumen de columnas con tipos mixtos o valores atípicos
            - cleaning_log: registro detallado de las transformaciones aplicadas
        """
        df = self.data.copy()
        report = {}

        # 1. Valores faltantes antes de limpieza
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        missing_report = pd.DataFrame({
            'missing_count': missing,
            'missing_percent': missing_percent
        }).sort_values(by='missing_percent', ascending=False)
        report['missing_values'] = missing_report

        # 2. Inconsistencias por columna (tipos y valores sospechosos)
        inconsistencies = {}
        for col in df.columns:
            series = df[col]
            unique_values = series.dropna().unique()
            types_detected = set(type(val).__name__ for val in unique_values)
            invalid_strings = [v for v in unique_values if isinstance(v, str) and v.strip().lower() in {'nan', 'none', '', 'null', 'na'}]

            inconsistencies[col] = {
                'unique_count': len(unique_values),
                'types_detected': list(types_detected),
                'example_values': list(unique_values[:5]),
                'suspicious_strings': invalid_strings
            }
        report['inconsistencies'] = inconsistencies

        # 3. Correcciones aplicadas
        cleaning_log = []

        # Regla 2: available_quantity negativa → 0
        if 'available_quantity' in df.columns:
            neg_count = (df['available_quantity'] < 0).sum()
            df['available_quantity'] = df['available_quantity'].apply(lambda x: 0 if x < 0 else x)
            cleaning_log.append(f"available_quantity: {neg_count} valores negativos reemplazados por 0.")

        # Regla 3: sold_quantity nulo → 0
        if 'sold_quantity' in df.columns:
            nulls = df['sold_quantity'].isna().sum()
            df['sold_quantity'] = df['sold_quantity'].fillna(0)
            cleaning_log.append(f"sold_quantity: {nulls} nulos reemplazados por 0.")

        # Regla 4: initial_quantity nulo → mediana
        if 'initial_quantity' in df.columns:
            nulls = df['initial_quantity'].isna().sum()
            median_val = df['initial_quantity'].median()
            df['initial_quantity'] = df['initial_quantity'].fillna(median_val)
            cleaning_log.append(f"initial_quantity: {nulls} nulos reemplazados con mediana ({median_val}).")

        # Regla 5: seller_loyalty con valores numéricos → NaN
        if 'seller_loyalty' in df.columns:
            numeric_loyalties = df['seller_loyalty'].apply(lambda x: str(x).replace('.', '', 1).isdigit())
            count = numeric_loyalties.sum()
            df.loc[numeric_loyalties, 'seller_loyalty'] = pd.NA
            cleaning_log.append(f"seller_loyalty: {count} valores numéricos inválidos reemplazados por NaN.")

        # Regla 6: status inválido → NaN
        if 'status' in df.columns:
            valid_statuses = ['active', 'paused', 'closed', 'not_yet_active']
            invalid_statuses = ~df['status'].isin(valid_statuses)
            count = invalid_statuses.sum()
            df.loc[invalid_statuses, 'status'] = pd.NA
            cleaning_log.append(f"status: {count} valores no válidos reemplazados por NaN.")

        # Regla 7: price y base_price con NaN
        for col in ['price', 'base_price']:
            if col in df.columns:
                null_pct = df[col].isna().mean() * 100
                null_count = df[col].isna().sum()
                if null_pct <= 5:
                    med = df[col].median()
                    df[col] = df[col].fillna(med)
                    cleaning_log.append(f"{col}: {null_count} nulos imputados con mediana ({med}).")
                else:
                    df = df[df[col].notna()]
                    cleaning_log.append(f"{col}: {null_count} filas eliminadas por >5% de nulos.")

        # Actualiza self.data con el DataFrame limpio
        self.data = df
        report['cleaning_log'] = cleaning_log

        return report

    def detect_price_outliers(self) -> dict:
        """
        Detecta outliers en las columnas de precios utilizando diferentes enfoques:
            1. Método del IQR (Interquartile Range)
            2. Método de z-score
            3. Precios mayores al base_price en más de un 50%
            4. Productos con precio alto y sin ventas
        Retorna un diccionario con los índices detectados por cada método.
        """
        df = self.data.copy()
        results = {}

        # 1. IQR para 'price'
        if 'price' in df.columns:
            q1 = df['price'].quantile(0.25)
            q3 = df['price'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)].index.tolist()
            results['price_iqr_outliers'] = iqr_outliers

        # 2. Z-score para 'price'
        if 'price' in df.columns:
            mean_price = df['price'].mean()
            std_price = df['price'].std()
            zscore_outliers = df[(df['price'] - mean_price).abs() > 3 * std_price].index.tolist()
            results['price_zscore_outliers'] = zscore_outliers

        # 3. Precio desproporcionado respecto a base_price
        if 'price' in df.columns and 'base_price' in df.columns:
            rel_diff = (df['price'] - df['base_price']) / df['base_price'].replace(0, pd.NA)
            rel_outliers = df[rel_diff > 0.5].index.tolist()  # más de 50% sobreprecio
            results['relative_price_outliers'] = rel_outliers

        # 4. Precios altos con cero ventas
        if 'price' in df.columns and 'sold_quantity' in df.columns:
            high_price = df['price'] > df['price'].quantile(0.95)
            zero_sales = df['sold_quantity'] == 0
            price_no_sales_outliers = df[high_price & zero_sales].index.tolist()
            results['high_price_no_sales'] = price_no_sales_outliers

        return results
    
    def clean_missing_values(self) -> pd.DataFrame:
        """
        Aplica reglas específicas para corregir valores nulos y eliminar la columna 'sub_status'.

        :return: DataFrame limpio con valores corregidos.
        """
        df = self.data.copy()
        cleaning_log = []

        # Eliminar columna 'sub_status' si existe
        if 'sub_status' in df.columns:
            df.drop(columns=['sub_status'], inplace=True)
            cleaning_log.append("Columna 'sub_status' eliminada.")

        # Reglas de imputación específicas
        imputations = {
            'warranty': 'No',
            'sold_quantity': 0,
            'initial_quantity': df['initial_quantity'].median() if 'initial_quantity' in df else 0,
            'available_quantity': 0,
            'conversion_rate': df['conversion_rate'].median() if 'conversion_rate' in df else 0,
            'stock_ratio': df['stock_ratio'].median() if 'stock_ratio' in df else 0,
            'price_per_unit': df['price_per_unit'].median() if 'price_per_unit' in df else 0
        }

        for col, val in imputations.items():
            if col in df.columns:
                null_count = df[col].isnull().sum()
                df[col] = df[col].fillna(val)
                cleaning_log.append(f"{col}: {null_count} nulos reemplazados por {val}.")

        # Eliminar filas con nulos en columnas críticas
        cols_criticas = ['sold_flag', 'price_per_unit']
        before_rows = len(df)
        df.dropna(subset=[col for col in cols_criticas if col in df.columns], inplace=True)
        removed = before_rows - len(df)
        cleaning_log.append(f"Filas eliminadas por nulos en columnas críticas {cols_criticas}: {removed}")
        df.dropna(inplace=True)

        self.data = df
        return df


