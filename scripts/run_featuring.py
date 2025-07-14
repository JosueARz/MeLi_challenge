import pandas as pd
from ml_models.data_analyzer import DataAnalyzer
from ml_models.feature_engineering import FeatureEngineer

def generar_explicacion_features(path_md: str):
    with open(path_md, "r", encoding="utf-8") as src:
        content = src.read()
    with open(path_md, "w", encoding="utf-8") as dst:
        dst.write(content)
    print("\nExplicación de variables:\n")
    print(content)

def main():
    # Cargar dataset limpio
    analyzer = DataAnalyzer("data/processed/cleaned_items_dataset.csv")
    df = analyzer.get_data()

    # Generar nuevas variables
    fe = FeatureEngineer(df)
    df_with_features = fe.create_features()

    # Seleccionar features relevantes
    selected = fe.select_features(target_col="sold_flag")
    print(f"Features seleccionadas automáticamente: {selected}")

    # Guardar dataset con las columnas originales + features seleccionadas + target
    cols_to_keep = df.columns.tolist() + selected + ['sold_flag']
    df_selected = df_with_features[cols_to_keep].copy()

    output_path = "data/processed/items_with_selected_features.csv"
    df_selected.to_csv(output_path, index=False)
    print(f"Variables seleccionadas guardadas en: {output_path}")
    print(f"Columnas agregadas: {selected}")

    # Generar la explicación
    generar_explicacion_features("reports/feature_engineering_explanation.md")

if __name__ == "__main__":
    main()
