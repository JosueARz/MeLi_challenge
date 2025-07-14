import sys
import os

# Agrega la raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ml_models.data_analyzer import DataAnalyzer

def main():
    # Crear carpeta de salida
    os.makedirs("outputs", exist_ok=True)

    # Cargar y preparar datos
    analyzer = DataAnalyzer("data/raw/new_items_dataset.csv")
    df = analyzer.get_data()
    summaries = analyzer.describe_prices_and_sales()

    output = []

    # 1. RESUMEN GENERAL
    output.append("========= RESUMEN GENERAL DE VARIABLES NUMÉRICAS =========")
    output.append(
        "Este resumen presenta estadísticas descriptivas de variables clave como precios, stock inicial, "
        "cantidad vendida y stock disponible. Se observa una gran dispersión en los precios, con valores máximos atípicos, "
        "lo cual sugiere outliers. Además, el promedio de ventas por producto es bajo, indicando poca rotación."
    )
    output.append(str(summaries['overall_statistics']))

    # Gráfico: distribución de precios
    plt.figure(figsize=(10, 5))
    sns.histplot(df['base_price'], bins=100, color='blue', label='base_price', kde=True)
    sns.histplot(df['price'], bins=100, color='orange', label='price', kde=True)
    plt.xlim(0, 5000)
    plt.title('Distribución de precios base vs precio con descuento')
    plt.xlabel('Precio')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/price_distribution.png')
    plt.close()

    # 2. DESCUENTOS
    if 'discount_summary' in summaries:
        output.append("\n========= RESUMEN DE DESCUENTOS =========")
        output.append(
            "El análisis indica que el 75% de los productos no tiene descuentos. Sin embargo, existen casos donde el precio final "
            "es mayor al base, lo que podría interpretarse como estrategia de pricing o inconsistencias puntuales."
        )
        output.append(str(summaries['discount_summary']))

    # Gráfico: precio vs ventas
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df[df['sold_quantity'] > 0], x='price', y='sold_quantity', alpha=0.3)
    plt.xlim(0, 5000)
    plt.ylim(0, 200)
    plt.title('Relación entre precio y cantidad vendida')
    plt.xlabel('Precio con descuento')
    plt.ylabel('Cantidad vendida')
    plt.tight_layout()
    plt.savefig('outputs/price_vs_sales.png')
    plt.close()

    # 3. CONVERSIÓN
    if 'conversion_rate_summary' in summaries:
        output.append("\n========= TASA DE CONVERSIÓN =========")
        output.append(
            "Más del 83% de las publicaciones no registran ventas. Esto sugiere problemas de visibilidad o competitividad en la oferta."
        )
        output.append(str(summaries['conversion_rate_summary']))

        df['conversion_rate'] = df['sold_quantity'] / df['initial_quantity'].replace(0, pd.NA)
        plt.figure(figsize=(8, 5))
        sns.histplot(df['conversion_rate'].dropna(), bins=100, kde=True)
        plt.xlim(0, 1)
        plt.title('Distribución de tasa de conversión')
        plt.xlabel('Tasa de conversión')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.savefig('outputs/conversion_distribution.png')
        plt.close()

    # 4. SEGMENTO DE VENDEDOR
    if 'grouped_by_seller_loyalty' in summaries:
        output.append("\n========= VENTAS POR SEGMENTO DE VENDEDOR =========")
        output.append(
            "Los segmentos 'gold' y 'gold_premium' lideran en ventas. También se detectaron valores atípicos como '70.0' o '5500.0' "
            "que fueron limpiados adecuadamente."
        )
        output.append(str(summaries['grouped_by_seller_loyalty']))

        top_loyalties = df['seller_loyalty'].value_counts().head(6).index
        df_filtered = df[df['seller_loyalty'].isin(top_loyalties)]
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_filtered, x='seller_loyalty', y='sold_quantity')
        plt.title('Distribución de ventas por tipo de vendedor')
        plt.xlabel('Segmento del vendedor')
        plt.ylabel('Cantidad vendida')
        plt.tight_layout()
        plt.savefig('outputs/sales_by_loyalty.png')
        plt.close()

    # 5. ESTADO DE PUBLICACIÓN
    if 'grouped_by_status' in summaries:
        output.append("\n========= VENTAS POR ESTADO DE PUBLICACIÓN =========")
        output.append(
            "Las publicaciones activas concentran la mayor parte de las ventas. Se identificaron valores no válidos como '1', que fueron limpiados."
        )
        output.append(str(summaries['grouped_by_status']))

        df_status = df.groupby('status')['sold_quantity'].sum().sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=df_status.index, y=df_status.values)
        plt.title('Total de ventas por estado de publicación')
        plt.xlabel('Estado de publicación')
        plt.ylabel('Unidades vendidas')
        plt.tight_layout()
        plt.savefig('outputs/sales_by_status.png')
        plt.close()

    # 6. VALORES FALTANTES E INCONSISTENTES
    issues = analyzer.analyze_missing_and_inconsistent()

    output.append("\n========= VALORES FALTANTES =========")
    output.append("Resumen de columnas con valores faltantes antes de la limpieza.")
    output.append(str(issues['missing_values']))

    output.append("\n========= INCONSISTENCIAS DETECTADAS =========")
    output.append("Se identifican columnas con tipos mezclados o formatos incorrectos.")
    for col, info in issues['inconsistencies'].items():
        output.append(f"\nColumna: {col}")
        output.append(f" - Tipos detectados: {info['types_detected']}")
        output.append(f" - Total únicos: {info['unique_count']}")
        output.append(f" - Valores sospechosos: {info['suspicious_strings']}")
        output.append(f" - Ejemplos: {info['example_values']}")

    output.append("\n========= CORRECCIONES APLICADAS =========")
    output.append("Transformaciones realizadas sobre el dataset con sus respectivas justificaciones.")
    for log in issues['cleaning_log']:
        output.append(f" - {log}")

    # DETECCIÓN DE OUTLIERS EN PRECIOS
    outliers = analyzer.detect_price_outliers()
    output.append("\n========= DETECCIÓN DE OUTLIERS EN PRECIOS =========")
    output.append(
        "Este análisis identifica valores atípicos en los precios utilizando diferentes enfoques:\n"
        "1. Rango intercuartílico (IQR)\n"
        "2. Z-score estándar\n"
        "3. Comparación relativa con el precio base\n"
        "4. Productos caros sin ventas\n"
    )
    for method, indices in outliers.items():
        output.append(f"\nMétodo: {method}")
        output.append(f"Total detectados: {len(indices)}")
        if len(indices) > 0:
            example_ids = indices[:5]
            output.append(f"Ejemplos de índices: {example_ids}")

    # GRAFICAS DE OUTLIERS
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[['price', 'base_price']])
    plt.title('Boxplot de price y base_price (outliers visibles)')
    plt.tight_layout()
    plt.savefig('outputs/outliers_boxplot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df[['price', 'base_price']])
    plt.title('Distribución de price y base_price (Violin Plot)')
    plt.tight_layout()
    plt.savefig('outputs/outliers_violinplot.png')
    plt.close()

    output.append("\nSe han generado las siguientes gráficas:")
    output.append(" - outputs/outliers_boxplot.png")
    output.append(" - outputs/outliers_violinplot.png")

    # GRAFICAS SIN OUTLIERS (usando percentiles para excluir extremos)
    q_low = df[['price', 'base_price']].quantile(0.10)
    q_high = df[['price', 'base_price']].quantile(0.90)
    df_no_outliers = df[
        (df['price'] >= q_low['price']) & (df['price'] <= q_high['price']) &
        (df['base_price'] >= q_low['base_price']) & (df['base_price'] <= q_high['base_price'])
    ]

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_no_outliers[['price', 'base_price']])
    plt.title('Boxplot de price y base_price (sin outliers extremos)')
    plt.tight_layout()
    plt.savefig('outputs/outliers_boxplot_no_outliers.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_no_outliers[['price', 'base_price']])
    plt.title('Distribución de price y base_price (Violin Plot sin outliers)')
    plt.tight_layout()
    plt.savefig('outputs/outliers_violinplot_no_outliers.png')
    plt.close()

    output.append(" - outputs/outliers_boxplot_no_outliers.png")
    output.append(" - outputs/outliers_violinplot_no_outliers.png")

    analyzer.clean_missing_values() 



    # Guardar dataset limpio
    cleaned_path = "data/processed/cleaned_items_dataset.csv"
    analyzer.get_data().to_csv(cleaned_path, index=False)
    output.append(f"\nDataset limpio guardado en: {cleaned_path}")

    # Imprimir todo
    print("\n\n".join(output))


if __name__ == "__main__":

    main()
