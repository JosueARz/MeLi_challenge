# Explicación de la construcción y selección de variables (Feature Engineering)

## Contexto

El propósito de este análisis es comprender qué factores influyen en la conversión de productos publicados en un marketplace. La variable objetivo definida es `sold_flag`, que representa si un producto tuvo al menos una unidad vendida (1) o no (0).

Se construyeron nuevas variables a partir del dataset limpio y se aplicaron métodos estadísticos y algorítmicos para identificar las características más relevantes para predecir la conversión.

---

## Variables creadas

| Variable              | Descripción                                                                                         |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| `has_discount`        | Indicador binario: 1 si el producto tiene un descuento (`price < base_price`), 0 si no             |
| `discount_value`      | Diferencia absoluta entre `base_price` y `price`                                                    |
| `discount_percent`    | Porcentaje de descuento respecto al `base_price`                                                    |
| `price_per_unit`      | Precio dividido por la cantidad inicial publicada (`price / initial_quantity`)                     |
| `is_high_priced`      | Indicador binario: 1 si el precio está por encima del percentil 75                                 |
| `sold_flag`           | Variable objetivo: 1 si se vendió al menos una unidad, 0 si no                                     |
| `stock_ratio`         | Proporción del stock restante (`available_quantity / initial_quantity`)                            |
| `is_scarce_stock`     | Indicador binario: 1 si el inventario disponible es menor o igual a 10 unidades                    |

---

## Selección de variables relevantes

Se aplicaron tres enfoques para evaluar la relevancia de cada variable predictora:

1. **ANOVA F-test:** mide si las medias entre grupos (`sold_flag`=0 y 1) difieren significativamente.
2. **Mutual Information (MI):** estima cuánta información proporciona cada feature sobre la variable objetivo, considerando relaciones no lineales.
3. **Importancia en Random Forest:** se entrena un modelo y se mide cuánto contribuye cada variable a la predicción.

### Resultados

| Feature          | F_score      | F_pvalue     | Mutual_Info | RandomForest_Importance |
|------------------|--------------|--------------|-------------|--------------------------|
| price_per_unit   | 0.749        | 0.386        | 0.057       | **0.5708**               |
| stock_ratio      | **16334.32** | 0.000        | **0.1011**  | **0.4037**               |
| is_scarce_stock  | **2989.13**  | 0.000        | 0.0248      | 0.0224                   |
| is_high_priced   | 10.49        | 0.0012       | 0.0047      | 0.0222                   |
| discount_value   | 19.61        | 0.000009     | 0.0009      | 0.0001                   |
| has_discount     | 19.64        | 0.000009     | 0.0004      | 0.00005                  |
| discount_percent | 0.347        | 0.556        | 0.0000      | 0.0001                   |

---

## Variables seleccionadas

Tras el análisis conjunto, se eligieron las siguientes variables como **más relevantes**:

### 1. `price_per_unit`
- Alta importancia en Random Forest (0.57)
- Valor significativo de MI (0.057)
- Aunque el F-test no fue significativo (p > 0.05), el modelo de árboles la posiciona como el mejor predictor.
- **Justificación:** representa el precio normalizado por volumen ofrecido, clave para entender percepción de valor.

### 2. `stock_ratio`
- F_score extremadamente alto (16334.32) y MI elevado (0.101)
- Alta importancia en Random Forest
- **Justificación:** refleja qué tan rápido se ha vendido el inventario original; mayor ratio implica baja conversión.

### 3. `is_scarce_stock`
- F_score alto (2989.13), importancia aceptable en Random Forest
- **Justificación:** puede indicar urgencia o interés en un producto, y correlacionarse con alto o bajo desempeño.

### 4. `is_high_priced`
- Estadísticamente significativa por F (p = 0.001), y capturada moderadamente por RF
- **Justificación:** permite diferenciar entre segmentos de precio y su impacto en conversión.

---

## Variables descartadas

Las siguientes variables fueron descartadas por tener bajo aporte predictivo en múltiples métricas:

- `has_discount` y `discount_value`: aunque con F_score significativo, su Mutual Info y peso en RF son mínimos.
- `discount_percent`: no muestra relevancia en ninguno de los métodos.

**Conclusión:** la presencia o valor del descuento en sí mismo no es suficiente para explicar la conversión, posiblemente por ruido en los datos o prácticas inconsistentes en precios de publicación.

---

## Conclusión

Las variables seleccionadas tienen una sólida justificación cuantitativa y de negocio. Capturan dinámicas clave como el valor relativo del producto, la eficiencia en ventas y la relación entre precio e inventario. Este conjunto de características es apropiado para alimentar modelos predictivos robustos sobre el comportamiento comercial en marketplaces.
