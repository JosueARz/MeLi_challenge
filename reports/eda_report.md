# Análisis de Valores Faltantes e Inconsistencias del Dataset `new_items_dataset.csv`

## 1. Objetivo

Este informe tiene como propósito documentar la presencia de **valores nulos** y **posibles inconsistencias** dentro del dataset, con el fin de respaldar el entendimiento estructural y semántico de los datos. El análisis es descriptivo y se enfoca en la calidad de la información disponible para su posterior utilización analítica o modelado.

---

## 2. Resumen de Valores Nulos

Se realizó un escaneo de todas las columnas para identificar la presencia de valores faltantes. A continuación se presenta un resumen de los campos analizados:

| Columna                  | Valores nulos | Observaciones                                             |
| ------------------------ | ------------- | --------------------------------------------------------- |
| `id`                     | No            | Identificador primario, completamente poblado.            |
| `title`                  | No            | Título descriptivo, sin ausencias.                        |
| `base_price`             | No            | Valor presente en todos los registros.                    |
| `price`                  | No            | Completo, aunque con valores extremos.                    |
| `initial_quantity`       | No            | Totalmente registrado.                                    |
| `sold_quantity`          | No            | Disponible en todos los casos.                            |
| `available_quantity`     | No            | No presenta nulos.                                        |
| `category_id`            | Sí            | Algunas ausencias puntuales.                              |
| `tags`                   | Sí            | Componente opcional, frecuentemente ausente.              |
| `attributes`             | Sí            | Estructura jerárquica, ausente en gran parte del dataset. |
| `variations`             | Sí            | Similar a `attributes`.                                   |
| `pictures`               | Sí            | No todos los productos tienen asociadas imágenes.         |
| `seller_id`              | No            | Completamente presente.                                   |
| `seller_country`         | Sí            | Muy pocos registros faltantes.                            |
| `seller_province`        | Sí            | Ausencias poco significativas.                            |
| `seller_city`            | Sí            | Faltantes similares a `province`.                         |
| `seller_loyalty`         | Sí            | Valores nulos y algunos registros inconsistentes.         |
| `buying_mode`            | Sí            | Campo categórico con ausencias menores.                   |
| `shipping_mode`          | Sí            | Algunos registros incompletos.                            |
| `shipping_admits_pickup` | Sí            | Faltantes de tipo booleano.                               |
| `shipping_is_free`       | Sí            | Similar al anterior.                                      |
| `status`                 | Sí            | Faltantes y casos con codificación no estándar.           |
| `sub_status`             | Sí            | Valor opcional, con alta tasa de nulos.                   |
| `warranty`               | Sí            | Campo textual opcional.                                   |
| `is_new`                 | Sí            | Valor booleano con ausencias poco frecuentes.             |

---

## 3. Inconsistencias Detectadas

### 3.1. `seller_loyalty`

Se observan valores categóricos esperados como `gold`, `silver`, `bronze`, `free`, `gold_premium`, `gold_special`. Sin embargo, se identificaron entradas numéricas como `70.0`, `100.0`, `5500.0`, que no corresponden a ninguna categoría conocida.


---

### 3.2. `status`

El campo `status` contiene valores esperados como `active`, `paused`, `closed`, `not_yet_active`, pero también se registran valores como `1`, lo que representa una posible codificación numérica incorrecta.

**Ejemplo:**

```text
status = '1'  # No corresponde a categoría conocida
```

---

### 3.3. `discount_value` y `discount_percent`

Los valores negativos de descuento no son incorrectos per se, pero se observaron casos extremos en los que el precio final (`price`) supera el precio base (`base_price`), generando descuentos negativos no razonables.

**Ejemplo:**

| base\_price | price   | discount\_percent |
| ----------- | ------- | ----------------- |
| 450.00      | 6584.00 | -6584%            |

---

### 3.4. `conversion_rate`

La mayor parte de las publicaciones tiene una tasa de conversión igual a `0`, indicando que no registraron ventas durante el periodo observado. Esto no es necesariamente una inconsistencia, pero es un dato crítico para entender la eficiencia comercial del inventario.

**Distribución resumida:**

* Valor modal: `0.0`
* Proporción de publicaciones sin ventas: \~83%

---

## 4. Observaciones finales

El dataset presenta un volumen manejable de valores nulos en campos clave, y varias inconsistencias semánticas que podrían afectar los resultados analíticos si no se interpretan correctamente. Las inconsistencias en campos categóricos como `seller_loyalty` y `status` son especialmente relevantes, así como los casos extremos en los descuentos calculados.

