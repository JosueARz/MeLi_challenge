````markdown
# Desafío Técnico – Data Scientist  


---

## 1 . Índice
1. [Índice](#Índice)
2. [Descripción general](#descripción-general)  
3. [Estructura del repositorio](#estructura-del-repositorio)  
4. [Flujo de trabajo](#flujo-de-trabajo)  
5. [Análisis exploratorio de datos (EDA)](#análisis-exploratorio-de-datos-eda)  
6. [Ingeniería de features](#ingeniería-de-features)  
7. [Modelado y resultados](#modelado-y-resultados)  
8. [Insights para Marketing & Negocio](#8-insights-para-marketing--negocio)
9. [Inferencia paso a paso](#inferencia-paso-a-paso)  
10. [Monitoreo en producción](#monitoreo-en-producción)  
11. [Persistencia BigQuery & Cloud Storage](#persistencia-bigquery--cloud-storage)  
12. [Limitaciones y futuro trabajo](#limitaciones-y-futuro-trabajo)  
 

---

## 2 . Descripción general
Este proyecto resuelve la prueba técnica de MercadoLibre (100 000 ítems, 26 columnas) con una **arquitectura modular y orientada a objetos**:

* **Código reproducible** (PEP-8, type-hints, docstrings).  
* **Separación de responsabilidades** — `DataAnalyzer`, `FeatureEngineer`, `ModelPredictor`.  
* **Automatización punta a punta** — un solo comando limpia, genera features, entrena y evalúa.  
* **Visión de negocio** — métricas y umbrales listos para Marketing & Operaciones.

---

## 3 . Estructura del repositorio
```text
mercadolibre-challenge/
│
├─ data/
│   ├─ raw/                 # CSV original
│   └─ processed/           # dataset limpio + features finales
│
├─ ml_models/               # clases DataAnalyzer, FeatureEngineer, ModelPredictor
├─ models/                  # artefactos .pkl del modelo
├─ scripts/                 # CLI: run_eda, run_featuring, run_models
├─ notebooks/               # predict.ipynb
├─ outputs/                 # gráficos EDA
├─ reports/                 # Detalle de eda y feature engineering
├─ requirements.txt
└─ README.md
````

---


## 4 . Flujo de trabajo

| Paso                | Comando                           | Resultado                                              |
| ------------------- | --------------------------------- | ------------------------------------------------------ |
| EDA                 | `python -m scripts.run_eda`       | Resumen estadístico, limpieza y gráficos en `outputs/` |
| Feature engineering | `python -m scripts.run_featuring` | CSV con features finales en `data/processed/`          |
| Modelado            | `python -m scripts.run_models`    | CV, test, umbral óptimo y modelo guardado en `models/` |

---

## 5 · Análisis exploratorio de datos (EDA)

La clase `DataAnalyzer` ejecuta más de 20 chequeos automáticos.  
A continuación se sintetizan **los hallazgos clave**.

### 5.1  Estadísticas descriptivas de las variables numéricas

| Métrica (n ≈ 100 k) | `base_price` | `price` | `initial_qty` | `sold_qty` | `avail_qty` |
|---------------------|-------------:|--------:|--------------:|-----------:|------------:|
| **Media**           | 55 067 MXN   | 52 528 MXN | 35.09 | 2.40 | 34.84 |
| **Mediana**         | 250          | 250      | 1    | 0    | 1    |
| **Desv. estándar**  | 8.63 M       | 8.62 M   | 421  | 42.69| 420  |
| **Máximo**          | 2.22 B       | 2.22 B   | 9 999| 8 676| 9 999 |

*Conclusión*: precios y cantidades exhiben colas largas; se aplicará `log1p` en el pipeline para estabilizar varianza.

---

### 5.2  Descuentos

| Estadístico | `discount_value` | `discount_percent` |
|-------------|-----------------:|-------------------:|
| **Media**   | –0.10            | –0.087            |
| **% con descuento** | 25 % | 25 % |

*75 % de las publicaciones se ofrecen sin descuento*.  
Casos donde `price > base_price` sugieren reajustes de precio o inconsistencias.

---

### 5.3  Conversión y tasa de ventas

* **83 %** de las publicaciones no registra ventas (`conversion_rate = 0`).  
* Distribución altamente sesgada con 1 740 valores distintos; la variable se normaliza a 0–1 en la fase de features.

---

### 5.4  Ventas por segmento de vendedor

| Seller loyalty | Precio medio (MXN) | Ventas totales |
|----------------|-------------------:|---------------:|
| **gold**            | 68 467 | 84 928 |
| **gold_premium**    | 225 596 | 39 342 |
| silver              | 272 553 | 65 430 |
| bronze              | 36 433 | 38 368 |
| free                | 4 882  | 2 606 |

Los segmentos **gold** y **gold_premium** aportan la mayoría de las ventas, validando la clasificación de lealtad de ML.

---

### 5.5  Ventas por estado de publicación

| Estado | Ventas % |
|--------|---------:|
| **active** | 93 % |
| paused     | 6 %  |
| closed + otros | 1 % |

*Publicaciones activas concentran casi todas las transacciones*; los estados “1” y valores nulos se corrigieron a `NaN`.

---

### 5.6  Valores faltantes

| Columna            | % faltante |
|--------------------|-----------:|
| `sub_status`       | **99 %** |
| `warranty`         | 61 % |
| Resto             | < 0.01 % |

*Acción*: columnas con > 60 % nulos no se usan para el MVP; los nulos críticos (`price`, `base_price`, cantidades) se imputan con la mediana de su categoría.

---

### 5.7  Inconsistencias de tipo

* Valores numéricos en `seller_loyalty`, `status`, `shipping_mode` → convertidos a `NaN`.
* Columnas JSON-like (`tags`, `attributes`, `variations`, `pictures`) quedan como *string* para futuros NLP.

---

### 5.8  Correcciones aplicadas

* `available_quantity` negativos → 0  
* `sold_quantity` nulo → 0  
* `initial_quantity`, `price`, `base_price` nulos → mediana por categoría  
* Limpieza de valores espurios en variables categóricas (`seller_loyalty`, `status`).

---

### 5.9  Outliers en precio

| Método | Filas marcadas |
|--------|---------------:|
| IQR (1.5× rango) | 14 635 |
| Z-score > 3       | 5 |
| Desviación relativa (`|price - base_price|`) | 25 |
| Caro + sin ventas (p95 & sold=0) | 4 576 |

*Acción*: se conservan para el modelo los registros con precios dentro de p1–p99; el resto pasa por winsorización automática de `HistGradientBoosting`.

---

> **Resumen EDA**  
> El dataset, tras la limpieza precedente, es **adecuado para modelado supervisado**.  
> Las variables más determinantes identificadas son:  
> 1) **precio normalizado** (`price_per_unit`),  
> 2) **rotación de inventario** (`stock_ratio`),  
> 3) **condición de escasez** (`is_scarce_stock`), y  
> 4) **estrato de precio** (`is_high_priced`).  
> Estas conclusiones guían la fase de ingeniería de features y la elección del modelo final.


---

## 6 . Ingeniería de features

| Variable          | Tipo | Intuición                          |
| ----------------- | ---- | ---------------------------------- |
| `price_per_unit`  | Num  | Valor por unidad ofertada          |
| `stock_ratio`     | Num  | Inventario restante / inicial      |
| `is_scarce_stock` | Bin  | Urgencia de compra (≤ 10 unidades) |
| `is_high_priced`  | Bin  | Segmenta productos premium         |

*Evaluación (F-test, Mutual Information, Random Forest) concluye que las cuatro variables anteriores capturan el 90 % de la señal predictiva.*

---

## 7 · Modelado y resultados

| Aspecto | Detalle |
|---------|---------|
| **Algoritmos evaluados** | Logistic Regression (baseline) y HistGradientBoosting (HGB). |
| **Selección** | **HGB + GridSearch** supera al baseline en todas las métricas. |
| **Variables de entrada** | 7 numéricas (sin fugas de información). |
| **Datos de validación** | 4-fold CV (PR-AUC) + test hold-out 20 %. |
| **Modelo guardado** | `models/best_pipeline.pkl`. |

### 7.1  Métricas finales (test set)

| Métrica (umbral 0.60) | Valor | Lectura rápida |
|-----------------------|------:|----------------|
| **ROC-AUC** | **0.833** | El ranking distingue 83 % de pares 0-vs-1 — robusto. |
| **PR-AUC**  | **0.623** | Triplica la línea base (0.17) pese al desbalance. |
| Precisión (1) | 0.562 | Más de la mitad de los ítems marcados “vendible” realmente venden. |
| Recall (1) | 0.495 | Capturamos ~50 % de todas las ventas potenciales. |
| **F1** | **0.526** | Buen equilibrio dadas las restricciones de ruido y sparsity. |
| Accuracy | 0.849 | Métrica auxiliar (dominada por la clase 0). |

> **Resumen**  
> *El modelo es **fiable para un MVP**:* identifica la mitad de los ítems que sí venderán con una precisión razonable, y su curva ROC indica un buen ordenamiento para fijar distintos umbrales según la tolerancia a riesgo del negocio.

---

### 7.2  Fortalezas actuales

- **Generaliza bien**: diferencia clara entre CV (0.625 PR-AUC) y test (0.623).
- **Sencillo de desplegar**: pipeline con ≤ 1 MB, sin dependencias exóticas.
- **Interpretabilidad**: HGB permite extraer importancia de features y explicar decisiones.

### 7.3  Deficiencias detectadas

- **Recall 50 %** puede resultar bajo si la prioridad es no perder ventas.
- **Sensibilidad a clases raras**: ítems de categorías muy poco frecuentes quedan infra-representados.
- **Sin features textuales ni de categoría**: el modelo solo mira variables numéricas; pierde contexto que podría aportar el título o la familia de producto.

### 7.4  Próximas palancas de mejora

1. **Clase minoritaria**  
   - Ajustar `class_weight` a {0:1, 1:6–8} o probar focal loss (LightGBM / XGBoost).
2. **Parámetros HGB**  
   - Grid fino en `min_samples_leaf`, `max_features`, `max_bins` y `l2_regularization`.
3. **Calibración**  
   - `CalibratedClassifierCV` (isotónica) para mejorar la fiabilidad de las probabilidades.
4. **Stacking**  
   - Ensemble con Logistic Regression sobre las probabilidades de HGB + LightGBM.

Con estos ajustes se espera elevar PR-AUC por encima de 0.70 y Recall ≥ 60 %
sin sacrificar demasiado la precisión, cumpliendo los objetivos de la siguiente
iteración del MVP.


---
### 8. Insights para Marketing & Negocio

> Basados en el **EDA** completo y en los resultados finales del modelo
> (ROC-AUC 0.84, PR-AUC 0.63, *lift* 4.1× en el decil TOP), se resumen los
> hallazgos más valiosos y cómo pueden transformarse en acciones concretas.

---

#### 8.1 Hallazgos clave y acciones recomendadas

| Insight                                                                                      | Evidencia                                                        | Acción táctica / estratégica                                                                                                                                                                             |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. El 20 % de las publicaciones concentra 57 % de las ventas potenciales**                 | Lift table ⇒ decil 9 con *lift* 4.1×                             | - Activar **campañas de visibilidad** (Ads / push) sólo a ítems del TOP-20 % para maximizar ROI.<br>- Aplicar badges “Best Seller Potential” en ficha de producto para incentivar compra.                |
| **2. Más del 80 % de los ítems nunca vende (conversión 0 %)**                                | EDA: `sold_quantity` mediana = 0; tasa de conversión global 17 % | - Diseñar un **programa de rescate de listings**: recomendaciones de precio, fotos, títulos SEO.<br>- Enviar reportes automatizados al vendedor con “Next best actions”.                                 |
| **3. Precio relativo importa más que el descuento nominal**                                  | Feature importance: `price_per_unit` > descuento                 | - Comunicar a Pricing la conveniencia de mostrar **precio/volumen** junto al precio total.<br>- Ajustar estrategias de “precio ancla” en categorías sensibles a unidad de medida (p. ej. supermercados). |
| **4. Productos con stock ≤ 10 unidades convierten mejor**                                    | Variables `is_scarce_stock`, `stock_ratio` con F-score alto      | - Añadir un mensaje “¡Últimas unidades!” cuando el inventario ≤ 10.<br>- Priorizar estos SKUs en e-mails de urgencia (FOMO).                                                                             |
| **5. Segmentos de vendedor “gold / gold premium” generan ticket medio ↑ y mayor conversión** | Tabla de ventas por `seller_loyalty`                             | - Crear **Marketplace Deals** exclusivos para vendedores top: envío gratis + posicionamiento.<br>- Incentivar a vendedores “silver/bronze” a mejorar SLA para subir de nivel.                            |
| **6. Dispersión extrema de precios → outliers que dañan la experiencia**                     | Máx 2.22 B MXN; outliers detectados por IQR y Z-score            | - Implementar validación automática de precios al publicar (> p99 de la categoría dispara revisión).                                                                                                     |

---

#### 8.2 Cómo usar el modelo en decisiones diarias

1. **Priorizar inventario promocionado**
   *Input:* probabilidad de venta `proba_sale` del modelo.
   *Regla:* si `proba_sale` ≥ 0.60 ⇒ incluir en la parrilla del home / newsletters; caso contrario ocultar o reciclar.

2. **Gestión dinámica del *paid traffic***

   * Ajustar pujas de CPC con `bid = base_bid × proba_sale`.
   * Resultado esperado: +12 % ingresos / –18 % gasto publicitario (simulación con lift 4×).

3. **Atención al vendedor predictiva**

   * Trigger de alerta a cuentas con > 100 publicaciones **baja probabilidad** (< 0.2).
   * El CRM sugiere cambio de título, fotos y precio referente a comparables del TOP-deciles.

---

#### 8.3 Casos de uso adicionales y técnicas recomendadas

| Caso de uso                           | Objetivo de negocio                       | Datos / Técnica                                                          | Justificación                                                         |
| ------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| **Re-pricing automático**             | Incrementar margen manteniendo conversión | *Features* de mercado + `price_per_unit`; algoritmo **Bayesian Bandits** | Ajusta precios en tiempo real y explora sin castigar ingresos.        |
| **Recomendador de bundle**            | Aumentar ticket medio                     | Historial de co-compra; **Association Rules** / **Word2Vec items2vec**   | Sugiere packs de alto margen basados en afinidad.                     |
| **Detección de fraude en listings**   | Reducir contracargos y devoluciones       | Outliers + texto (`title`, `warranty`) con **Isolation Forest**          | Pre-filtra precios irreales y descripciones potencialmente engañosas. |
| **Segmentación de vendedores**        | Programas de loyalty personalizados       | KPIs de ventas, SLA, cancelaciones; **K-Means** o **Gaussian Mixture**   | Permite incentivos diferenciados y políticas de comisión escalonadas. |
| **Forecast de demanda por categoría** | Optimizar logística y stock               | Series temporales (Prophet / SARIMA) + `conversion_rate`                 | Planifica inventario y warehousing, evitando ruptura de stock.        |

---

> **Próximo MVP:** integrar *features* de texto (TF-IDF del título) y categoría
> jerárquica; recalibrar el modelo cada mes con **CalibratedClassifierCV** para
> asegurar probabilidades confiables a lo largo del tiempo.

Con estos insights, Marketing y Operaciones disponen de un mapa de acción claro para aumentar ventas, optimizar inversión publicitaria y mejorar la experiencia de vendedores y compradores.

---

## 9 . Inferencia paso a paso

Ejemplo completo usado en `notebooks/predict.ipynb`:

```python
%cd ../                 # 1) ubicarse en la raíz
from pathlib import Path
import pandas as pd
from ml_models.model import ModelPredictor

pipe = ModelPredictor.load_model(Path("models/best_pipeline.pkl"))
df_new = pd.read_csv("data/processed/items_with_features.csv")

SAFE_COLS = ["base_price","price","is_new","initial_quantity","price_per_unit"]
proba = pipe.predict_proba(df_new[SAFE_COLS].head(10))[:, 1]
pred  = (proba >= 0.62).astype(int)

print("Probabilidades:", proba)
print("Predicciones  :", pred)
```

---

## 10 . Monitoreo en producción (métricas off-line)

* **Crédito de datos**: cada scoring guarda *ID, timestamp, features, probabilidad*.
* **Batch nocturno (Airflow)**

  * *Performance*: ROC-AUC, PR-AUC, F1 vs. ventas reales del día anterior.
  * *Deriva de concepto*: test Page-Hinkley sobre pérdida log.
  * *Negocio*: `lift` respecto a grupo control.
* **Alertas**: caída de PR-AUC > 5 p.p., PSI > 0.20 o `lift` < +1 p.p. siete días → notificación Slack/Jira y playbook de retrain.

---

## 11 . Persistencia BigQuery & Cloud Storage

* **Dataset**

  * Crear dataset `ml_challenge` en BigQuery.
  * Cargar tabla `items_features` (modo *replace* diario, columna `ingestion_date`).
* **Modelo**

  * Subir `best_pipeline.pkl` a `gs://ml-model-artifacts/best_pipeline.pkl`.
  * Registrar versión en tabla `models_registry` con fecha, URI, ROC-AUC y umbral óptimo.
* **Inferencia**

  * Servicios descargan el .pkl desde la URI y predicen con `joblib.load`.
  * Para lotes masivos es viable un modelo nativo BigQuery ML y `ML.PREDICT`.
* **Ventajas**: centralización, escalabilidad y trazabilidad de versiones.

---

## 12 . Limitaciones y futuro trabajo

1. Incorporar texto del título (TF-IDF) y categorías jerárquicas.
2. Probar LightGBM / CatBoost con `scale_pos_weight`.
3. Calibrar probabilidades con `CalibratedClassifierCV`.
4. Automatizar retrain mensual y agregar alertas de deriva en tiempo real.

---