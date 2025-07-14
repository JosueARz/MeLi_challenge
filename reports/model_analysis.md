
# Model evaluation

| Metric | Value | 95 % CI |
|--------|------:|--------:|
| ROC-AUC | **0.841** | [0.834, 0.849] |
| PR-AUC  | **0.630** | [0.615, 0.646] |

**Threshold usado:** 0.60

## Classification report
              precision    recall  f1-score   support

           0      0.903     0.919     0.911     16670
           1      0.555     0.507     0.530      3328

    accuracy                          0.850     19998
   macro avg      0.729     0.713     0.720     19998
weighted avg      0.845     0.850     0.848     19998

## Confusion matrix
TN = 15317, FP = 1353
FN = 1641, TP = 1687


## Lift by decile
|   decile |   rate |   lift |
|---------:|-------:|-------:|
|    9.000 |  0.688 |  4.131 |
|    8.000 |  0.282 |  1.698 |
|    7.000 |  0.209 |  1.259 |
|    6.000 |  0.184 |  1.106 |
|    5.000 |  0.133 |  0.799 |
|    4.000 |  0.058 |  0.346 |
|    3.000 |  0.037 |  0.219 |
|    2.000 |  0.029 |  0.177 |
|    1.000 |  0.032 |  0.192 |
|    0.000 |  0.012 |  0.072 |

Curvas ROC y PR guardadas en `outputs/`.
