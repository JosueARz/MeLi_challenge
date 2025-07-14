"""
scripts/analyze_results_simple.py
---------------------------------
Análisis esencial del modelo
- ROC-AUC, PR-AUC, F1, matriz de confusión
- Intervalos 95 % (bootstrap)
- Lift por deciles
- Curvas ROC y PR
- Informe Markdown en outputs/model_analysis.md
"""

from __future__ import annotations
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, roc_curve
)
import joblib

# -------- ubicaciones -------------------------------------------------
ROOT  = Path(__file__).resolve().parent.parent
DATA  = ROOT / "data" / "processed" / "items_with_selected_features.csv"
MODEL = ROOT / "models" / "best_pipeline.pkl"
OUT   = ROOT / "reports"
OUT.mkdir(exist_ok=True)

BOOTSTRAP = 1000          # iteraciones bootstrap
TEST_FRAC = 0.20          # mismo 20 % usado en train

# -------- cargar modelo + meta ----------------------------------------
pipeline, meta = joblib.load(MODEL)
COLS       = meta["feature_cols"]
THRESHOLD  = float(meta.get("threshold_opt", 0.60))

# -------- dataset de prueba ------------------------------------------
df = pd.read_csv(DATA)
df_test = df.sample(frac=TEST_FRAC, random_state=42)

X_test = df_test[COLS]
y_test = df_test["sold_flag"].to_numpy()
proba  = pipeline.predict_proba(X_test)[:, 1]
pred   = (proba >= THRESHOLD).astype(int)

# -------- métricas principales ---------------------------------------
roc_auc = roc_auc_score(y_test, proba)
pr_auc  = average_precision_score(y_test, proba)

# IC-95 % con bootstrap
rng = np.random.default_rng(0)
roc_samples, pr_samples = [], []
for _ in range(BOOTSTRAP):
    idx = rng.integers(0, len(y_test), len(y_test))
    roc_samples.append(roc_auc_score(y_test[idx], proba[idx]))
    pr_samples.append(average_precision_score(y_test[idx], proba[idx]))

roc_lo, roc_hi = np.percentile(roc_samples, [2.5, 97.5])
pr_lo,  pr_hi  = np.percentile(pr_samples,  [2.5, 97.5])

report_str = classification_report(y_test, pred, digits=3)
cm         = confusion_matrix(y_test, pred)

# -------- lift por deciles -------------------------------------------
df_lift = pd.DataFrame({"proba": proba, "y": y_test})
df_lift["decile"] = pd.qcut(df_lift["proba"].rank(method="first"),
                            10, labels=False)
lift_tbl = (df_lift.groupby("decile")["y"].mean()
            .sort_index(ascending=False)
            .to_frame("rate"))
lift_tbl["lift"] = lift_tbl["rate"] / df_lift["y"].mean()
lift_md = lift_tbl.reset_index().to_markdown(index=False, floatfmt=".3f")

# -------- curvas ------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, proba)
rec, prec, _ = precision_recall_curve(y_test, proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
plt.legend(); plt.tight_layout()
plt.savefig(OUT / "roc_curve.png"); plt.close()

plt.figure()
plt.plot(rec, prec, label=f"PR AUC {pr_auc:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve")
plt.legend(); plt.tight_layout()
plt.savefig(OUT / "pr_curve.png"); plt.close()

# -------- informe Markdown -------------------------------------------
md = f"""
# Model evaluation

| Metric | Value | 95 % CI |
|--------|------:|--------:|
| ROC-AUC | **{roc_auc:.3f}** | [{roc_lo:.3f}, {roc_hi:.3f}] |
| PR-AUC  | **{pr_auc:.3f}** | [{pr_lo:.3f}, {pr_hi:.3f}] |

**Threshold usado:** {THRESHOLD:.2f}

## Classification report
{report_str}
## Confusion matrix
TN = {cm[0,0]}, FP = {cm[0,1]}
FN = {cm[1,0]}, TP = {cm[1,1]}


## Lift by decile
{lift_md}

Curvas ROC y PR guardadas en `outputs/`.
"""
(OUT / "model_analysis.md").write_text(textwrap.dedent(md), encoding="utf-8")
print("Informe completo en outputs/model_analysis.md")
