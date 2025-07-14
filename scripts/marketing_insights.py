"""
Extrae insights y los guarda en outputs/insights.md

- % de conversión por segmento de vendedor
- impacto del descuento en probabilidad de venta
- top 5 categorías por PR-AUC (si existiera category_id limpio)
"""

from pathlib import Path
import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent
DATA   = ROOT / "data" / "processed" / "items_with_features.csv"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# --------- Insight 1: tasa de conversión por loyalty ----------
conv_loyalty = (df.groupby("seller_loyalty")
                  .apply(lambda d: (d["sold_flag"].mean()*100).round(2))
                  .sort_values(ascending=False))

# --------- Insight 2: efecto descuento ----------
df["has_discount"] = (df["price"] < df["base_price"]).astype(int)
lift = (df.groupby("has_discount")["sold_flag"]
          .mean()
          .rename({0:"sin_desc", 1:"con_desc"}))

# --------- construir markdown ----------
md = ["## Insights para Marketing\n"]
md.append("### 1. Conversión por segmento de vendedor\n")
md.append(conv_loyalty.to_markdown())

md.append("\n### 2. Impacto del descuento (lift)\n")
lift_pct = ((lift[1] / lift[0]) - 1) * 100
md.append(f"- La publicación con descuento vende **{lift_pct:.1f}%** más que la sin descuento.\n")

(OUTDIR / "insights.md").write_text("\n".join(md))
print("Insights guardados en outputs/insights.md")
