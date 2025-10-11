# services/metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import json

import pandas as pd


# JSON olarak tutulacak güncel KPI dosyası (app.py'de caption'da da JSON deniyordu)
METRICS_FILE = Path("data/metrics.json")


def get_latest_metrics() -> Dict[str, Any]:
    """
    data/metrics.json varsa oku ve dict döndür.
    Yoksa boş dict döndür (app.py bu durumu already handle ediyor).
    """
    try:
        if METRICS_FILE.exists():
            with METRICS_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Temel tip doğrulaması
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _select_row(df: pd.DataFrame, *, hit_col: Optional[str], prefer_group: Optional[str]) -> pd.Series:
    """
    Metrik CSV'sinden 'en iyi' satırı seçmek için basit bir seçim mantığı.
    Öncelik sırası:
      1) prefer_group belirtilmişse o gruba filtrele (örn: 'stacking')
      2) hit_col (örn: 'hit_rate@100' veya 'hit_rate_topk') varsa en yüksek değeri seç
      3) pr_auc varsa en yüksek
      4) auc varsa en yüksek
      5) brier varsa en düşük
      6) aksi halde ilk satır
    """
    cand = df.copy()

    # 1) Grup tercihi
    if prefer_group and "group" in cand.columns:
        sub = cand[cand["group"].astype(str) == str(prefer_group)]
        if not sub.empty:
            cand = sub

    def best_by(col: str, ascending: bool = False) -> Optional[pd.Series]:
        if col in cand.columns:
            try:
                return cand.sort_values(col, ascending=ascending).iloc[0]
            except Exception:
                return None
        return None

    # 2) hit_col
    if hit_col and hit_col in cand.columns:
        row = best_by(hit_col, ascending=False)
        if row is not None:
            return row

    # 3) pr_auc (yüksek iyi)
    row = best_by("pr_auc", ascending=False)
    if row is not None:
        return row

    # 4) auc (yüksek iyi)
    row = best_by("auc", ascending=False)
    if row is not None:
        return row

    # 5) brier (düşük iyi)
    row = best_by("brier", ascending=True)
    if row is not None:
        return row

    # 6) fallback
    return cand.iloc[0]


def update_from_csv(
    csv_path: Optional[str],
    *,
    hit_col: Optional[str] = None,
    prefer_group: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bir metrik CSV'sini okuyup en iyi satırı seçer ve METRICS_FILE'a (JSON) yazar.
    - csv_path None ise FileNotFoundError fırlatır (app.py bunu yakalıyor).
    - CSV'de beklenen olası kolonlar: ['model_name','group','pr_auc','auc','brier','hit_rate_topk','timestamp','source_path','selection_metric','selection_value', ...]
    """
    if not csv_path:
        raise FileNotFoundError("csv_path is None")

    # CSV'yi oku
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("metrics CSV is empty")

    # En iyi satırı seç
    best = _select_row(df, hit_col=hit_col, prefer_group=prefer_group)

    # Çıkış sözlüğünü derle
    out: Dict[str, Any] = {}

    # Sık kullanılan alanlar
    for col in ["model_name", "group", "pr_auc", "auc", "brier", "hit_rate_topk", "timestamp", "source_path"]:
        if col in best.index:
            val = best[col]
            # NaN → None
            if isinstance(val, float) and pd.isna(val):
                val = None
            out[col] = val

    # Seçim bilgisini açıkça yaz
    sel_metric = None
    sel_value = None
    if hit_col and hit_col in best.index and pd.notna(best[hit_col]):
        sel_metric = hit_col
        sel_value = float(best[hit_col])
    elif "pr_auc" in best.index and pd.notna(best["pr_auc"]):
        sel_metric = "pr_auc"
        sel_value = float(best["pr_auc"])
    elif "auc" in best.index and pd.notna(best["auc"]):
        sel_metric = "auc"
        sel_value = float(best["auc"])
    elif "brier" in best.index and pd.notna(best["brier"]):
        sel_metric = "brier"
        sel_value = float(best["brier"])

    if sel_metric is not None:
        out["selection_metric"] = sel_metric
        out["selection_value"] = sel_value

    # Dosyaya yaz
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_FILE.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out
