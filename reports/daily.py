# reports/daily.py
from __future__ import annotations
import pandas as pd
from typing import Optional, List, Dict, Tuple

from .builder import (   # ← paket içi göreli import
    daily_tables,
    planned_patrols_table,
    heuristic_suggestions,
    export_report_zip,
)

def _infer_end(df: pd.DataFrame) -> pd.Timestamp:
    """Veride 'date' varsa en güncel günü, yoksa bugün."""
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce").dropna()
        if len(d) > 0:
            return d.max().normalize()
    return pd.Timestamp.today().normalize()

def make_daily(df: pd.DataFrame,
               end: Optional[pd.Timestamp] = None,
               cats: Optional[List[str]] = None
               ) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
    """
    Günlük rapor: end gününü kapsar (00:00–23:59). d1 = d2 = end.
    Dönenler:
      - tables: {'top_geoid','offense','by_hour','subset'}
      - tips:   öneri kartları listesi
    """
    end = pd.to_datetime(end or _infer_end(df)).normalize()
    d1, d2 = end, end

    tables = daily_tables(df, d1, d2, cats=cats)
    planned = planned_patrols_table(limit=500)  # approvals.jsonl → tablo
    tips = heuristic_suggestions(
        df_sub=tables["subset"],
        top_geoid=tables["top_geoid"],
        planned_df=planned
    )
    return tables, tips

def export_daily_zip(df: pd.DataFrame,
                     end: Optional[pd.Timestamp] = None,
                     cats: Optional[List[str]] = None) -> str:
    """
    Günlük raporu ZIP’e yazar ve dosya yolunu string olarak döndürür.
    ZIP adı: daily_YYYY-MM-DD_*.zip
    """
    end_norm = pd.to_datetime(end or _infer_end(df)).normalize()
    tables, tips = make_daily(df, end=end_norm, cats=cats)
    # tarihli önek
    prefix = f"daily_{end_norm.date()}"
    zip_path = export_report_zip(prefix, tables, tips)
    return str(zip_path)
