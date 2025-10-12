# reports/monthly.py
from __future__ import annotations
import pandas as pd
from typing import Optional, List, Dict, Tuple

from .builder import (  # göreli import
    daily_tables,
    planned_patrols_table,
    heuristic_suggestions,
    export_report_zip,
)

def _infer_end(df: pd.DataFrame) -> pd.Timestamp:
    """Veride 'date' varsa en güncel günü, yoksa bugün (normalize)."""
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce").dropna()
        if len(d) > 0:
            return d.max().normalize()
    return pd.Timestamp.today().normalize()

def make_monthly(df: pd.DataFrame,
                 end: Optional[pd.Timestamp] = None,
                 cats: Optional[List[str]] = None
                 ) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
    """
    Aylık rapor: end dâhil son 30 gün (end-29 .. end).
    Dönenler:
      - tables: {'top_geoid','offense','by_hour','subset'}
      - tips:   öneri kartları listesi
    """
    end = pd.to_datetime(end or _infer_end(df)).normalize()
    d1 = (end - pd.Timedelta(days=29)).normalize()
    d2 = end

    tables = daily_tables(df, d1, d2, cats=cats)
    planned = planned_patrols_table(limit=1500)
    tips = heuristic_suggestions(
        df_sub=tables["subset"],
        top_geoid=tables["top_geoid"],
        planned_df=planned
    )
    return tables, tips

def export_monthly_zip(df: pd.DataFrame,
                       end: Optional[pd.Timestamp] = None,
                       cats: Optional[List[str]] = None) -> str:
    """
    Aylık raporu ZIP’e yazar ve dosya yolunu string olarak döndürür.
    ZIP adı: monthly_YYYY-MM-DD_YYYY-MM-DD_*.zip
    """
    end_norm = pd.to_datetime(end or _infer_end(df)).normalize()
    d1 = (end_norm - pd.Timedelta(days=29)).normalize()
    d2 = end_norm

    tables, tips = make_monthly(df, end=end_norm, cats=cats)
    prefix = f"monthly_{d1.date()}_{d2.date()}"
    zip_path = export_report_zip(prefix, tables, tips)
    return str(zip_path)
