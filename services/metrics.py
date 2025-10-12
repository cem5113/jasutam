# services/metrics.py
from pathlib import Path
import pandas as pd, json
from typing import Dict, Any

def build_basic_metadata(df: pd.DataFrame, source: str, out_path: Path) -> Dict[str, Any]:
    date_col = "date" if "date" in df.columns else None
    meta = {
        "source": source,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "date_min": (df[date_col].min().isoformat() if date_col and not df.empty else None),
        "date_max": (df[date_col].max().isoformat() if date_col and not df.empty else None),
        "has_latlon": all(c in df.columns for c in ["lat","lon"]),
    }
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
