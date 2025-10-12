# services/metrics.py
from __future__ import annotations

import os
import io
import json
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, Tuple
from zipfile import ZipFile, BadZipFile
import pandas as pd

# -----------------------------------------------------------------------------
# Çözümleyiciler
# -----------------------------------------------------------------------------
def _resolve_artifact_zip_or_dir(
    artifact_zip: Optional[str] = None,
    artifact_dir: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    1) Parametreler
    2) ENV: SUTAM_ARTIFACT_ZIP ya da SUTAM_ARTIFACT_DIR
    3) Fallback: data/artifacts/sf-crime-pipeline-output.zip  VEYA data/artifacts/
    """
    z = Path(artifact_zip) if artifact_zip else (Path(os.environ.get("SUTAM_ARTIFACT_ZIP")) if os.environ.get("SUTAM_ARTIFACT_ZIP") else None)
    d = Path(artifact_dir) if artifact_dir else (Path(os.environ.get("SUTAM_ARTIFACT_DIR")) if os.environ.get("SUTAM_ARTIFACT_DIR") else None)

    if not z and not d:
        default_dir = Path("data/artifacts")
        default_zip = default_dir / "sf-crime-pipeline-output.zip"
        if default_zip.exists():
            z = default_zip
        elif default_dir.exists():
            d = default_dir

    return (z if (z and z.exists()) else None,
            d if (d and d.exists()) else None)


# App caption’da göstermek için dışarıya sunalım
def artifact_location() -> str:
    z, d = _resolve_artifact_zip_or_dir()
    return str(z or d or "N/A")


# -----------------------------------------------------------------------------
# Dosya okuma yardımcıları
# -----------------------------------------------------------------------------
_CANDIDATE_BASENAMES: Tuple[str, ...] = (
    "metrics_all",
    "metrics_stacking",
    "metrics_base",
    "metrics_base_ohe",
    "metrics_stacking_ohe",
)

_EXTS: Tuple[str, ...] = (".csv", ".tsv", ".xlsx", ".parquet")


def _iter_artifact_files(zip_path: Optional[Path], dir_path: Optional[Path]) -> Iterable[Tuple[str, bytes]]:
    """
    Artifact içindeki dosyaları (ad, içerik-bytes) olarak üretir.
    ZIP varsa ZIP içinden; yoksa klasörden.
    """
    if zip_path:
        try:
            with ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    # yalnızca dosyalar
                    if name.endswith("/"):
                        continue
                    yield name, zf.read(name)
        except BadZipFile:
            return
    elif dir_path:
        for p in dir_path.rglob("*"):
            if p.is_file():
                try:
                    yield str(p.relative_to(dir_path)), p.read_bytes()
                except Exception:
                    continue


def _is_candidate(name: str) -> bool:
    low = name.lower()
    return any(b in low for b in _CANDIDATE_BASENAMES) and any(low.endswith(ext) for ext in _EXTS)


def _read_table(name: str, data: bytes) -> Optional[pd.DataFrame]:
    low = name.lower()
    try:
        if low.endswith(".csv"):
            return pd.read_csv(io.BytesIO(data))
        if low.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(data), sep="\t")
        if low.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(data))
        if low.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(data))
    except Exception:
        return None
    return None


# -----------------------------------------------------------------------------
# Seçim mantığı
# -----------------------------------------------------------------------------
def _select_row(df: pd.DataFrame, *, hit_col: Optional[str], prefer_group: Optional[str]) -> pd.Series:
    cand = df.copy()

    # grup filtresi
    if prefer_group and "group" in cand.columns:
        sub = cand[cand["group"].astype(str) == str(prefer_group)]
        if not sub.empty:
            cand = sub

    def best_by(col: str, ascending: bool = False) -> Optional[pd.Series]:
        if col in cand.columns:
            try:
                return cand.sort_values(col, ascending=ascending, kind="mergesort").iloc[0]
            except Exception:
                return None
        return None

    if hit_col and hit_col in cand.columns and cand[hit_col].notna().any():
        row = best_by(hit_col, ascending=False)
        if row is not None:
            return row

    for col, asc in (("pr_auc", False), ("auc", False), ("brier", True)):
        row = best_by(col, ascending=asc)
        if row is not None:
            return row

    return cand.iloc[0]


def _summarize_row(row: pd.Series, *, hit_col: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in ["model_name", "group", "pr_auc", "auc", "brier", "hit_rate_topk", "timestamp", "source_path"]:
        if col in row.index:
            val = row[col]
            if isinstance(val, float) and pd.isna(val):
                val = None
            out[col] = val

    # seçim bilgisi
    sel_metric, sel_value = None, None
    if hit_col and hit_col in row.index and pd.notna(row[hit_col]):
        sel_metric, sel_value = hit_col, float(row[hit_col])
    elif "pr_auc" in row.index and pd.notna(row["pr_auc"]):
        sel_metric, sel_value = "pr_auc", float(row["pr_auc"])
    elif "auc" in row.index and pd.notna(row["auc"]):
        sel_metric, sel_value = "auc", float(row["auc"])
    elif "brier" in row.index and pd.notna(row["brier"]):
        sel_metric, sel_value = "brier", float(row["brier"])

    if sel_metric is not None:
        out["selection_metric"] = sel_metric
        out["selection_value"] = sel_value
    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def get_latest_metrics_from_artifact(
    *,
    artifact_zip: Optional[str] = None,
    artifact_dir: Optional[str] = None,
    hit_col: Optional[str] = None,          # örn: "hit_rate@100" veya "hit_rate_topk"
    prefer_group: Optional[str] = None,     # örn: "stacking"
) -> Dict[str, Any]:
    """
    Sadece ARTIFACT içinden (ZIP veya klasör) okur.
    Candidate dosyalar: metrics_all, metrics_stacking, metrics_base, *_ohe (csv/tsv/xlsx/parquet)
    En iyi satırı seçer ve özet dict döndürür.
    """
    z, d = _resolve_artifact_zip_or_dir(artifact_zip, artifact_dir)

    # tüm candidate tablolardan concat
    tables: list[pd.DataFrame] = []
    for name, blob in _iter_artifact_files(z, d):
        if not _is_candidate(name):
            continue
        df = _read_table(name, blob)
        if df is None or df.empty:
            continue
        # tabloya kaynak bilgisi ekleyelim
        df = df.copy()
        df["source_path"] = name
        tables.append(df)

    if not tables:
        return {}

    big = pd.concat(tables, ignore_index=True, sort=False)
    row = _select_row(big, hit_col=hit_col, prefer_group=prefer_group)
    return _summarize_row(row, hit_col=hit_col)
