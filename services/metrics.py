# services/metrics.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, Iterable

# ---------------------------------------------------------------------
# Ortak yardımcılar (DataFrame tabanlı metrikler için)
# ---------------------------------------------------------------------
def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def _pick_score_col(df: pd.DataFrame) -> str:
    for c in ("pred_expected", "pred_p_occ", "crime_count"):
        if c in df.columns:
            return c
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else df.columns[0]

def _recent_day(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if "date" not in df.columns:
        return None
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    return None if d.empty else d.max().normalize()

# ---------------------------------------------------------------------
# Vektör/array tabanlı metrikler
# ---------------------------------------------------------------------
def brier_score_vector(y_true: Iterable[float], p_hat: Iterable[float]) -> float:
    """
    Brier skoru: (p - y)^2 ortalaması.
    y_true: {0,1}, p_hat: [0,1]
    """
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(p_hat), dtype=float)
    return float(np.mean((p - y) ** 2))

def roc_auc(y_true: Iterable[int], score: Iterable[float]) -> Optional[float]:
    """
    Sklearn olmadan AUC (rank tabanlı). Tüm örnekler tek sınıfsa None döner.
    """
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(score), dtype=float)
    if y.size == 0 or y.min() == y.max():
        return None
    order = s.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(s)) + 1.0
    n1 = float((y == 1).sum()); n0 = float((y == 0).sum())
    if n0 == 0 or n1 == 0:
        return None
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return float(auc)

def calibration_table(y_true: Iterable[int], p_hat: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    """
    Kalibrasyon tablosu: bin başına ortalama tahmin vs gerçekleşen pozitif oranı.
    """
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(p_hat), dtype=float)
    if y.size == 0:
        return pd.DataFrame(columns=["bin","bin_lo","bin_hi","count","avg_pred","frac_pos"])
    p = np.clip(p, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins, right=True)
    recs = []
    for b in range(1, n_bins + 1):
        m = idx == b
        if m.sum() == 0:
            recs.append({"bin": b, "bin_lo": bins[b - 1], "bin_hi": bins[b],
                         "count": 0, "avg_pred": np.nan, "frac_pos": np.nan})
        else:
            recs.append({"bin": b, "bin_lo": bins[b - 1], "bin_hi": bins[b],
                         "count": int(m.sum()),
                         "avg_pred": float(p[m].mean()),
                         "frac_pos": float(y[m].mean())})
    return pd.DataFrame(recs)

def hit_rate_at_k_pairs(
    df_pred: pd.DataFrame,
    df_actual: pd.DataFrame,
    k: int = 10,
    key_col: str = "GEOID",
    actual_y_col: str = "crime_count",
) -> float:
    """
    Tahmin ve gerçekleşen pencereleri ayrık olduğunda HitRate@K.
      - df_pred: {GEOID, score} veya {GEOID, pred_*}
      - df_actual: {GEOID, crime_count} (>=1 olanlar pozitif kabul edilir)
    Dönüş: hits / min(pozitif hücre sayısı, k)
    """
    if key_col not in df_pred.columns or key_col not in df_actual.columns:
        return 0.0

    score_col = _pick_score_col(df_pred)
    topk = (df_pred[[key_col, score_col]]
            .dropna()
            .sort_values(score_col, ascending=False)
            .head(int(k)))

    if actual_y_col not in df_actual.columns:
        df_actual = df_actual.assign(**{actual_y_col: 1})

    pos_geoids = set(df_actual.loc[df_actual[actual_y_col] > 0, key_col].astype(str))
    top_geoids = set(topk[key_col].astype(str))
    if len(pos_geoids) == 0:
        return 0.0
    hits = len(top_geoids & pos_geoids)
    return float(hits / min(len(pos_geoids), k))

# ---------------------------------------------------------------------
# DataFrame tabanlı metrikler (tarih filtresi, kolon seçimi otomatik)
# ---------------------------------------------------------------------
def hitrate_at_k_df(
    df: pd.DataFrame,
    k: int = 10,
    geoid_col: str = "GEOID",
    label_col: str = "crime_count",
    score_col: Optional[str] = None,
) -> Optional[float]:
    """
    Son gün için: en yüksek skorlu K GEOID içinde gerçekleşen olay oranı.
    - Aynı DataFrame içinde skor & gerçekleşen değerler var varsayılır.
    - Son güne normalize edilerek hesaplanır (date yoksa tüm veri üzerinden).
    """
    if geoid_col not in df.columns or label_col not in df.columns:
        return None
    df = _ensure_date(df)
    day = _recent_day(df)
    if day is not None:
        df = df[pd.to_datetime(df["date"], errors="coerce").dt.normalize() == day]
    if df.empty:
        return None

    s_col = score_col or _pick_score_col(df)

    g = df.groupby(geoid_col, as_index=False).agg(
        score=(s_col, "sum"),
        y=(label_col, "sum"),
    ).sort_values("score", ascending=False)

    top = g.head(max(1, int(k)))
    total_y = float(g["y"].sum())
    if total_y <= 0:
        return None
    return float(top["y"].sum() / total_y)

def brier_score_df(df: pd.DataFrame) -> Optional[float]:
    """
    Basit Brier (DataFrame): p=pred_p_occ, y=(crime_count>0) ikili hedefi.
    Son 30 gün üzerinden hesaplanır (varsa).
    """
    if "pred_p_occ" not in df.columns:
        return None
    df = _ensure_date(df)
    if "date" in df.columns:
        day = _recent_day(df)
        if day is not None:
            start = day - pd.Timedelta(days=29)
            mask = (pd.to_datetime(df["date"], errors="coerce") >= start) & \
                   (pd.to_datetime(df["date"], errors="coerce") <= day)
            df = df[mask]
    if df.empty:
        return None

    p = pd.to_numeric(df["pred_p_occ"], errors="coerce")
    if "crime_count" in df.columns:
        y = (pd.to_numeric(df["crime_count"], errors="coerce").fillna(0) > 0).astype(int)
    else:
        return None
    m = (~p.isna()) & (~y.isna())
    if m.sum() == 0:
        return None
    return float(np.mean((p[m] - y[m]) ** 2))

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    UI için güvenli özet: HitRate@10 (son gün), Brier (son 30 gün), son gün tarihi.
    """
    hr = hitrate_at_k_df(df, k=10)
    br = brier_score_df(df)
    last_day = _recent_day(df)
    return {
        "hitrate_top10": None if hr is None else round(hr * 100, 1),  # yüzde
        "brier": None if br is None else round(br, 4),
        "last_day": None if last_day is None else str(last_day.date()),
    }
