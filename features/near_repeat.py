# features/near_repeat.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Literal

# ------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------
def _ensure_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """date kolonunu güvenle datetime'a çevir ve NaN tarihleri düş."""
    if date_col not in df.columns:
        return df.iloc[0:0].copy()
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    return d.dropna(subset=[date_col])

def _pick_value_col(df: pd.DataFrame) -> str:
    """
    Tercih sırası:
    - pred_expected (beklenen olay)
    - pred_p_occ    (olasılık)
    - crime_count   (gerçek sayım)
    """
    for c in ("pred_expected", "pred_p_occ", "crime_count"):
        if c in df.columns:
            return c
    # hiçbirini bulamazsa ilk sayısal kolonu dön
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols[0] if num_cols else df.columns[0]

def _is_probability(col_name: str) -> bool:
    name = col_name.lower()
    return name.startswith("pred_p_") or name.endswith("_prob") or name == "pred_p_occ"

# ------------------------------------------------------------
# Geçici (temporal) hotspot
# ------------------------------------------------------------
def compute_temp_hotspot(
    df: pd.DataFrame,
    window_days: int = 2,
    baseline_days: int = 30,
    date_col: str = "date",
    geoid_col: str = "GEOID",
    value_col: Optional[str] = None,
    min_baseline_days: int = 7,
    method: Literal["ratio", "zscore"] = "ratio",
) -> pd.DataFrame:
    """
    Son 'window_days' penceresindeki yoğunluğu, önceki 'baseline_days' tabanına göre anomali skoru üretir.

    Dönen kolonlar:
      GEOID, temp_score, temp_score_norm, recent, expected, baseline_mean, base_days,
      method, flag_low_baseline, flag_fallback

    Notlar:
      - value_col = pred_expected / pred_p_occ / crime_count (otomatik seçilir)
      - Olasılık kolonlarında günlük toplama 'mean', sayımlarda 'sum' yapılır.
      - 'ratio' varsayılanı: (recent - expected) / max(1, expected)
      - 'zscore': (recent - expected) / std_baseline (std yoksa şehir ortalamasıyla fall-back)
    """
    d = _ensure_date(df, date_col)
    if d.empty or geoid_col not in d.columns:
        return pd.DataFrame(columns=[geoid_col, "temp_score"])

    d = d.copy()
    d[geoid_col] = d[geoid_col].astype(str)
    value_col = value_col or _pick_value_col(d)
    is_prob = _is_probability(value_col)

    end = d[date_col].max().normalize()
    start_recent = end - pd.Timedelta(days=window_days - 1)
    start_base = end - pd.Timedelta(days=baseline_days)

    # Günlük seriye indir
    daily = (
        d.groupby([geoid_col, d[date_col].dt.date], as_index=False)[value_col]
         .agg("mean" if is_prob else "sum")
         .rename(columns={value_col: "value", date_col: "day"})
    )
    daily["day"] = pd.to_datetime(daily["day"])

    # Pencereler
    rec = (daily[(daily["day"] >= start_recent) & (daily["day"] <= end)]
           .groupby(geoid_col, as_index=False)["value"]
           .agg("mean" if is_prob else "sum")
           .rename(columns={"value": "recent"}))

    base = daily[(daily["day"] >= start_base) & (daily["day"] < start_recent)]
    base_mean = (base.groupby(geoid_col, as_index=False)["value"].mean()
                    .rename(columns={"value": "baseline_mean"}))
    base_days = base.groupby(geoid_col)["day"].nunique().rename("base_days").reset_index()

    out = rec.merge(base_mean, on=geoid_col, how="left").merge(base_days, on=geoid_col, how="left")
    out[["baseline_mean", "base_days"]] = out[["baseline_mean", "base_days"]].fillna({"baseline_mean": 0.0, "base_days": 0})

    # Yetersiz taban için şehir ortalamasıyla fallback
    city_base_mean = float(base["value"].mean()) if len(base) else 0.0
    low_base_mask = out["base_days"] < int(min_baseline_days)
    out["flag_low_baseline"] = low_base_mask
    out["flag_fallback"] = False

    # expected = baseline_mean * window_days (prob için: mean * window_days kabulü)
    expected = out["baseline_mean"] * float(window_days)

    fb_mask = low_base_mask | (out["baseline_mean"] <= 0)
    if city_base_mean > 0:
        expected = np.where(fb_mask, city_base_mean * float(window_days), expected)
        out.loc[fb_mask, "flag_fallback"] = True
    else:
        expected = np.where(fb_mask, 1.0, expected)  # son çare: sıfıra bölmeyi önle

    # Skor
    if method == "zscore":
        base_std_geo = (base.groupby(geoid_col, as_index=False)["value"].std()
                           .rename(columns={"value": "baseline_std"}))
        out = out.merge(base_std_geo, on=geoid_col, how="left").fillna({"baseline_std": 0.0})
        city_std = float(base["value"].std()) if len(base) else 0.0
        denom = out["baseline_std"].where(out["baseline_std"] > 0, city_std if city_std > 0 else 1.0)
        out["temp_score"] = (out["recent"] - expected) / denom
    else:
        denom = np.maximum(1.0, expected)
        out["temp_score"] = (out["recent"] - expected) / denom

    # 0..1 normalize (UI renkleri için)
    m = out["temp_score"].min()
    M = out["temp_score"].max()
    out["temp_score_norm"] = (out["temp_score"] - m) / (M - m) if M > m else 0.0
    out["method"] = method

    return (out[[geoid_col, "temp_score", "temp_score_norm", "recent", "baseline_mean", "base_days",
                 "method", "flag_low_baseline", "flag_fallback"]]
            .sort_values("temp_score", ascending=False)
            .reset_index(drop=True))

# ------------------------------------------------------------
# Kalıcı (istikrarlı) hotspot
# ------------------------------------------------------------
def compute_stable_hotspot(
    df: pd.DataFrame,
    horizon_days: int = 90,
    date_col: str = "date",
    geoid_col: str = "GEOID",
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Son 'horizon_days' penceresinde istikrarlı yoğunluk skoru üretir.

    Dönen kolonlar:
      GEOID, stable_score, stable_score_norm, agg_value

    Not:
      - Olasılık kolonlarında günlük 'mean', sayımlarda 'sum' toplanır.
      - Skor z-benzeri; 0..1 normalize alanı UI için hazır.
    """
    d = _ensure_date(df, date_col)
    if d.empty or geoid_col not in d.columns:
        return pd.DataFrame(columns=[geoid_col, "stable_score"])

    d = d.copy()
    d[geoid_col] = d[geoid_col].astype(str)
    value_col = value_col or _pick_value_col(d)
    is_prob = _is_probability(value_col)

    end = d[date_col].max().normalize()
    start = end - pd.Timedelta(days=horizon_days - 1)

    scope = d[(d[date_col] >= start) & (d[date_col] <= end)]
    if scope.empty:
        return pd.DataFrame(columns=[geoid_col, "stable_score"])

    daily = (scope.groupby([geoid_col, scope[date_col].dt.date], as_index=False)[value_col]
                  .agg("mean" if is_prob else "sum")
                  .rename(columns={value_col: "value", date_col: "day"}))

    agg_geo = (daily.groupby(geoid_col, as_index=False)["value"]
                    .agg("mean" if is_prob else "sum")
                    .rename(columns={"value": "agg_value"}))

    mu = float(agg_geo["agg_value"].mean())
    sd = float(agg_geo["agg_value"].std(ddof=0))
    if sd <= 0 or np.isnan(sd):
        z = agg_geo["agg_value"] - mu
    else:
        z = (agg_geo["agg_value"] - mu) / sd

    # z'yi 0..1’a ölçekle
    z = z - z.min()
    vmax = float(z.max())
    z01 = z / vmax if vmax > 0 else 0.0

    agg_geo["stable_score"] = z
    agg_geo["stable_score_norm"] = z01

    return (agg_geo[[geoid_col, "stable_score", "stable_score_norm", "agg_value"]]
            .sort_values("stable_score", ascending=False)
            .reset_index(drop=True))
