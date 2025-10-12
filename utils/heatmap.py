# utils/heatmap.py
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

# Projeden sabit SF ofsetini al
try:
    from .constants import SF_TZ_OFFSET
except Exception:
    SF_TZ_OFFSET = -7  # güvenli varsayılan

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def _to_sf(dt_utc: datetime) -> datetime:
    return dt_utc + timedelta(hours=SF_TZ_OFFSET)

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _detect_value_col(df: pd.DataFrame) -> str | None:
    """base_profile için metrik kolonunu bul."""
    prefer = ["expected", "lambda", "intensity", "rate", "mean", "base"]
    for c in prefer:
        if c in df.columns:
            return c
    # sayı kolonları içinden ilkini al
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for drop in ["geoid", "GEOID", "GeoID", "cell_id", "id", "index"]:
        if drop in num:
            num.remove(drop)
    return num[0] if num else None

def _hours_in_horizon(start_iso: str | None, H: int) -> pd.DataFrame:
    """Ufuktaki SF saatlerini (dow,hour) frekans tablosu olarak döndür."""
    try:
        start = pd.to_datetime(start_iso) if start_iso else datetime.utcnow()
    except Exception:
        start = datetime.utcnow()
    start = start.replace(minute=0, second=0, microsecond=0)
    start_sf = _to_sf(start)
    hours = [start_sf + timedelta(hours=i) for i in range(max(1, H))]
    freq = (
        pd.DataFrame({"dow": [h.weekday() for h in hours], "hour": [h.hour for h in hours]})
        .value_counts(["dow", "hour"])
        .rename("freq")
        .reset_index()
    )
    return freq

def _profile_from_events(events_df: pd.DataFrame, weeks: int = 8) -> pd.DataFrame | None:
    """Ham olaylardan (ts) 8 haftalık dow×hour profil çıkar."""
    if not isinstance(events_df, pd.DataFrame) or events_df.empty or "ts" not in events_df.columns:
        return None
    df = events_df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    # SF yereline çevir
    df["ts_sf"] = df["ts"].dt.tz_convert(None) + pd.Timedelta(hours=SF_TZ_OFFSET)
    cutoff = df["ts_sf"].max() - pd.Timedelta(weeks=weeks)
    df = df[df["ts_sf"] >= cutoff]
    if df.empty:
        return None
    prof = (
        pd.DataFrame({
            "dow": df["ts_sf"].dt.weekday,
            "hour": df["ts_sf"].dt.hour,
            "cnt": 1.0
        })
        .groupby(["dow", "hour"], as_index=False)["cnt"].sum()
        .rename(columns={"cnt": "profile"})
    )
    return prof

def _normalize_shares(df: pd.DataFrame, col: str, eps: float = 1e-9) -> pd.Series:
    s = _safe_num(df[col])
    tot = float(s.sum())
    if tot <= eps:
        return pd.Series(np.full(len(df), 1.0 / len(df)), index=df.index)
    return s / tot

# ─────────────────────────────────────────────────────────────────────────────
# Ana fonksiyon
# ─────────────────────────────────────────────────────────────────────────────

def render_heatmap(
    agg: pd.DataFrame,
    start_iso: str | None,
    horizon_h: int | None,
    base_profile: pd.DataFrame | None = None,
    events_df: pd.DataFrame | None = None,
    mode: str = "Haftalık (7×24)",  # "Günlük (7×1)" | "Saatlik (1×24)"
):
    """
    - Eğer model saatlik döküm vermiyorsa sum(agg.expected) şehir toplamını:
      AĞIRLIK = (ufuktaki saatlerin (dow,hour) frekansı) × (tarihsel dow×hour profil)
      ile dağıtır.
    - `mode` ile görünümü seçersin.
    """

    if agg is None or agg.empty:
        st.info("Isı matrisi için veri yok.")
        return pd.DataFrame()

    H = max(1, int(horizon_h or 24))
    total_expected = float(_safe_num(agg.get("expected", pd.Series(dtype=float))).sum())

    # 1) Ufuk frekansı
    freq = _hours_in_horizon(start_iso, H)  # (dow,hour,freq)

    # 2) Tarihsel profil (öncelik: base_profile → events_df)
    prof = None
    if isinstance(base_profile, pd.DataFrame) and not base_profile.empty:
        vcol = _detect_value_col(base_profile)
        if vcol and {"dow", "hour"}.issubset(base_profile.columns):
            prof = (
                base_profile.groupby(["dow", "hour"], as_index=False)[vcol]
                .sum()
                .rename(columns={vcol: "profile"})
            )
    if prof is None:
        prof = _profile_from_events(events_df)

    # 3) Ağırlıklar
    weights = freq.copy()
    if prof is not None:
        weights = weights.merge(prof, on=["dow", "hour"], how="left")
        if "profile" not in weights or weights["profile"].isna().all():
            weights["profile"] = 1.0
        else:
            # eksikleri medyanla doldur
            med = float(_safe_num(weights["profile"]).median() or 1.0)
            weights["profile"] = _safe_num(weights["profile"]).replace(0.0, med).fillna(med)
        weights["w"] = _safe_num(weights["freq"]) * _safe_num(weights["profile"])
    else:
        weights["w"] = _safe_num(weights["freq"])

    weights["share"] = _normalize_shares(weights, "w")
    weights["E_city"] = total_expected * weights["share"]

    # 4) Pivot ve gösterim
    if mode.startswith("Haftalık"):
        mat = (
            weights.pivot(index="dow", columns="hour", values="E_city")
            .reindex(range(7))
            .fillna(0.0)
        )
        mat.index = DOW_NAMES
        mat.columns = [f"{h:02d}" for h in mat.columns]
        st.dataframe(mat.round(2), use_container_width=True)
        arr = mat.to_numpy()
        i, j = np.unravel_index(np.argmax(arr), arr.shape)
        st.caption(f"Toplam beklenen: {total_expected:.2f} • En yoğun: {mat.index[i]} {mat.columns[j]}")
        return mat

    elif mode.startswith("Günlük"):
        # Gün toplamları (sütun sayısı 1)
        daily = (
            weights.groupby("dow", as_index=False)["E_city"].sum()
            .set_index("dow")
            .reindex(range(7))
            .fillna(0.0)
            .rename(columns={"E_city": "Total"})
        )
        daily.index = DOW_NAMES
        st.dataframe(daily.round(2), use_container_width=True)
        i = int(np.argmax(daily["Total"].to_numpy()))
        st.caption(f"Toplam beklenen: {total_expected:.2f} • En yoğun gün: {daily.index[i]}")
        return daily

    else:  # Saatlik (1×24): dow ayrımı olmadan saat profili
        hourly = (
            weights.groupby("hour", as_index=False)["E_city"].sum()
            .set_index("hour")
            .reindex(range(24))
            .fillna(0.0)
            .T
        )
        hourly.columns = [f"{h:02d}" for h in hourly.columns]
        st.dataframe(hourly.round(2), use_container_width=True)
        j = int(np.argmax(hourly.to_numpy()[0]))
        st.caption(f"Toplam beklenen: {total_expected:.2f} • En yoğun saat: {hourly.columns[j]}")
        return hourly
