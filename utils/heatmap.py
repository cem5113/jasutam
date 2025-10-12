# utils/heatmap.py
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from .constants import SF_TZ_OFFSET

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def _as_sf(dt_utc: datetime) -> datetime:
    # UTC naive -> SF offset (sabit)
    return dt_utc + timedelta(hours=SF_TZ_OFFSET)

def render_day_hour_heatmap(agg: pd.DataFrame, start_iso: str | None, horizon_h: int | None):
    """
    7x24 (gün × saat) ısı matrisi üretir.
    Şehir toplam beklenen olayı (sum(agg.expected)) ufuk içindeki her (dow,hour)
    gerçekleşme frekansına orantılı dağıtır. Böylece Mon..Sun satırları aynı çıkmaz.
    """
    if agg is None or agg.empty:
        st.info("Isı matrisi için veri yok.")
        return pd.DataFrame()

    # 1) Başlangıç ve ufuk
    try:
        start = pd.to_datetime(start_iso) if start_iso else datetime.utcnow()
    except Exception:
        start = datetime.utcnow()
    H = int(horizon_h or 24)
    if H <= 0:
        H = 24

    # 2) SF saatine göre saat dizisi (ufuk boyunca)
    start_sf = _as_sf(start.replace(minute=0, second=0, microsecond=0) - timedelta(hours=SF_TZ_OFFSET))
    hours = [start_sf + timedelta(hours=i) for i in range(H)]

    # 3) (dow,hour) frekansları
    cnt = (
        pd.DataFrame({"dow": [h.weekday() for h in hours], "hour": [h.hour for h in hours]})
        .value_counts(["dow", "hour"])
        .rename("count")
        .reset_index()
    )
    cnt["share"] = cnt["count"] / cnt["count"].sum()

    # 4) Şehir toplam beklenen olay
    total_expected = float(pd.to_numeric(agg.get("expected", pd.Series(dtype=float)), errors="coerce")
                           .clip(lower=0).sum())
    cnt["E_city"] = total_expected * cnt["share"]

    # 5) 7×24 pivot tablo
    mat = (
        cnt.pivot(index="dow", columns="hour", values="E_city")
        .reindex(range(7))
        .fillna(0.0)
    )
    mat.index = DOW_NAMES
    mat.columns = [f"{h:02d}" for h in mat.columns]

    # 6) Görüntüle
    st.dataframe(mat.round(2), use_container_width=True)
    # En yoğun (dow,hour)
    arr = mat.to_numpy()
    if arr.size:
        i, j = np.unravel_index(np.argmax(arr), arr.shape)
        st.caption(f"Toplam beklenen: {total_expected:.2f} • En yoğun: {mat.index[i]} {mat.columns[j]}")
    else:
        st.caption(f"Toplam beklenen: {total_expected:.2f}")

    return mat
