# utils/heatmap.py 
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import timedelta
import streamlit as st
from utils.constants import SF_TZ_OFFSET

def render_day_hour_heatmap(
    agg: pd.DataFrame,
    start_iso: str | None,
    horizon_h: int | None,
    events_df: pd.DataFrame | None = None,
):
    """
    7×24 ısı matrisi:
      - Öncelik: agg ('dow','hour' varsa) → doğrudan pivot
      - Aksi: seçilen ufuktaki saatlere göre toplam λ'yı (agg.sum) dağıt
              (önce events tabanlı (dow,hour) profil, yoksa default 7×24)
      - Ufuk > 168 ise uyarı ver ve ilk 168 saati hesapla
    """
    if agg is None or agg.empty:
        st.caption("Isı matrisi için veri yok.")
        return

    # ❶ Önce saatlik veri var mı? → doğrudan pivot (en doğru senaryo)
    if {"dow", "hour"}.issubset(agg.columns):
        mat = (
            agg.pivot_table(index="dow", columns="hour", values="expected", aggfunc="sum")
              .reindex(index=range(7), columns=range(24), fill_value=0.0)
        )
        mat.index  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        mat.columns = [f"{h:02d}" for h in range(24)]
        st.dataframe(mat.round(2), use_container_width=True)
        tot = float(np.nansum(mat.values))
        i, j = np.unravel_index(np.nanargmax(mat.values), mat.shape)
        st.caption(f"Toplam beklenen: {tot:.2f} • En yoğun: {mat.index[i]} {mat.columns[j]}")
        return

    # ❷ Hızlı mod: agg toplam λ
    total_expected = float(pd.to_numeric(agg.get("expected", pd.Series(dtype=float)),
                                         errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(0).clip(lower=0).sum())
    if total_expected <= 0:
        st.caption("Toplam beklenen 0; ısı matrisi üretilemedi.")
        return

    if start_iso is None or horizon_h is None:
        st.caption("Başlangıç/zaman ufku gelmedi; varsayım: şu an + 24 saat.")
    H = int(horizon_h or 24)
    if H > 7*24:
        st.info("Uyarı: 7×24 matrise sığdığı için yalnızca ilk 168 saat gösteriliyor.")
        H = 7*24

    # Ufuktaki GERÇEK saatleri SF yereline çevir
    start = pd.to_datetime(start_iso) if start_iso else pd.Timestamp.utcnow()
    start = start.floor("H")
    hours = [start + pd.to_timedelta(i, "h") + pd.Timedelta(hours=SF_TZ_OFFSET) for i in range(H)]

    # ❸ (dow,hour) Ağırlık tablosu: önce events profili, yoksa default profil
    W = None
    if isinstance(events_df, pd.DataFrame) and not events_df.empty and "ts" in events_df.columns:
        tmp = events_df[["ts"]].copy()
        tmp["ts"] = pd.to_datetime(tmp["ts"], utc=True, errors="coerce")
        tmp = tmp.dropna(subset=["ts"])
        if not tmp.empty:
            tmp["ts_sf"] = tmp["ts"] + pd.Timedelta(hours=SF_TZ_OFFSET)
            tmp["dow"]   = tmp["ts_sf"].dt.dayofweek
            tmp["hour"]  = tmp["ts_sf"].dt.hour
            prof = (tmp.value_counts(["dow","hour"]).rename("cnt").reset_index()
                      .pivot(index="dow", columns="hour", values="cnt")
                      .reindex(index=range(7), columns=range(24)).fillna(0.0))
            if prof.values.sum() > 0:
                W = prof.values.astype(float)

    if W is None:
        # Default 7×24 (hafta içi gün modifikasyonları + diurnal iki harmonik)
        h = np.arange(24, dtype=float)
        diurnal = 1.0 + 0.45*np.sin(((h-18)/24)*2*np.pi) + 0.15*np.sin(((h-6)/24)*4*np.pi)
        diurnal -= diurnal.min()
        diurnal /= diurnal.sum() + 1e-12
        dow_mod = np.array([0.96, 0.99, 1.03, 1.00, 1.06, 1.12, 1.14], dtype=float)
        dow_mod /= dow_mod.sum()/7.0
        W = dow_mod[:, None] * diurnal[None, :]

    # ❹ Ufuktaki saatler için ağırlık vektörü ve dağıtım
    w_vec = np.array([W[t.weekday(), t.hour] for t in hours], dtype=float)
    w_vec = w_vec / (w_vec.sum() + 1e-12)

    mat = pd.DataFrame(0.0, index=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                       columns=[f"{h:02d}" for h in range(24)])
    for t, ww in zip(hours, w_vec):
        mat.loc[mat.index[t.weekday()], f"{t.hour:02d}"] += total_expected * float(ww)

    st.dataframe(mat.round(2), use_container_width=True)
    arr = mat.to_numpy()
    i, j = np.unravel_index(np.argmax(arr), arr.shape)
    st.caption(f"Toplam beklenen: {total_expected:.2f} • En yoğun: {mat.index[i]} {mat.columns[j]}")
