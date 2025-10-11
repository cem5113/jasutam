from __future__ import annotations

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from utils.geo import load_geoid_layer, resolve_clicked_gid
from utils.forecast import precompute_base_intensity, aggregate_fast, prob_ge_k
from utils.patrol import allocate_patrols
from utils.ui import SMALL_UI_CSS, render_result_card, build_map_fast, render_kpi_row
try:
    from components.report_view import render_reports
    HAS_REPORTS = True
except ModuleNotFoundError:
    HAS_REPORTS = False
    def render_reports(**kwargs):
        import streamlit as st
        st.info("Raporlar modülü bulunamadı (components/report_view.py).")

# Isı matrisi: ayrı modül varsa oradan, yoksa ui'dan
try:
    from utils.heatmap import render_day_hour_heatmap
except ImportError:
    from utils.ui import render_day_hour_heatmap

# Pydeck yardımcıları
try:
    from utils.deck import build_map_fast_deck
except ImportError:
    build_map_fast_deck = None

from utils.constants import (
    SF_TZ_OFFSET, KEY_COL,
    MODEL_VERSION, MODEL_LAST_TRAIN,
    CATEGORIES,
)
from components.last_update import show_last_update_badge

try:
    from utils.reports import load_events
except Exception:
    # Son çare fallback: basit CSV okuyucu (uygun kolonu ts yapar)
    def load_events(path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        lower = {str(c).strip().lower(): c for c in df.columns}
        for cand in ["ts", "timestamp", "datetime", "date_time", "reported_at", "occurred_at", "time", "date"]:
            if cand in lower:
                ts_col = lower[cand]
                break
        else:
            df["ts"] = pd.NaT
            return df.dropna(subset=["ts"])
        df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])
        if "latitude" not in df.columns and "lat" in df.columns:
            df = df.rename(columns={"lat": "latitude"})
        if "longitude" not in df.columns and "lon" in df.columns:
            df = df.rename(columns={"lon": "longitude"})
        return df

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar: KEY_COL ve centroid sütunlarını savunmalı şekilde garanti et
# ─────────────────────────────────────────────────────────────────────────────
def ensure_keycol(df: pd.DataFrame, want: str = KEY_COL) -> pd.DataFrame:
    """DataFrame içinde KEY_COL adını garanti eder ve string tipe çevirir."""
    if df is None or df.empty:
        return df
    if want in df.columns:
        out = df.copy()
        out[want] = out[want].astype(str)
        return out
    alts = {want.upper(), want.lower(), "GEOID", "geoid", "GeoID"}
    hit = None
    for c in df.columns:
        if c in alts:
            hit = c
            break
    out = df.copy()
    if hit is not None and hit != want:
        out = out.rename(columns={hit: want})
    if want in out.columns:
        out[want] = out[want].astype(str)
    return out

def ensure_centroid_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Centroid kolon adlarını standardize eder."""
    if df is None or df.empty:
        return df
    out = df.copy()
    rename_map = {}
    if "centroid_lat" not in out.columns:
        if "Centroid_Lat" in out.columns: rename_map["Centroid_Lat"] = "centroid_lat"
        if "CENTROID_LAT" in out.columns: rename_map["CENTROID_LAT"] = "centroid_lat"
        if "lat" in out.columns and "centroid_lon" in out.columns:
            rename_map["lat"] = "centroid_lat"
    if "centroid_lon" not in out.columns:
        if "Centroid_Lon" in out.columns: rename_map["Centroid_Lon"] = "centroid_lon"
        if "CENTROID_LON" in out.columns: rename_map["CENTROID_LON"] = "centroid_lon"
        if "lon" in out.columns and "centroid_lat" in out.columns:
            rename_map["lon"] = "centroid_lon"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out
# ─────────────────────────────────────────────────────────────────────────────

# ── Sayfa ayarı
st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli", layout="wide")
st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)

# ── Başlık ve rozet
st.title("SUTAM: Suç Tahmin Modeli")

try:
    events_df = load_events("data/events.csv")
    st.session_state["events_df"] = events_df if isinstance(events_df, pd.DataFrame) else None
    st.session_state["events"] = st.session_state["events_df"]

    if isinstance(events_df, pd.DataFrame) and not events_df.empty and "ts" in events_df.columns:
        data_upto_val = pd.to_datetime(events_df["ts"]).max().date().isoformat()
    else:
        data_upto_val = None
except Exception:
    st.session_state["events_df"] = None
    st.session_state["events"] = None
    data_upto_val = None

show_last_update_badge(
    data_upto=data_upto_val,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

# ── Geo katmanı
GEO_DF, GEO_FEATURES = load_geoid_layer("data/sf_cells.geojson")
GEO_DF = ensure_keycol(ensure_centroid_cols(GEO_DF), KEY_COL)
if GEO_DF.empty:
    st.error("GEOJSON yüklenemedi veya satır yok.")
    st.stop()

# ── Model tabanı
BASE_INT = precompute_base_intensity(GEO_DF)

def now_sf_iso() -> str:
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")

# ── Sidebar
st.sidebar.markdown("### Görünüm")
sekme_options = ["Operasyon"]
if HAS_REPORTS:
    sekme_options.append("Raporlar")
sekme = st.sidebar.radio("", options=sekme_options, index=0, horizontal=True)
st.sidebar.divider()

# ---- AYARLAR ----
st.sidebar.header("Devriye Parametreleri")
engine = st.sidebar.radio("Harita motoru", ["Folium", "pydeck"], index=0, horizontal=True)

# Harita katmanları
show_popups = st.sidebar.checkbox("Hücre popup'larını (en olası 3 suç) göster", value=True)
# POI/Transit şu an kullanılmadığı için güvenli varsayılanlar:
show_poi = False
show_transit = False

# Grafik kapsamı (istatistikler için)
scope = st.sidebar.radio("Grafik kapsamı", ["Tüm şehir", "Seçili hücre"], index=0)

# Hotspot ayarları
show_hotspot      = True
show_temp_hotspot = True

# Tek kategori seçimi → hem tahmin hem hotspot
selected_category = st.sidebar.selectbox(
    "Kategori (tahmin & hotspot)",
    options=["(Tüm suçlar)"] + CATEGORIES,
    index=0,
    help="Tahmin ve hotspot katmanları aynı kategoriye göre filtrelenir."
)

# Zaman aralığı
ufuk = st.sidebar.radio(
    "Zaman Aralığı (şimdiden)",
    options=["0–24 saat", "72 saat", "1 hafta"],
    index=0, horizontal=True
)
max_h, step = (24, 1) if ufuk == "0–24 saat" else (72, 3) if ufuk == "72 saat" else (7*24, 24)
start_h, end_h = st.sidebar.slider("Saat filtresi", min_value=0, max_value=max_h, value=(0, max_h), step=step)

# Kategori filtresi
filters = {"cats": None if selected_category == "(Tüm suçlar)" else [selected_category]}

show_advanced = st.sidebar.checkbox("Gelişmiş metrikleri göster (analist)", value=False)

st.sidebar.divider()
st.sidebar.subheader("Devriye Parametreleri")
K_planned    = st.sidebar.number_input("Planlanan devriye sayısı (K)", min_value=1, max_value=50, value=6, step=1)
duty_minutes = st.sidebar.number_input("Devriye görev süresi (dk)",   min_value=15, max_value=600, value=120, step=15)
cell_minutes = st.sidebar.number_input("Hücre başına ort. kontrol (dk)", min_value=2, max_value=30, value=6, step=1)

colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et")
btn_patrol  = colB.button("Devriye öner")
# -----------------

# ── State
if "agg" not in st.session_state:
    st.session_state.update({
        "agg": None, "patrol": None, "start_iso": None, "horizon_h": None, "explain": None
    })

# ── Operasyon
if sekme == "Operasyon":
    col1, col2 = st.columns([2.4, 1.0])

    with col1:
        st.caption(f"Son güncelleme (SF): {now_sf_iso()}")

        if btn_predict or st.session_state["agg"] is None:
            start_dt  = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET + start_h)).replace(minute=0, second=0, microsecond=0)
            horizon_h = max(1, end_h - start_h)
            start_iso = start_dt.isoformat()

            events_df = load_events("data/events.csv")  # ts, lat, lon kolonları olmalı
            st.session_state["events_df"] = events_df

            # Tahmin
            agg = aggregate_fast(
                start_iso, horizon_h, GEO_DF, BASE_INT,
                events=events_df,
                near_repeat_alpha=0.35,
                nr_lookback_h=24,
                nr_radius_m=400,
                nr_decay_h=12.0,
                filters=filters,
            )

            # === 5 kademeli tier ataması (Çok Hafif → Çok Yüksek) ===
            try:
                if isinstance(agg, pd.DataFrame) and "expected" in agg.columns:
                    q20, q40, q60, q80 = np.quantile(agg["expected"].to_numpy(), [0.20, 0.40, 0.60, 0.80])
                    bins   = [-np.inf, q20, q40, q60, q80, np.inf]
                    labels = ["Çok Hafif", "Hafif", "Düşük", "Orta", "Çok Yüksek"]
                    agg = agg.copy()
                    agg["tier"] = pd.cut(agg["expected"], bins=bins, labels=labels, include_lowest=True).astype(str)
            except Exception:
                pass

            # KEY_COL standardizasyonu
            agg = ensure_keycol(agg, KEY_COL)

            st.session_state.update({
                "agg": agg,
                "patrol": None,
                "start_iso": start_iso,
                "horizon_h": horizon_h,
                "events": events_df,
            })

            # Uzun ufuk referansı (raporlar için)
            try:
                long_start_iso = (
                    datetime.utcnow()
                    + timedelta(hours=SF_TZ_OFFSET - 30*24)
                ).replace(minute=0, second=0, microsecond=0).isoformat()

                agg_long = aggregate_fast(
                    long_start_iso, 30*24, GEO_DF, BASE_INT,
                    events=events_df,
                    near_repeat_alpha=0.0,
                    filters=None
                )
                st.session_state["agg_long"] = ensure_keycol(agg_long, KEY_COL)
            except Exception:
                st.session_state["agg_long"] = None

        agg = st.session_state["agg"]

        # Geçici hotspot verisi (tek kategori ile filtreli)
        events_all = st.session_state.get("events")
        lookback_h = int(np.clip(2 * st.session_state.get("horizon_h", 24), 24, 72))

        ev_recent_df = None
        if isinstance(events_all, pd.DataFrame) and not events_all.empty:
            ev_recent_df = events_all.copy()
            _ts = "ts" if "ts" in ev_recent_df.columns else ("timestamp" if "timestamp" in ev_recent_df.columns else None)
            ev_recent_df["ts"] = pd.to_datetime(ev_recent_df[_ts], utc=True, errors="coerce") if _ts else pd.NaT
            if "ts" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df[ev_recent_df["ts"] >= (pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_h))]
            if selected_category != "(Tüm suçlar)" and "type" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df[ev_recent_df["type"] == selected_category]
            if "latitude" not in ev_recent_df.columns and "lat" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df.rename(columns={"lat": "latitude"})
            if "longitude" not in ev_recent_df.columns and "lon" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df.rename(columns={"lon": "longitude"})
            ev_recent_df = ev_recent_df.dropna(subset=["latitude", "longitude"])
            if not ev_recent_df.empty:
                ev_recent_df["weight"] = 1.0

        # Grafik kapsamı için veri seti
        if isinstance(ev_recent_df, pd.DataFrame) and not ev_recent_df.empty:
            keep_cols = [c for c in ["ts", "latitude", "longitude", KEY_COL] if c in ev_recent_df.columns]
            df_plot = ev_recent_df[keep_cols].copy()
        else:
            df_plot = pd.DataFrame(columns=["ts", "latitude", "longitude"])

        if scope == "Seçili hücre" and st.session_state.get("explain", {}).get("geoid"):
            gid = str(st.session_state["explain"]["geoid"])
            if KEY_COL in df_plot.columns:
                df_plot = df_plot[df_plot[KEY_COL].astype(str) == gid]

        # Geçici hotspot HeatMap girdisi
        if isinstance(ev_recent_df, pd.DataFrame) and not ev_recent_df.empty:
            temp_points = ev_recent_df[["latitude", "longitude"]].copy()
            temp_points["weight"] = ev_recent_df["weight"] if "weight" in ev_recent_df.columns else 1.0
        else:
            temp_points = pd.DataFrame(columns=["latitude", "longitude", "weight"])

        # Fallback: üst risk hücrelerinden sentetik ısı (→ BURADA normalize ET!)
        if show_temp_hotspot and temp_points.empty and isinstance(agg, pd.DataFrame) and not agg.empty:
            topn = 80
            agg2 = ensure_keycol(agg, KEY_COL)
            geo2 = ensure_keycol(ensure_centroid_cols(GEO_DF), KEY_COL)
            try:
                tmp = (
                    agg2.nlargest(topn, "expected")
                        .merge(geo2[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL, how="left")
                        .dropna(subset=["centroid_lat", "centroid_lon"])
                )
                temp_points = tmp.rename(columns={"centroid_lat": "latitude", "centroid_lon": "longitude"})[
                    ["latitude", "longitude"]
                ]
                temp_points["weight"] = tmp["expected"].clip(lower=0).astype(float)
            except Exception as e:
                st.warning(f"Hotspot fallback oluşturulamadı: {e}")
                temp_points = pd.DataFrame(columns=["latitude", "longitude", "weight"])

        st.sidebar.caption(f"Geçici hotspot noktası: {len(temp_points)}")

        if agg is not None:
            if engine == "Folium":
                # Folium harita
                m = build_map_fast(
                    df_agg=agg,
                    geo_features=GEO_FEATURES,
                    geo_df=GEO_DF,
                    show_popups=show_popups,
                    patrol=st.session_state.get("patrol"),
                    show_hotspot=show_hotspot,
                    perm_hotspot_mode="heat",
                    show_temp_hotspot=show_temp_hotspot,
                    temp_hotspot_points=temp_points,
                )
                import folium
                assert isinstance(m, folium.Map), f"st_folium beklediği tipte değil: {type(m)}"

                ret = st_folium(
                    m,
                    key="riskmap",
                    height=540,
                    width=1600,  # genişlik
                    returned_objects=["last_object_clicked", "last_clicked"]
                )
                if ret:
                    gid, _ = resolve_clicked_gid(GEO_DF, ret)
                    if gid:
                        st.session_state["explain"] = {"geoid": gid}

            else:
                # Pydeck harita
                if build_map_fast_deck is None:
                    st.error("Pydeck harita modülü bulunamadı (utils/deck.py). Lütfen Folium motorunu seçin.")
                    ret = None
                else:
                    deck = build_map_fast_deck(
                        agg, GEO_DF,
                        show_poi=show_poi,
                        show_transit=show_transit,
                        patrol=st.session_state.get("patrol"),
                        show_hotspot=show_hotspot,
                        show_temp_hotspot=show_temp_hotspot,
                        temp_hotspot_points=temp_points,
                    )
                    st.pydeck_chart(deck)
                    ret = None

            # Açıklama kartı
            start_iso  = st.session_state["start_iso"]
            horizon_h  = st.session_state["horizon_h"]
            st.caption("Ufuk: seçilen saat aralığı (SF).")

            info = st.session_state.get("explain")
            if info and info.get("geoid"):
                render_result_card(agg, info["geoid"], start_iso, horizon_h)
            else:
                st.info("Haritada bir hücreye tıklayın veya listeden seçin; kart burada görünecek.")
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

    with col2:
        st.subheader("Risk Özeti", anchor=False)

        if st.session_state["agg"] is not None:
            a = st.session_state["agg"]
            kpi_expected = round(float(a["expected"].sum()), 2)

            cnt_cok_yuksek = int((a["tier"] == "Çok Yüksek").sum())
            cnt_orta       = int((a["tier"] == "Orta").sum())
            cnt_dusuk      = int((a["tier"] == "Düşük").sum())
            cnt_hafif      = int((a["tier"] == "Hafif").sum())
            cnt_cok_hafif  = int((a["tier"] == "Çok Hafif").sum())

            render_kpi_row([
                ("Beklenen olay (ufuk)", kpi_expected, "Seçili zaman ufkunda toplam beklenen olay sayısı"),
                ("Çok Yüksek",          cnt_cok_yuksek, "En yüksek riskli hücre sayısı (üst %20)"),
                ("Orta",                cnt_orta,       "Orta kademe riskli hücre sayısı"),
                ("Düşük",               cnt_dusuk,      "Düşük kademe riskli hücre sayısı"),
                ("Hafif",               cnt_hafif,      "Hafif kademe riskli hücre sayısı"),
                ("Çok Hafif",           cnt_cok_hafif,  "En düşük riskli hücre sayısı (alt %20)"),
            ])
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

        st.subheader("Top-5 kritik GEOID")
        if st.session_state["agg"] is not None:

            def top_risky_table(df_agg: pd.DataFrame, n: int = 5, show_ci: bool = False) -> pd.DataFrame:
                # Poisson ~%95 güven aralığı (normal approx.)
                def poisson_ci(lam: float, z: float = 1.96) -> tuple[float, float]:
                    s = float(np.sqrt(max(lam, 1e-9)))
                    return max(0.0, lam - z * s), lam + z * s

                cols = [KEY_COL, "expected"]
                if "nr_boost" in df_agg.columns:
                    cols.append("nr_boost")

                # KEY_COL güvenliği
                df_agg2 = ensure_keycol(df_agg, KEY_COL)

                tab = (
                    df_agg2[cols]
                    .sort_values("expected", ascending=False)
                    .head(n).reset_index(drop=True)
                )

                lam = tab["expected"].to_numpy()
                tab["P(≥1)%"] = [round(prob_ge_k(l, 1) * 100, 1) for l in lam]

                # Saat aralığı (SF)
                start_iso_val = st.session_state.get("start_iso")
                try:
                    if start_iso_val:
                        _start = pd.to_datetime(start_iso_val)
                        _end   = _start + pd.to_timedelta(st.session_state.get("horizon_h", 0), unit="h")
                        start_hh = _start.strftime("%H:%M")
                        end_hh   = _end.strftime("%H:%M")
                        tab["Saat"] = f"{start_hh}–{end_hh} (SF)"
                    else:
                        tab["Saat"] = "-"
                except Exception:
                    tab["Saat"] = "-"

                if show_ci:
                    ci_vals = [poisson_ci(float(l)) for l in lam]
                    tab["95% Güven Aralığı"] = [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in ci_vals]

                if "nr_boost" in tab.columns:
                    tab["NR"] = tab["nr_boost"].round(2)

                tab["E[olay] (λ)"] = tab["expected"].round(2)

                drop_cols = ["expected"]
                if "nr_boost" in tab.columns:
                    drop_cols.append("nr_boost")
                return tab.drop(columns=drop_cols)

            st.dataframe(
                top_risky_table(st.session_state["agg"], n=5, show_ci=show_advanced),
                use_container_width=True, height=300
            )
            if not show_advanced:
                st.caption("P(≥1)%: Seçilen ufukta en az bir olay olma olasılığı.")
            else:
                st.caption(
                    "95% Güven Aralığı: Aynı koşullar tekrarlansa, gerçek sayının ~%95 bu aralıkta kalması beklenir. "
                    "Hızlı hesap: λ ± 1.96·√λ (alt sınır 0'a kırpılır)."
                )

        st.subheader("Devriye özeti")
        if st.session_state.get("agg") is not None and btn_patrol:
            st.session_state["patrol"] = allocate_patrols(
                st.session_state["agg"], GEO_DF,
                k_planned=int(K_planned),
                duty_minutes=int(duty_minutes),
                cell_minutes=int(cell_minutes),
                travel_overhead=0.40
            )
        patrol = st.session_state.get("patrol")
        if patrol and patrol.get("zones"):
            rows = [{
                "zone": z["id"],
                "cells_planned": z["planned_cells"],
                "capacity_cells": z["capacity_cells"],
                "eta_minutes": z["eta_minutes"],
                "utilization_%": z["utilization_pct"],
                "avg_risk(E[olay])": round(z["expected_risk"], 2),
            } for z in patrol["zones"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

        st.subheader("Gün × Saat Isı Matrisi")
        if st.session_state.get("agg") is not None and st.session_state.get("start_iso"):
            render_day_hour_heatmap(st.session_state["agg"],
                                    st.session_state.get("start_iso"),
                                    st.session_state.get("horizon_h"))
        else:
            st.caption("Isı matrisi, bir tahmin üretildiğinde gösterilir.")

        st.subheader("Dışa aktar")
        if st.session_state["agg"] is not None:
            csv = st.session_state["agg"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "CSV indir", data=csv,
                file_name=f"risk_export_{int(time.time())}.csv",
                mime="text/csv"
            )

elif sekme == "Raporlar":
    agg_current = st.session_state.get("agg")
    agg_long    = st.session_state.get("agg_long")

    events_src = st.session_state.get("events")
    if not isinstance(events_src, pd.DataFrame) or events_src.empty:
        events_src = st.session_state.get("events_df")

    render_reports(
        events_df     = events_src,
        agg_current   = agg_current,
        agg_long_term = agg_long,
    )
