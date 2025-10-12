# app.py — birleşik (Lite + Pro) sürüm
# - "Lite" panel: dataio.loaders varsa hızlı önizleme + metrikler + basit harita
# - "Pro (SUTAM)" paneli: utils/* modülleri varsa gelişmiş tahmin/harita/raporlar
# Her iki modül de opsiyonel; bulunamazsa zarifçe düşer ve kullanıcıya bilgi verir.

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────── Opsiyonel bağımlılıklar ───────────────────────────────
# data loader (Lite panel)
try:
    from dataio.loaders import load_sf_crime_latest, RESULTS_DIR  # type: ignore
except Exception:  # noqa: BLE001
    load_sf_crime_latest = None  # type: ignore
    RESULTS_DIR = Path("results")

# last update (Lite panel)
try:
    from components.last_update import get_last_update  # type: ignore
except Exception:  # noqa: BLE001
    def get_last_update(_results_dir: Path) -> Optional[dict]:
        meta = _results_dir / "metadata.json"
        if not meta.exists():
            return None
        try:
            j = pd.read_json(meta)
            # pandas read_json on file returns a Series if dict-like; normalize
            if isinstance(j, pd.Series):
                j = j.to_dict()
            elif isinstance(j, pd.DataFrame) and j.shape[0] == 1:
                j = j.iloc[0].to_dict()
            return {
                "date_min": j.get("date_min"),
                "date_max": j.get("date_max"),
                "rows": j.get("rows"),
                "source": j.get("src"),
                "when": j.get("updated_at_sf") or j.get("last_trained_at"),
            }
        except Exception:
            return None

# constants (Pro panel)
try:
    from utils.constants import SF_TZ_OFFSET, KEY_COL, MODEL_VERSION, MODEL_LAST_TRAIN, CATEGORIES  # type: ignore
except Exception:  # noqa: BLE001
    SF_TZ_OFFSET = -7
    KEY_COL = "geoid"
    MODEL_VERSION = "v0.4.0"
    MODEL_LAST_TRAIN = datetime.utcnow().strftime("%Y-%m-%d")
    CATEGORIES = []

# geo helpers (Pro)
try:
    from utils.geo import load_geoid_layer, resolve_clicked_gid, join_neighborhood  # type: ignore
except Exception:  # noqa: BLE001
    load_geoid_layer = None  # type: ignore
    resolve_clicked_gid = None  # type: ignore
    join_neighborhood = None  # type: ignore

# forecast / patrol / ui (Pro)
try:
    from utils.forecast import precompute_base_intensity, aggregate_fast, prob_ge_k  # type: ignore
except Exception:  # noqa: BLE001
    precompute_base_intensity = None  # type: ignore
    aggregate_fast = None  # type: ignore
    def prob_ge_k(lam: float, k: int) -> float:
        from math import exp
        if k <= 0:
            return 1.0
        # P(N>=1) = 1 - e^{-lam}; yaklaşık kullanım
        return 1.0 - float(np.exp(-float(lam)))

try:
    from utils.patrol import allocate_patrols  # type: ignore
except Exception:  # noqa: BLE001
    allocate_patrols = None  # type: ignore

try:
    from utils.ui import (
        SMALL_UI_CSS,
        render_result_card,
        build_map_fast,
        render_kpi_row,
        render_day_hour_heatmap as _fallback_heatmap,
    )  # type: ignore
except Exception:  # noqa: BLE001
    SMALL_UI_CSS = """
        <style>
        .smallcaps {font-size: 0.9rem; opacity: .85}
        </style>
    """
    def render_result_card(*args, **kwargs):
        st.info("Açıklama kartı modülü bulunamadı.")
    def build_map_fast(*args, **kwargs):
        st.warning("Harita modülü (folium) bulunamadı.")
        return None
    def render_kpi_row(items):
        cols = st.columns(len(items))
        for col, (lbl, val, help_) in zip(cols, items):
            col.metric(lbl, val, help_)
    def _fallback_heatmap(*args, **kwargs):
        st.info("Isı matrisi modülü bulunamadı.")

# optional heatmap override
try:
    from utils.heatmap import render_day_hour_heatmap  # type: ignore
except Exception:  # noqa: BLE001
    render_day_hour_heatmap = _fallback_heatmap  # type: ignore

# reports (optional)
try:
    from components.report_view import render_reports  # type: ignore
    HAS_REPORTS = True
except Exception:  # noqa: BLE001
    HAS_REPORTS = False
    def render_reports(**kwargs):
        st.info("Raporlar modülü bulunamadı (components/report_view.py)")

# deck (optional)
try:
    from utils.deck import build_map_fast_deck  # type: ignore
except Exception:  # noqa: BLE001
    build_map_fast_deck = None  # type: ignore

# folium (optional)
try:
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:  # noqa: BLE001
    folium = None  # type: ignore
    st_folium = None  # type: ignore

# events loader (fallback for Pro)
try:
    from utils.reports import load_events  # type: ignore
except Exception:  # noqa: BLE001
    def load_events(path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        lower = {str(c).strip().lower(): c for c in df.columns}
        ts_col = next((lower[c] for c in [
            "ts","timestamp","datetime","date_time","reported_at","occurred_at","time","date"
        ] if c in lower), None)
        if not ts_col:
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"]).copy()
        if "latitude" not in df.columns and "lat" in df.columns:
            df = df.rename(columns={"lat": "latitude"})
        if "longitude" not in df.columns and "lon" in df.columns:
            df = df.rename(columns={"lon": "longitude"})
        return df

# ───────────────────────────────── yardımcılar ─────────────────────────────────

def now_sf_iso() -> str:
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")

@st.cache_data(show_spinner=False)
def load_lite_df() -> tuple[pd.DataFrame, str, Optional[dict]]:
    if callable(load_sf_crime_latest):
        df, src = load_sf_crime_latest()
        info = get_last_update(RESULTS_DIR)
        return df, src, info
    # no loader: attempt local CSV
    p = Path("results/sf_crime_latest.csv")
    if p.exists():
        df = pd.read_csv(p)
        return df, f"csv:{p}", get_last_update(p.parent)
    return pd.DataFrame(), "none", None

@st.cache_data(show_spinner=False)
def load_events_safe(path: str = "data/events.csv") -> pd.DataFrame:
    try:
        df = load_events(path)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()

def ensure_keycol(df: pd.DataFrame, want: str = KEY_COL) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if want not in out.columns:
        alts = {want.upper(), want.lower(), "GEOID", "geoid", "GeoID"}
        hit = next((c for c in out.columns if c in alts), None)
        if hit:
            out = out.rename(columns={hit: want})
    if want in out.columns:
        out[want] = out[want].astype(str)
    return out

def ensure_centroid_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out, rn = df.copy(), {}
    if "centroid_lat" not in out.columns:
        if "Centroid_Lat" in out.columns: rn["Centroid_Lat"] = "centroid_lat"
        if "CENTROID_LAT" in out.columns: rn["CENTROID_LAT"] = "centroid_lat"
        if "lat" in out.columns and "centroid_lon" in out.columns: rn["lat"] = "centroid_lat"
    if "centroid_lon" not in out.columns:
        if "Centroid_Lon" in out.columns: rn["Centroid_Lon"] = "centroid_lon"
        if "CENTROID_LON" in out.columns: rn["CENTROID_LON"] = "centroid_lon"
        if "lon" in out.columns and "centroid_lat" in out.columns: rn["lon"] = "centroid_lon"
    return out.rename(columns=rn) if rn else out

# ──────────────────────────────── Streamlit Başlık ────────────────────────────────

st.set_page_config(page_title="SF Crime Monitor / SUTAM", layout="wide")

st.title("SF Crime Monitor / SUTAM")

with st.sidebar:
    mode = st.radio("Mod", ["Lite (Monitor)", "Pro (SUTAM)"] , index=0)

# ───────────────────────────────────── Lite Panel ─────────────────────────────────────
if mode == "Lite (Monitor)":
    df, src, info = load_lite_df()

    # üst metrikler
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kaynak", src)
    c2.metric("Satır", f"{df.shape[0]:,}")
    c3.metric("Sütun", f"{df.shape[1]:,}")
    if info:
        c4.metric("Tarih Aralığı", f"{info.get('date_min') or '?'} → {info.get('date_max') or '?'}")
    else:
        c4.metric("Tarih Aralığı", "—")

    # filtreler
    with st.sidebar:
        st.header("Filtreler")
        hour = st.slider("Saat (event_hour)", 0, 23, (0, 23))
        geoid_prefix = st.text_input("GEOID prefix (opsiyonel)", "")

    mask = (df["event_hour"].between(hour[0], hour[1])) if "event_hour" in df.columns else np.ones(len(df), dtype=bool)
    if geoid_prefix and "GEOID" in df.columns:
        mask &= df["GEOID"].astype(str).str.startswith(geoid_prefix)

    dfv = df.loc[mask].copy()

    st.write("### Ön İzleme")
    st.dataframe(dfv.head(200), use_container_width=True)

    # basit harita
    latc = next((c for c in ["lat","latitude","Centroid_Lat","centroid_lat"] if c in dfv.columns), None)
    lonc = next((c for c in ["lon","longitude","Centroid_Lon","centroid_lon"] if c in dfv.columns), None)
    if latc and lonc and len(dfv) > 0:
        st.write("### Nokta Haritası (örnek)")
        sample = dfv[[latc, lonc]].dropna().rename(columns={latc: "lat", lonc: "lon"})
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        st.map(sample, size=10)
    else:
        st.info("Harita için lat/lon bulunamadı.")

    # alt bilgi
    if info:
        st.caption(f"Last update: {info.get('when','?')} • rows={info.get('rows','?')} • src={info.get('source','?')}")

# ───────────────────────────────────── Pro Panel ─────────────────────────────────────
else:
    st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)
    st.subheader("SUTAM: Suç Tahmin Modeli", anchor=False)

    LAST_UPDATE_ISO_SF = now_sf_iso()
    st.caption(f"Model: {MODEL_VERSION} • Son eğitim: {MODEL_LAST_TRAIN} • Günlük güncelleme ~19:00 (SF) • Son güncelleme: {LAST_UPDATE_ISO_SF}")

    # GEO katmanı
    if not callable(load_geoid_layer):
        st.error("utils.geo.load_geoid_layer bulunamadı. Pro paneli için utils/* modülleri gerekli.")
        st.stop()

    GEO_DF, GEO_FEATURES = load_geoid_layer("data/sf_cells.geojson")
    GEO_DF = ensure_keycol(ensure_centroid_cols(GEO_DF), KEY_COL)
    if GEO_DF.empty:
        st.error("GEOJSON yüklenemedi veya satır yok.")
        st.stop()

    if not callable(precompute_base_intensity):
        st.error("utils.forecast.precompute_base_intensity bulunamadı.")
        st.stop()

    BASE_INT = precompute_base_intensity(GEO_DF)

    # sidebar – kontroller
    with st.sidebar:
        sekme = st.radio("Sekme", ["Operasyon", "Raporlar"], index=0, horizontal=True)
        engine = st.radio("Harita motoru", ["Folium", "pydeck" if build_map_fast_deck else "Folium"], index=0, horizontal=True)
        st.markdown("**Harita katmanları**")
        show_popups = st.checkbox("Hücre popup'larını göster", value=True)
        st.markdown("**Grafik kapsamı**")
        scope = st.radio("Grafik kapsamı", ["Tüm şehir", "Seçili hücre"], index=0, label_visibility="collapsed")
        hotspot_cat = st.selectbox("Hotspot kategorisi", ["(Tüm suçlar)"] + list(CATEGORIES), index=0)
        use_hot_hours = st.checkbox("Geçici hotspot için saat filtresi", value=False)
        if use_hot_hours:
            hot_hours_rng = st.slider("Saat aralığı (hotspot)", 0, 24, (0, 24))
        else:
            hot_hours_rng = (0, 24)
        ufuk = st.radio("Zaman Ufku", ["24s", "48s", "7g"], index=0, horizontal=True)
        max_h, step = (24, 1) if ufuk == "24s" else (48, 3) if ufuk == "48s" else (7*24, 24)
        start_h, end_h = st.slider("Saat filtresi", 0, max_h, (0, max_h), step=step)
        sel_categories = st.multiselect("Kategori", ["(Hepsi)"] + list(CATEGORIES), default=[])
        filters = {"cats": CATEGORIES if sel_categories and "(Hepsi)" in sel_categories else (sel_categories or None)}
        K_planned = st.number_input("Planlanan devriye sayısı (K)", 1, 50, 6, 1)
        duty_minutes = st.number_input("Devriye görev süresi (dk)", 15, 600, 120, 15)
        cell_minutes = st.number_input("Hücre başına ort. kontrol (dk)", 2, 30, 6, 1)
        colA, colB = st.columns(2)
        btn_predict = colA.button("Tahmin et")
        btn_patrol  = colB.button("Devriye öner")

    # state
    st.session_state.setdefault("agg", None)
    st.session_state.setdefault("agg_long", None)
    st.session_state.setdefault("patrol", None)
    st.session_state.setdefault("start_iso", None)
    st.session_state.setdefault("horizon_h", None)
    st.session_state.setdefault("events", None)
    st.session_state.setdefault("explain", {})

    # yardımcı: yakın geçmiş olayları
    def recent_events(df: pd.DataFrame, lookback_h: int, category: Optional[str]) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["ts", "latitude", "longitude", KEY_COL])
        out = df.copy()
        if "ts" in out.columns:
            out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
            out = out[out["ts"] >= (pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_h))]
        if category and category != "(Tüm suçlar)" and "type" in out.columns:
            out = out[out["type"] == category]
        if "latitude" not in out.columns and "lat" in out.columns:
            out = out.rename(columns={"lat": "latitude"})
        if "longitude" not in out.columns and "lon" in out.columns:
            out = out.rename(columns={"lon": "longitude"})
        out = out.dropna(subset=["latitude", "longitude"]).copy()
        out["weight"] = 1.0
        return out

    def make_temp_hotspot_from_agg(agg: pd.DataFrame, geo_df: pd.DataFrame, topn: int = 80) -> pd.DataFrame:
        if agg is None or agg.empty:
            return pd.DataFrame(columns=["latitude", "longitude", "weight"])
        agg2 = ensure_keycol(agg, KEY_COL)
        geo2 = ensure_keycol(ensure_centroid_cols(geo_df), KEY_COL)
        tmp = (
            agg2.nlargest(topn, "expected")
            .merge(geo2[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL, how="left")
            .dropna(subset=["centroid_lat", "centroid_lon"])
        )
        pts = tmp.rename(columns={"centroid_lat": "latitude", "centroid_lon": "longitude"})[["latitude", "longitude"]]
        pts["weight"] = tmp["expected"].clip(lower=0).astype(float)
        return pts

    def run_prediction(
        start_h: int,
        end_h: int,
        filters: dict,
        geo_df: pd.DataFrame,
        base_int: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str, int]:
        if not callable(aggregate_fast):
            return pd.DataFrame(), None, datetime.utcnow().isoformat(), 0
        start_dt = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET + start_h)).replace(minute=0, second=0, microsecond=0)
        horizon_h = max(1, end_h - start_h)
        start_iso = start_dt.isoformat()
        events_df = load_events_safe()
        st.session_state["events"] = events_df
        agg = aggregate_fast(
            start_iso,
            horizon_h,
            geo_df,
            base_int,
            events=events_df,
            near_repeat_alpha=0.35,
            nr_lookback_h=24,
            nr_radius_m=400,
            nr_decay_h=12.0,
            filters=filters,
        )
        # tier ata
        def assign_tier_safe(agg_in: pd.DataFrame) -> pd.DataFrame:
            if agg_in is None or agg_in.empty or "expected" not in agg_in.columns:
                return agg_in
            out = agg_in.copy()
            x = pd.to_numeric(out["expected"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
            out["expected"] = x
            labels5 = ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek"]
            if x.nunique(dropna=True) < 5 or x.count() < 5:
                out["tier"] = "Çok Düşük"; return out
            try:
                out["tier"] = pd.qcut(x, q=5, labels=labels5, duplicates="drop").astype(str)
                if out["tier"].isna().all():
                    raise ValueError
                return out
            except Exception:
                q = np.quantile(x.to_numpy(), [0.20, 0.40, 0.60, 0.80]).astype(float)
                eps = max(1e-9, 1e-6 * float(np.nanmax(x) - np.nanmin(x)))
                for i in range(1, len(q)):
                    if q[i] <= q[i - 1]:
                        q[i] = q[i - 1] + eps
                bins = np.concatenate(([-np.inf], q, [np.inf]))
                out["tier"] = pd.cut(x, bins=bins, labels=labels5, include_lowest=True).astype(str)
                return out
        agg = assign_tier_safe(agg)
        agg = ensure_keycol(agg, KEY_COL)
        # uzun dönem referans (opsiyonel)
        try:
            long_start_iso = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET - 30 * 24)).replace(minute=0, second=0, microsecond=0).isoformat()
            agg_long = aggregate_fast(long_start_iso, 30 * 24, geo_df, base_int, events=events_df, near_repeat_alpha=0.0, filters=None)
            agg_long = ensure_keycol(agg_long, KEY_COL)
        except Exception:
            agg_long = None
        return agg, agg_long, start_iso, horizon_h

    # layout
    col1, col2 = st.columns([2.4, 1.0])

    with col1:
        if st.session_state["agg"] is None or btn_predict:
            agg, agg_long, start_iso, horizon_h = run_prediction(start_h, end_h, filters, GEO_DF, BASE_INT)
            st.session_state.update({"agg": agg, "agg_long": agg_long, "patrol": None,
                                     "start_iso": start_iso, "horizon_h": horizon_h})
        agg = st.session_state["agg"]
        events_all = st.session_state.get("events") or pd.DataFrame()
        lookback_h = int(np.clip(2 * (st.session_state.get("horizon_h") or 24), 24, 72))
        ev_recent_df = recent_events(events_all, lookback_h, hotspot_cat)
        temp_points = (
            ev_recent_df[["latitude", "longitude", "weight"]]
            if not ev_recent_df.empty else make_temp_hotspot_from_agg(agg, GEO_DF, topn=80)
        )
        st.sidebar.caption(f"Geçici hotspot noktası: {0 if temp_points is None else len(temp_points)}")

        # neighborhood join
        if isinstance(agg, pd.DataFrame) and not agg.empty and (join_neighborhood is not None) and ("neighborhood" not in agg.columns) and ("neighborhood" in GEO_DF.columns):
            try:
                agg = join_neighborhood(agg, GEO_DF)
            except Exception:
                pass

        # harita
        if isinstance(agg, pd.DataFrame) and not agg.empty:
            if engine == "Folium" and folium is not None and st_folium is not None:
                m = build_map_fast(
                    df_agg=agg, geo_features=GEO_FEATURES, geo_df=GEO_DF,
                    show_popups=show_popups, patrol=st.session_state.get("patrol"),
                    show_hotspot=True, perm_hotspot_mode="heat",
                    show_temp_hotspot=True, temp_hotspot_points=temp_points,
                    add_layer_control=False, risk_layer_show=True,
                    perm_hotspot_show=True, temp_hotspot_show=True,
                    risk_layer_name="Tahmin (risk)", perm_hotspot_layer_name="Hotspot (kalıcı)",
                    temp_hotspot_layer_name="Hotspot (geçici)",
                )
                # Layer control yeniden ekle
                for k, ch in list(m._children.items()):
                    if isinstance(ch, folium.map.LayerControl):
                        del m._children[k]
                folium.TileLayer(
                    tiles="CartoDB positron", name="cartodbpositron", control=True,
                    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
                         '&copy; <a href="https://carto.com/attributions">CARTO</a>',
                ).add_to(m)
                ret = st_folium(m, key="riskmap", height=540, width=860,
                                returned_objects=["last_object_clicked", "last_clicked"])
                if ret and resolve_clicked_gid is not None:
                    gid, _ = resolve_clicked_gid(GEO_DF, ret)
                    if gid:
                        st.session_state["explain"] = {"geoid": gid}
            else:
                if build_map_fast_deck is None:
                    st.error("Pydeck harita modülü yok (utils/deck.py). Lütfen Folium'u kullanın.")
                else:
                    deck = build_map_fast_deck(
                        df_agg=agg, geo_df=GEO_DF, show_hotspot=True, show_temp_hotspot=True,
                        temp_hotspot_points=temp_points, show_risk_layer=True,
                        map_style="mapbox://styles/mapbox/light-v11",
                        initial_view={"lat": 37.7749, "lon": -122.4194, "zoom": 11.8},
                    )
                    st.pydeck_chart(deck)
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

        # result card
        start_iso = st.session_state.get("start_iso")
        horizon_h = st.session_state.get("horizon_h")
        info_sel = st.session_state.get("explain")
        if info_sel and info_sel.get("geoid"):
            render_result_card(agg, info_sel["geoid"], start_iso, horizon_h)
        else:
            st.info("Haritada bir hücreye tıklayın; kart burada görünecek.")

    with col2:
        st.subheader("Risk Özeti", anchor=False)
        a = st.session_state.get("agg")
        if isinstance(a, pd.DataFrame) and not a.empty:
            kpi_expected = round(float(a["expected"].sum()), 2)
            cnts = {
                "Çok Yüksek": int((a.get("tier", pd.Series(dtype=str)) == "Çok Yüksek").sum()),
                "Yüksek":     int((a.get("tier", pd.Series(dtype=str)) == "Yüksek").sum()),
                "Orta":       int((a.get("tier", pd.Series(dtype=str)) == "Orta").sum()),
                "Düşük":      int((a.get("tier", pd.Series(dtype=str)) == "Düşük").sum()),
                "Çok Düşük":  int((a.get("tier", pd.Series(dtype=str)) == "Çok Düşük").sum()),
            }
            render_kpi_row([
                ("Beklenen olay (ufuk)", kpi_expected, "Seçili ufukta toplam beklenen olay"),
                ("Çok Yüksek", cnts["Çok Yüksek"], "Üst %20 hücre sayısı"),
                ("Yüksek",     cnts["Yüksek"],     "Yüksek kademe hücre sayısı"),
                ("Orta",       cnts["Orta"],       "Orta kademe hücre sayısı"),
                ("Düşük",      cnts["Düşük"],      "Düşük kademe hücre sayısı"),
                ("Çok Düşük",  cnts["Çok Düşük"],  "Alt %20 hücre sayısı"),
            ])
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

        st.subheader("Top-5 kritik GEOID")
        def top_risky_table(df_agg: pd.DataFrame, n: int, show_ci: bool, start_iso: Optional[str], horizon_h: int) -> pd.DataFrame:
            def poisson_ci(lam: float, z: float = 1.96) -> tuple[float, float]:
                s = float(np.sqrt(max(lam, 1e-9))); return max(0.0, lam - z * s), lam + z * s
            cols = [KEY_COL, "expected"] + (["nr_boost"] if "nr_boost" in df_agg.columns else [])
            df = ensure_keycol(df_agg, KEY_COL)[cols].sort_values("expected", ascending=False).head(n).reset_index(drop=True)
            lam = df["expected"].to_numpy()
            # P(N>=1)
            df["P(≥1)%"] = [round((1.0 - float(np.exp(-float(l)))) * 100, 1) for l in lam]
            try:
                if start_iso:
                    _start = pd.to_datetime(start_iso); _end = _start + pd.to_timedelta(horizon_h, unit="h")
                    df["Saat"] = f"{_start.strftime('%H:%M')}–{_end.strftime('%H:%M')} (SF)"
                else:
                    df["Saat"] = "-"
            except Exception:
                df["Saat"] = "-"
            if show_ci:
                ci = [poisson_ci(float(l)) for l in lam]
                df["95% GA"] = [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in ci]
            if "nr_boost" in df.columns:
                df["NR"] = df["nr_boost"].round(2)
            df["E[olay] (λ)"] = df["expected"].round(2)
            drop = ["expected"] + (["nr_boost"] if "nr_boost" in df.columns else [])
            return df.drop(columns=drop)

        if isinstance(a, pd.DataFrame) and not a.empty:
            tab = top_risky_table(a, n=5, show_ci=True, start_iso=st.session_state.get("start_iso"), horizon_h=int(st.session_state.get("horizon_h") or 0))
            st.dataframe(tab, use_container_width=True)
            st.markdown("Seç / odağı haritada göster:")
            cols = st.columns(len(tab))
            for i, row in enumerate(tab.itertuples()):
                with cols[i]:
                    if st.button(str(row.geoid)):
                        st.session_state["explain"] = {"geoid": str(row.geoid)}
                        st.rerun()
            st.caption("Butona tıklayınca haritada centroid işaretlenir ve açıklama kartı güncellenir.")

        st.subheader("Gün × Saat Isı Matrisi")
        if st.session_state.get("agg") is not None and st.session_state.get("start_iso"):
            H = int(st.session_state.get("horizon_h") or 24)
            render_day_hour_heatmap(
                agg=st.session_state["agg"],
                start_iso=st.session_state["start_iso"],
                horizon_h=H,
                geo_df=GEO_DF,
                base_int=BASE_INT,
                filters=filters,
                events_df=st.session_state.get("events"),
            )
        else:
            st.caption("Isı matrisi, bir tahmin üretildiğinde gösterilir.")

        # devriye özeti (opsiyonel)
        if btn_patrol and allocate_patrols is not None and isinstance(a, pd.DataFrame) and not a.empty:
            try:
                st.session_state["patrol"] = allocate_patrols(a, k=K_planned, duty_minutes=duty_minutes, cell_minutes=cell_minutes)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Devriye hesabı başarısız: {e}")

    # Raporlar
    if sekme == "Raporlar":
        agg_current = st.session_state.get("agg")
        agg_long = st.session_state.get("agg_long")
        events_src = st.session_state.get("events")
        render_reports(events_df=events_src, agg_current=agg_current, agg_long_term=agg_long)
