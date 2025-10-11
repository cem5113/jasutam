from __future__ import annotations

import os
import sys
import time
import folium
from datetime import datetime, timedelta
from utils.constants import SF_TZ_OFFSET  
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from services.metrics import get_latest_metrics

# Yerel paket yolları
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Yerel modüller
from utils.geo import load_geoid_layer, resolve_clicked_gid
from utils.forecast import precompute_base_intensity, aggregate_fast, prob_ge_k
from utils.patrol import allocate_patrols
from utils.ui import (
    SMALL_UI_CSS,
    render_result_card,
    build_map_fast,
    render_kpi_row,
    render_day_hour_heatmap as fallback_heatmap,
)
from utils.constants import SF_TZ_OFFSET, KEY_COL, MODEL_VERSION, MODEL_LAST_TRAIN, CATEGORIES
from components.last_update import show_last_update_badge
from services.metrics import get_latest_metrics, METRICS_FILE

# Opsiyonel modüller
try:
    from components.report_view import render_reports  # type: ignore
    HAS_REPORTS = True
except ModuleNotFoundError:
    HAS_REPORTS = False

    def render_reports(**kwargs):
        st.info("Raporlar modülü bulunamadı (components/report_view.py).")

try:
    from utils.heatmap import render_day_hour_heatmap  # type: ignore
except ImportError:
    # Geriye dönük uyumluluk: utils.ui içindeki fonksiyonu kullan
    render_day_hour_heatmap = fallback_heatmap  # type: ignore

try:
    from utils.deck import build_map_fast_deck  # type: ignore
except ImportError:
    build_map_fast_deck = None

try:
    from utils.reports import load_events  # type: ignore
except Exception:

    def load_events(path: str) -> pd.DataFrame:
        """Basit CSV okuyucu (ts/timestamp kolonu varsa UTC'ye çevirir)."""
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        lower = {str(c).strip().lower(): c for c in df.columns}
        ts_col = next(
            (
                lower[c]
                for c in [
                    "ts",
                    "timestamp",
                    "datetime",
                    "date_time",
                    "reported_at",
                    "occurred_at",
                    "time",
                    "date",
                ]
                if c in lower
            ),
            None,
        )
        if not ts_col:
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])  # yalnız geçerli zamanlar
        if "latitude" not in df.columns and "lat" in df.columns:
            df = df.rename(columns={"lat": "latitude"})
        if "longitude" not in df.columns and "lon" in df.columns:
            df = df.rename(columns={"lon": "longitude"})
        return df

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def ensure_keycol(df: pd.DataFrame, want: str = KEY_COL) -> pd.DataFrame:
    """DataFrame'de KEY_COL adını garanti eder ve string'e çevirir."""
    if df is None or df.empty:
        return df
    if want in df.columns:
        out = df.copy()
        out[want] = out[want].astype(str)
        return out
    alts = {want.upper(), want.lower(), "GEOID", "geoid", "GeoID"}
    hit = next((c for c in df.columns if c in alts), None)
    out = df.copy()
    if hit and hit != want:
        out = out.rename(columns={hit: want})
    if want in out.columns:
        out[want] = out[want].astype(str)
    return out


def ensure_centroid_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Centroid kolon adlarını standardize eder (centroid_lat/lon)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    rn: dict[str, str] = {}
    if "centroid_lat" not in out.columns:
        if "Centroid_Lat" in out.columns:
            rn["Centroid_Lat"] = "centroid_lat"
        if "CENTROID_LAT" in out.columns:
            rn["CENTROID_LAT"] = "centroid_lat"
        if "lat" in out.columns and "centroid_lon" in out.columns:
            rn["lat"] = "centroid_lat"
    if "centroid_lon" not in out.columns:
        if "Centroid_Lon" in out.columns:
            rn["Centroid_Lon"] = "centroid_lon"
        if "CENTROID_LON" in out.columns:
            rn["CENTROID_LON"] = "centroid_lon"
        if "lon" in out.columns and "centroid_lat" in out.columns:
            rn["lon"] = "centroid_lon"
    return out.rename(columns=rn) if rn else out


def now_sf_iso() -> str:
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")


def load_events_safe(path: str = "data/events.csv") -> pd.DataFrame:
    try:
        df = load_events(path)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


def recent_events(df: pd.DataFrame, lookback_h: int, category: Optional[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts", "latitude", "longitude", KEY_COL])
    out = df.copy()
    ts_col = "ts" if "ts" in out.columns else None
    if ts_col:
        out["ts"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
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
    try:
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
    except Exception:
        return pd.DataFrame(columns=["latitude", "longitude", "weight"])


def run_prediction(
    start_h: int,
    end_h: int,
    filters: dict,
    geo_df: pd.DataFrame,
    base_int: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str, int]:
    start_dt = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET + start_h)).replace(
        minute=0, second=0, microsecond=0
    )
    horizon_h = max(1, end_h - start_h)
    start_iso = start_dt.isoformat()

    events_df = load_events_safe()

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

    # Sağlam kademe sınıflayıcı: expected → ['Çok Düşük','Düşük','Orta','Yüksek','Çok Yüksek']
    def assign_tier_safe(agg_in: pd.DataFrame) -> pd.DataFrame:
        if agg_in is None or agg_in.empty or "expected" not in agg_in.columns:
            return agg_in
        out = agg_in.copy()
        x = (
            pd.to_numeric(out["expected"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower=0.0)
        )
        out["expected"] = x
        labels5 = ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek"]

        # Veri çeşitliliği azsa tek seviyeye düş
        if x.nunique(dropna=True) < 5 or x.count() < 5:
            out["tier"] = "Çok Düşük"
            return out

        # qcut ile dene
        try:
            out["tier"] = pd.qcut(x, q=5, labels=labels5, duplicates="drop").astype(str)
            if out["tier"].isna().all():
                raise ValueError("qcut collapsed")
            return out
        except Exception:
            pass

        # Elle kantil: epsilon ile kenarları ayır
        try:
            q = np.quantile(x.to_numpy(), [0.20, 0.40, 0.60, 0.80]).astype(float)
            eps = max(1e-9, 1e-6 * float(np.nanmax(x) - np.nanmin(x)))
            for i in range(1, len(q)):
                if q[i] <= q[i - 1]:
                    q[i] = q[i - 1] + eps
            bins = np.concatenate(([-np.inf], q, [np.inf]))
            out["tier"] = pd.cut(x, bins=bins, labels=labels5, include_lowest=True).astype(str)
            return out
        except Exception:
            med = float(np.nanmedian(x))
            p75 = float(np.nanquantile(x, 0.75))
            p90 = float(np.nanquantile(x, 0.90))

            def fallback(v: float) -> str:
                if v <= med * 0.5:
                    return "Çok Düşük"
                if v <= med:
                    return "Düşük"
                if v <= p75:
                    return "Orta"
                if v <= p90:
                    return "Yüksek"
                return "Çok Yüksek"

            out["tier"] = [fallback(float(v)) for v in x]
            return out

    # Eski pd.cut temelli kademe bloğu kaldırıldı; güvenli sınıflayıcı uygulanıyor
    agg = assign_tier_safe(agg)
    agg = ensure_keycol(agg, KEY_COL)

    # Uzun ufuk referansı (30 gün geriden bugüne)
    try:
        long_start_iso = (
            datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET - 30 * 24)
        ).replace(minute=0, second=0, microsecond=0).isoformat()
        agg_long = aggregate_fast(
            long_start_iso,
            30 * 24,
            geo_df,
            base_int,
            events=events_df,
            near_repeat_alpha=0.0,
            filters=None,
        )
        agg_long = ensure_keycol(agg_long, KEY_COL)
    except Exception:
        agg_long = None

    return agg, agg_long, start_iso, horizon_h


def top_risky_table(
    df_agg: pd.DataFrame, n: int, show_ci: bool, start_iso: Optional[str], horizon_h: int
) -> pd.DataFrame:
    def poisson_ci(lam: float, z: float = 1.96) -> tuple[float, float]:
        s = float(np.sqrt(max(lam, 1e-9)))
        return max(0.0, lam - z * s), lam + z * s

    cols = [KEY_COL, "expected"] + (["nr_boost"] if "nr_boost" in df_agg.columns else [])
    df = ensure_keycol(df_agg, KEY_COL)[cols].sort_values("expected", ascending=False).head(n).reset_index(drop=True)

    lam = df["expected"].to_numpy()
    df["P(≥1)%"] = [round(prob_ge_k(l, 1) * 100, 1) for l in lam]

    # Saat aralığı (SF)
    try:
        if start_iso:
            _start = pd.to_datetime(start_iso)
            _end = _start + pd.to_timedelta(horizon_h, unit="h")
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


# ─────────────────────────────────────────────────────────────────────────────
# UI: Sayfa & Başlık
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli", layout="wide")
st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)
st.title("SUTAM: Suç Tahmin Modeli")

metrics = get_latest_metrics()
if metrics:
    col1, col2, col3 = st.columns(3)
    if metrics.get("auc") is not None:
        col1.metric("AUC (7g)", f"{metrics['auc']:.3f}")
    if metrics.get("hit_rate_topk") is not None:
        col2.metric("HitRate@TopK", f"{metrics['hit_rate_topk']*100:.1f}%")
    if metrics.get("brier") is not None:
        col3.metric("Brier Score", f"{metrics['brier']:.3f}")
else:
    # İstersen bu teşhis satırını kaldırabilirsin
    st.caption(f"📊 KPI için ölçüm dosyası bulunamadı: {METRICS_FILE}")

# Veri sonu
try:
    _events_df = load_events_safe()
    st.session_state["events_df"] = _events_df if isinstance(_events_df, pd.DataFrame) else None
    st.session_state["events"] = st.session_state["events_df"]
    data_upto_val = (
        pd.to_datetime(_events_df["ts"]).max().date().isoformat()
        if isinstance(_events_df, pd.DataFrame)
        and not _events_df.empty
        and "ts" in _events_df.columns
        else None
    )
except Exception:
    st.session_state["events_df"] = None
    st.session_state["events"] = None
    data_upto_val = None

show_last_update_badge(
    data_upto=data_upto_val,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

# Geo katmanı
GEO_DF, GEO_FEATURES = load_geoid_layer("data/sf_cells.geojson")
GEO_DF = ensure_keycol(ensure_centroid_cols(GEO_DF), KEY_COL)
if GEO_DF.empty:
    st.error("GEOJSON yüklenemedi veya satır yok.")
    st.stop()

# Model tabanı
BASE_INT = precompute_base_intensity(GEO_DF)

# Sidebar
st.sidebar.markdown("### Görünüm")
sekme_options = ["Operasyon"] + (["Raporlar"] if HAS_REPORTS else [])
sekme = st.sidebar.radio("", options=sekme_options, index=0, horizontal=True)

st.sidebar.divider()
st.sidebar.header("Devriye Parametreleri")
engine = st.sidebar.radio("Harita motoru", ["Folium", "pydeck"], index=0, horizontal=True)

st.sidebar.subheader("Harita katmanları")
show_poi = st.sidebar.checkbox("POI overlay", value=False)
show_transit = st.sidebar.checkbox("Toplu taşıma overlay", value=False)
show_popups = st.sidebar.checkbox("Hücre popup'larını (en olası 3 suç) göster", value=True)

scope = st.sidebar.radio("Grafik kapsamı", ["Tüm şehir", "Seçili hücre"], index=0)

# Hotspot ayarları
show_hotspot = True
show_temp_hotspot = True
hotspot_cat = st.sidebar.selectbox(
    "Hotspot kategorisi",
    ["(Tüm suçlar)"] + CATEGORIES,
    index=0,
    help="Kalıcı/Geçici hotspot katmanları bu kategoriye göre gösterilir.",
)
use_hot_hours = st.sidebar.checkbox("Geçici hotspot için gün içi saat filtresi", value=False)
hot_hours_rng = st.sidebar.slider("Saat aralığı (hotspot)", 0, 24, (0, 24), disabled=not use_hot_hours)

# Zaman ufku (gerçek zamanlı gösterim)
current_time = datetime.now().strftime('%H:%M')
current_date = datetime.now().strftime('%Y-%m-%d')
ufuk_label = f"Zaman Aralığı (from {current_time}, today, {current_date})"

ufuk = st.sidebar.radio(ufuk_label, ["24s", "48s", "7g"], index=0, horizontal=True)
max_h, step = (24, 1) if ufuk == "24s" else (48, 3) if ufuk == "48s" else (7*24, 24)
start_h, end_h = st.sidebar.slider("Saat filtresi", min_value=0, max_value=max_h, value=(0, max_h), step=step)

# Kategori filtresi (tahmin motoru)
sel_categories = st.sidebar.multiselect("Kategori", ["(Hepsi)"] + CATEGORIES, default=[])
filters = {"cats": CATEGORIES if sel_categories and "(Hepsi)" in sel_categories else (sel_categories or None)}

show_advanced = st.sidebar.checkbox("Gelişmiş metrikleri göster (analist)", value=False)

st.sidebar.divider()
st.sidebar.subheader("Devriye Parametreleri")
K_planned = st.sidebar.number_input("Planlanan devriye sayısı (K)", 1, 50, 6, 1)
duty_minutes = st.sidebar.number_input("Devriye görev süresi (dk)", 15, 600, 120, 15)
cell_minutes = st.sidebar.number_input("Hücre başına ort. kontrol (dk)", 2, 30, 6, 1)

colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et")
btn_patrol = colB.button("Devriye öner")

# State
st.session_state.setdefault("agg", None)
st.session_state.setdefault("agg_long", None)
st.session_state.setdefault("patrol", None)
st.session_state.setdefault("start_iso", None)
st.session_state.setdefault("horizon_h", None)
st.session_state.setdefault("explain", {})

# ─────────────────────────────────────────────────────────────────────────────
# SEKME: Operasyon
# ─────────────────────────────────────────────────────────────────────────────
if sekme == "Operasyon":
    col1, col2 = st.columns([2.4, 1.0])

    with col1:
        st.caption(f"Son güncelleme (SF): {now_sf_iso()}")

        if btn_predict or st.session_state["agg"] is None:
            agg, agg_long, start_iso, horizon_h = run_prediction(start_h, end_h, filters, GEO_DF, BASE_INT)
            st.session_state.update(
                {
                    "agg": agg,
                    "agg_long": agg_long,
                    "patrol": None,
                    "start_iso": start_iso,
                    "horizon_h": horizon_h,
                    "events": st.session_state.get("events_df"),
                }
            )

        agg = st.session_state["agg"]
        events_all = st.session_state.get("events")
        lookback_h = int(np.clip(2 * (st.session_state.get("horizon_h") or 24), 24, 72))
        ev_recent_df = recent_events(
            events_all if isinstance(events_all, pd.DataFrame) else pd.DataFrame(),
            lookback_h,
            hotspot_cat,
        )

        # Grafik kapsamı: seçili hücre
        if scope == "Seçili hücre" and st.session_state.get("explain", {}).get("geoid") and KEY_COL in ev_recent_df.columns:
            gid = str(st.session_state["explain"]["geoid"])  # str karşılaştırma
            ev_recent_df = ev_recent_df[ev_recent_df[KEY_COL].astype(str) == gid]

        # Geçici hotspot noktaları
        temp_points = (
            ev_recent_df[["latitude", "longitude", "weight"]]
            if not ev_recent_df.empty
            else pd.DataFrame(columns=["latitude", "longitude", "weight"])
        )
        if use_hot_hours and not temp_points.empty and "ts" in ev_recent_df.columns:
            h1, h2 = hot_hours_rng[0], (hot_hours_rng[1] - 1) % 24
            temp_points = ev_recent_df[ev_recent_df["ts"].dt.hour.between(h1, h2)][["latitude", "longitude", "weight"]]

        if temp_points.empty and isinstance(agg, pd.DataFrame) and not agg.empty:
            temp_points = make_temp_hotspot_from_agg(agg, GEO_DF, topn=80)

        st.sidebar.caption(f"Geçici hotspot noktası: {len(temp_points)}")

        if isinstance(agg, pd.DataFrame):
            if "neighborhood" not in agg.columns and "neighborhood" in GEO_DF.columns:
                try:
                    from utils.geo import join_neighborhood  # 2. adımda eklediğimiz yardımcı
                    agg = join_neighborhood(agg, GEO_DF)
                except Exception:
                    # utils.geo.join_neighborhood yoksa sessizce geç
                    pass

        # — HARİTA — (gömülü katman simgesi = sağ üst ikon)
        if agg is not None:
            if engine == "Folium":
                try:
                    # build_map_fast varsa LayerControl'u biz ekleyeceğiz
                    m = build_map_fast(
                        df_agg=agg,
                        geo_features=GEO_FEATURES,
                        geo_df=GEO_DF,
                        show_popups=show_popups,
                        patrol=st.session_state.get("patrol"),
                        # hotspot katmanlarını üret
                        show_hotspot=True,
                        perm_hotspot_mode="heat",
                        show_temp_hotspot=True,
                        temp_hotspot_points=temp_points,
                        # kendi kontrolümüzü ekleyeceğiz
                        add_layer_control=False,
                        risk_layer_show=True,
                        perm_hotspot_show=True,
                        temp_hotspot_show=True,
                        risk_layer_name="Tahmin (risk)",
                        perm_hotspot_layer_name="Hotspot (kalıcı)",
                        temp_hotspot_layer_name="Hotspot (geçici)",
                    )
                except TypeError:
                    # Eski imza: add_layer_control yok
                    m = build_map_fast(
                        df_agg=agg,
                        geo_features=GEO_FEATURES,
                        geo_df=GEO_DF,
                        show_popups=show_popups,
                        patrol=st.session_state.get("patrol"),
                        show_hotspot=True,
                        perm_hotspot_mode="heat",
                        show_temp_hotspot=True,
                        temp_hotspot_points=temp_points,
                    )

                # build_map_fast LayerControl eklediyse kaldır
                for k, ch in list(m._children.items()):
                    if isinstance(ch, folium.map.LayerControl):
                        del m._children[k]

                # — Taban katman + açık atıf (OSM & CARTO) —
                folium.TileLayer(
                    tiles="CartoDB positron",
                    name="cartodbpositron",
                    control=True,
                    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
                    'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                ).add_to(m)

                # — Katman menüsü: tek ikon, kapalı (collapsed) —
                folium.LayerControl(position="topright", collapsed=True, autoZIndex=True).add_to(m)

                ret = st_folium(
                    m,
                    key="riskmap",
                    height=540,
                    width=800,
                    returned_objects=["last_object_clicked", "last_clicked"],
                )
                if ret:
                    gid, _ = resolve_clicked_gid(GEO_DF, ret)
                    if gid:
                        st.session_state["explain"] = {"geoid": gid}
            else:
                # pydeck
                if build_map_fast_deck is None:
                    st.error("Pydeck harita modülü bulunamadı (utils/deck.py). Lütfen Folium motorunu seçin.")
                else:
                    deck = build_map_fast_deck(
                        df_agg=agg,
                        geo_df=GEO_DF,
                        show_hotspot=True,  # kalıcı hotspot (üst %10)
                        show_temp_hotspot=show_temp_hotspot,  # geçici hotspot
                        temp_hotspot_points=temp_points,
                        show_risk_layer=True,  # risk katmanı (tier paletine göre)
                        map_style=(
                            "mapbox://styles/mapbox/dark-v11"
                            if st.session_state.get("dark_mode")
                            else "mapbox://styles/mapbox/light-v11"
                        ),
                        initial_view={"lat": 37.7749, "lon": -122.4194, "zoom": 11.8},
                    )
                    st.pydeck_chart(deck)
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

        # Açıklama kartı
        start_iso = st.session_state.get("start_iso")
        horizon_h = st.session_state.get("horizon_h")
        info = st.session_state.get("explain")
        if info and info.get("geoid"):
            render_result_card(agg, info["geoid"], start_iso, horizon_h)
        else:
            st.info("Haritada bir hücreye tıklayın; kart burada görünecek.")

    # Sağ panel – özetler
    with col2:
        st.subheader("Risk Özeti", anchor=False)
        a = st.session_state.get("agg")
        if isinstance(a, pd.DataFrame) and not a.empty:
            kpi_expected = round(float(a["expected"].sum()), 2)
            cnts = {
                "Çok Yüksek": int((a.get("tier", pd.Series(dtype=str)) == "Çok Yüksek").sum()),
                "Yüksek": int((a.get("tier", pd.Series(dtype=str)) == "Yüksek").sum()),
                "Orta": int((a.get("tier", pd.Series(dtype=str)) == "Orta").sum()),
                "Düşük": int((a.get("tier", pd.Series(dtype=str)) == "Düşük").sum()),
                "Çok Düşük": int((a.get("tier", pd.Series(dtype=str)) == "Çok Düşük").sum()),
            }
            render_kpi_row(
                [
                    ("Beklenen olay (ufuk)", kpi_expected, "Seçili zaman ufkunda toplam beklenen olay sayısı"),
                    ("Çok Yüksek", cnts["Çok Yüksek"], "En yüksek riskli hücre sayısı (üst %20)"),
                    ("Yüksek", cnts["Yüksek"], "Yüksek kademe riskli hücre sayısı"),
                    ("Orta", cnts["Orta"], "Orta kademe riskli hücre sayısı"),
                    ("Düşük", cnts["Düşük"], "Düşük kademe riskli hücre sayısı"),
                    ("Çok Düşük", cnts["Çok Düşük"], "En düşük riskli hücre sayısı (alt %20)"),
                ]
            )
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

        st.subheader("Top-5 kritik GEOID")
        if isinstance(a, pd.DataFrame) and not a.empty:
            tab = top_risky_table(
                a,
                n=5,
                show_ci=show_advanced,
                start_iso=st.session_state.get("start_iso"),
                horizon_h=int(st.session_state.get("horizon_h") or 0),
            )
            st.dataframe(tab, use_container_width=True, height=300)
            st.caption(
                "P(≥1)%: Seçilen ufukta en az bir olay olma olasılığı."
                if not show_advanced
                else "95% GA: λ ± 1.96·√λ (alt sınır 0'a kırpılır)."
            )

        st.subheader("Devriye özeti")
        if isinstance(a, pd.DataFrame) and not a.empty and btn_patrol:
            st.session_state["patrol"] = allocate_patrols(
                a,
                GEO_DF,
                k_planned=int(K_planned),
                duty_minutes=int(duty_minutes),
                cell_minutes=int(cell_minutes),
                travel_overhead=0.40,
            )

        patrol = st.session_state.get("patrol")
        if patrol and patrol.get("zones"):
            rows = [
                {
                    "zone": z["id"],
                    "cells_planned": z["planned_cells"],
                    "capacity_cells": z["capacity_cells"],
                    "eta_minutes": z["eta_minutes"],
                    "utilization_%": z["utilization_pct"],
                    "avg_risk(E[olay])": round(z["expected_risk"], 2),
                }
                for z in patrol["zones"]
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

        st.subheader("Gün × Saat Isı Matrisi")
        if st.session_state.get("agg") is not None and st.session_state.get("start_iso"):
            render_day_hour_heatmap(
                st.session_state["agg"],
                st.session_state.get("start_iso"),
                st.session_state.get("horizon_h"),
            )
        else:
            st.caption("Isı matrisi, bir tahmin üretildiğinde gösterilir.")

        from services.metrics import get_latest_metrics
        sf_now = datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)
        label = f"Güncel Model Metrikleri ({sf_now.strftime('%Y-%m-%d')}, {sf_now.strftime('%H:%M')} SF time)"
        st.subheader(label, anchor=False)
        
        m = get_latest_metrics()  # services/metrics.py: metrics_all.csv → latest_metrics.json → read
        if m:
            k1, k2, k3 = st.columns(3)
            if m.get("auc") is not None:
                k1.metric("AUC (ROC)", f"{m['auc']:.3f}")
            if m.get("hit_rate_topk") is not None:
                k2.metric("HitRate@TopK", f"{m['hit_rate_topk']*100:.1f}%")
            if m.get("brier") is not None:
                k3.metric("Brier Score", f"{m['brier']:.3f}")
        
            # Kaynak ve sürüm bilgisi (artefact -> json akışını şeffaf göster)
            try:
                rel_path = os.path.relpath(METRICS_FILE, PROJECT_ROOT)
            except Exception:
                rel_path = METRICS_FILE
            ts_text = m.get("timestamp", "—")
            st.caption(f"Kaynak: {rel_path} • Zaman damgası: {ts_text} • Model sürümü: {MODEL_VERSION}")
        else:
            # Dosya yok/bozuksa sade uyarı (UI’yi bloklamadan)
            try:
                rel_path = os.path.relpath(METRICS_FILE, PROJECT_ROOT)
            except Exception:
                rel_path = METRICS_FILE
            st.caption(f"📊 KPI dosyası bulunamadı veya geçersiz ({rel_path}).")
        
        st.subheader("Dışa aktar")
        if isinstance(a, pd.DataFrame) and not a.empty:
            csv = a.to_csv(index=False).encode("utf-8")
            st.download_button(
                "CSV indir",
                data=csv,
                file_name=f"risk_export_{int(time.time())}.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────────────────────────────────────
# SEKME: Raporlar
# ─────────────────────────────────────────────────────────────────────────────
elif sekme == "Raporlar":
    agg_current = st.session_state.get("agg")
    agg_long = st.session_state.get("agg_long")
    events_src = st.session_state.get("events")
    if not isinstance(events_src, pd.DataFrame) or events_src.empty:
        events_src = st.session_state.get("events_df")
    render_reports(events_df=events_src, agg_current=agg_current, agg_long_term=agg_long)
