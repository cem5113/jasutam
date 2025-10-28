from __future__ import annotations
import os, sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

import streamlit as st
st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli Yeni Versiyon", layout="wide")  # ← ilk Streamlit çağrısı

import folium
import numpy as np
import pandas as pd
from streamlit_folium import st_folium

# ── constants / local imports vs.
from utils.constants import SF_TZ_OFFSET, KEY_COL, MODEL_VERSION, MODEL_LAST_TRAIN, CATEGORIES
from utils.geo import load_geoid_layer, resolve_clicked_gid
from utils.forecast import precompute_base_intensity, aggregate_fast, prob_ge_k
from utils.patrol import allocate_patrols
# (buraya esnek sarmalayıcıyı ekle)

import inspect
from utils.patrol import allocate_patrols as _ap

# güvenli imza yazdırma (log/debug)
try:
    st.write("allocate_patrols signature:", str(inspect.signature(_ap)))
except Exception as e:
    st.write("allocate_patrols signature okunamadı:", str(e))

# ── constants
from utils.constants import SF_TZ_OFFSET, KEY_COL, MODEL_VERSION, MODEL_LAST_TRAIN, CATEGORIES

# ── local path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── local modules
# ── local modules
from utils.geo import load_geoid_layer, resolve_clicked_gid
from utils.forecast import precompute_base_intensity, aggregate_fast, prob_ge_k
from utils.patrol import allocate_patrols         # ← BU SATIR VAR
from utils.ui import (
    SMALL_UI_CSS,
    render_result_card,
    build_map_fast,
    render_kpi_row,
    render_day_hour_heatmap as _fallback_heatmap,
)

import inspect
try:
    from utils.patrol import allocate_patrols as _allocate_patrols
except Exception:
    _allocate_patrols = None

from utils.patrol import allocate_patrols as __allocate_patrols
def allocate_patrols(*args, **kwargs):
    if "k_planned" not in kwargs:
        for alias in ["K", "k", "n", "num_units", "n_units", "units", "num_zones", "count"]:
            if alias in kwargs:
                kwargs["k_planned"] = kwargs.pop(alias)
                break
    return __allocate_patrols(*args, **kwargs)

    # K/k/… → k_planned’e eşle
    for alias in ["K", "k", "n", "num_units", "n_units", "units", "num_zones", "count"]:
        if alias in kwargs:
            if "k_planned" not in kwargs:
                kwargs["k_planned"] = kwargs.pop(alias)
            else:
                kwargs.pop(alias, None)
            break

    return _allocate_patrols(*args, **kwargs)
    
# utils/heatmap varsa onu kullan, yoksa ui.py'deki fallback'i kullan
try:
    from utils.heatmap import render_day_hour_heatmap  # type: ignore
except Exception:
    render_day_hour_heatmap = _fallback_heatmap

# reports (optional)
try:
    from components.report_view import render_reports  # type: ignore
    HAS_REPORTS = True
except ModuleNotFoundError:
    HAS_REPORTS = False
    def render_reports(**kwargs):
        st.info("Raporlar modülü bulunamadı (components/report_view.py).")

# pydeck (optional)
try:
    from utils.deck import build_map_fast_deck  # type: ignore
except ImportError:
    build_map_fast_deck = None

# events loader (fallback)
try:
    from utils.reports import load_events  # type: ignore
except Exception:
    def load_events(path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        lower = {str(c).strip().lower(): c for c in df.columns}
        ts_col = next((lower[c] for c in ["ts","timestamp","datetime","date_time",
                                          "reported_at","occurred_at","time","date"]
                       if c in lower), None)
        if not ts_col:
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])
        if "latitude" not in df.columns and "lat" in df.columns:
            df = df.rename(columns={"lat": "latitude"})
        if "longitude" not in df.columns and "lon" in df.columns:
            df = df.rename(columns={"lon": "longitude"})
        return df

# ───────────────────────────────── helpers ─────────────────────────────────
def _to_float(x, default=None):
    try: return float(x)
    except Exception: return default

def _norm_point(pt):
    # (lat,lon) | (lon,lat) | dict | [..] -> (lat,lon)
    if pt is None: return None
    if isinstance(pt, dict):
        lat = _to_float(pt.get("lat", pt.get("latitude")))
        lon = _to_float(pt.get("lon", pt.get("longitude")))
        return (lat, lon) if lat is not None and lon is not None else None
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        a, b = _to_float(pt[0]), _to_float(pt[1])
        if a is None or b is None: return None
        # Heuristik: |lon| > 100 ise (lon,lat) gelmiştir → çevir
        return (b, a) if abs(a) > 100 and abs(b) < 100 else (a, b)
    return None

def _route_from_cells(zone, geo_df):
    cells = zone.get("cells") or zone.get("planned_cells") or []
    if not cells: return []
    df = geo_df.copy()
    df[KEY_COL] = df[KEY_COL].astype(str)
    sel = df[df[KEY_COL].isin([str(c) for c in cells])]
    if sel.empty or not {"centroid_lat","centroid_lon"}.issubset(sel.columns): return []
    if "expected" in sel.columns:
        sel = sel.sort_values("expected", ascending=False)
    return list(zip(sel["centroid_lat"].astype(float), sel["centroid_lon"].astype(float)))

def _centroid_for_zone(zone, route):
    c = zone.get("centroid")
    if isinstance(c, dict) and {"lat","lon"} <= set(c.keys()):
        lat, lon = _to_float(c["lat"]), _to_float(c["lon"])
        if lat is not None and lon is not None:
            return {"lat": lat, "lon": lon}
    if route:
        i = len(route)//2
        return {"lat": float(route[i][0]), "lon": float(route[i][1])}
    return None

def _normalize_patrol(patrol, geo_df, agg_df):
    """utils.patrol.allocate_patrols çıktısını folium çizimine uygun hale getirir."""
    if not patrol or not isinstance(patrol, dict):
        return {"zones": []}
    zones = patrol.get("zones") or patrol.get("Zones") or patrol.get("data") or []
    out = []
    for z in zones:
        zid = z.get("id") or z.get("zone") or f"Z{len(out)+1}"
        # route normalizasyonu
        raw_route = z.get("route") or []
        route = []
        for pt in raw_route:
            p = _norm_point(pt)
            if p: route.append(p)
        if not route:
            # cell listesi varsa centroidlerden rota türet
            route = _route_from_cells(z, geo_df)
        cen = _centroid_for_zone(z, route)
        out.append({
            "id": str(zid),
            "route": route,  # [(lat,lon), ...] olabilir ya da boş olabilir
            "centroid": cen or {"lat": 37.7749, "lon": -122.4194},
            "cells": z.get("cells") or z.get("planned_cells") or [],
            "expected_risk": float(z.get("expected_risk", 0.0)),
            "eta_minutes": int(z.get("eta_minutes", 0)),
            "utilization_pct": float(z.get("utilization_pct", 0.0)),
            "planned_cells": z.get("planned_cells", []),
            "capacity_cells": z.get("capacity_cells", 0),
        })
    return {"zones": out}


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

def render_top_badge(model_version: str, last_train: str, last_update_iso: str, daily_time_label: str = "19:00"):
    parts = [
        "**SUTAM**",
        f"• Model: {model_version}",
        f"• Son eğitim: {last_train}",
        f"• Günlük güncellenir: ~{daily_time_label} (SF)",
        f"• Son güncelleme (SF): {last_update_iso}",
    ]
    st.markdown(" ".join(parts))

def run_prediction(
    start_h: int,
    end_h: int,
    filters: dict,
    geo_df: pd.DataFrame,
    base_int: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], str, int]:
    start_dt = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET + start_h)).replace(minute=0, second=0, microsecond=0)
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

    # robust tier
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
            if out["tier"].isna().all(): raise ValueError
            return out
        except Exception:
            pass
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
            med = float(np.nanmedian(x)); p75 = float(np.nanquantile(x, 0.75)); p90 = float(np.nanquantile(x, 0.90))
            def fallback(v: float) -> str:
                if v <= med * 0.5: return "Çok Düşük"
                if v <= med:       return "Düşük"
                if v <= p75:       return "Orta"
                if v <= p90:       return "Yüksek"
                return "Çok Yüksek"
            out["tier"] = [fallback(float(v)) for v in x]; return out

    agg = assign_tier_safe(agg)
    agg = ensure_keycol(agg, KEY_COL)

    # long horizon reference (30d) – optional
    try:
        long_start_iso = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET - 30 * 24)).replace(minute=0, second=0, microsecond=0).isoformat()
        agg_long = aggregate_fast(long_start_iso, 30 * 24, geo_df, base_int, events=events_df, near_repeat_alpha=0.0, filters=None)
        agg_long = ensure_keycol(agg_long, KEY_COL)
    except Exception:
        agg_long = None

    return agg, agg_long, start_iso, horizon_h

def top_risky_table(
    df_agg: pd.DataFrame, n: int, show_ci: bool, start_iso: Optional[str], horizon_h: int
) -> pd.DataFrame:
    def poisson_ci(lam: float, z: float = 1.96) -> tuple[float, float]:
        s = float(np.sqrt(max(lam, 1e-9))); return max(0.0, lam - z * s), lam + z * s
    cols = [KEY_COL, "expected"] + (["nr_boost"] if "nr_boost" in df_agg.columns else [])
    df = ensure_keycol(df_agg, KEY_COL)[cols].sort_values("expected", ascending=False).head(n).reset_index(drop=True)
    lam = df["expected"].to_numpy()
    df["P(≥1)%"] = [round(prob_ge_k(l, 1) * 100, 1) for l in lam]
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

# ─────────────────────────── UI ───────────────────────────

st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli YEni Versiyon", layout="wide")
st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)
st.title("SUTAM: Suç Tahmin Modeli")

LAST_UPDATE_ISO_SF = now_sf_iso()
render_top_badge(MODEL_VERSION, MODEL_LAST_TRAIN, LAST_UPDATE_ISO_SF, daily_time_label="19:00")

# GEO
# ── GEO (revize)
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def _load_geo_cached(path: str):
    df, feats = load_geoid_layer(path)
    # Kolon isimlerini güvenceye al
    df = ensure_keycol(ensure_centroid_cols(df), KEY_COL)

    if df is None or df.empty:
        raise RuntimeError("GEO katmanı boş döndü.")

    # GEOID zorunlu: stringe çevir + trim
    df[KEY_COL] = df[KEY_COL].astype(str).str.strip()
    df = df.dropna(subset=[KEY_COL])

    # centroid kolonlarını güvenli sayıya çevir
    for c in ("centroid_lat", "centroid_lon"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Kaç satırda centroid eksik?
    miss_centroid = (
        int(df[["centroid_lat", "centroid_lon"]].isna().any(axis=1).sum())
        if {"centroid_lat", "centroid_lon"}.issubset(df.columns) else None
    )

    return df, feats, miss_centroid

try:
    GEO_DF, GEO_FEATURES, _MISS_CENTROID = _load_geo_cached("data/sf_cells.geojson")
except Exception as e:
    st.error(f"GEOJSON yüklenemedi: {e}")
    st.stop()

if GEO_DF.empty:
    st.error("GEOJSON yüklendi fakat satır bulunamadı.")
    st.stop()

# Centroid kolonları yoksa uyar; varsa çok eksikse bilgi ver
if not {"centroid_lat", "centroid_lon"}.issubset(GEO_DF.columns):
    st.warning("Uyarı: 'centroid_lat' / 'centroid_lon' kolonları yok. Rota/hotspot merkezleri çizilemeyebilir.")
elif _MISS_CENTROID and _MISS_CENTROID > 0:
    st.info(f"Bilgi: {int(_MISS_CENTROID)} hücrede centroid eksik. Bu hücreler rota/işaretleme sırasında atlanabilir.")

# base intensity
BASE_INT = precompute_base_intensity(GEO_DF)

# ── sidebar
with st.sidebar.expander("🔎 Veri Tanı / Health Check", expanded=False):
    st.markdown("**GEO katmanı**")
    st.write("satır:", len(GEO_DF))
    st.write("kolonlar:", list(GEO_DF.columns))
    miss_centroid = GEO_DF[["centroid_lat","centroid_lon"]].isna().any(axis=1).sum() if {"centroid_lat","centroid_lon"}.issubset(GEO_DF.columns) else "—"
    st.write("eksik centroid:", miss_centroid)
    _events = load_events_safe()
    st.markdown("**Events (ham)**")
    st.write("satır:", len(_events))
    st.write("kolonlar:", list(_events.columns))
    if not _events.empty and "ts" in _events.columns:
        _tmp = _events.copy()
        _tmp["ts"] = pd.to_datetime(_tmp["ts"], utc=True, errors="coerce")
        _tmp = _tmp.dropna(subset=["ts"])
        _tmp["ts_sf"] = _tmp["ts"] + pd.Timedelta(hours=SF_TZ_OFFSET)
        st.write("hour dağılımı (son):", _tmp["ts_sf"].dt.hour.value_counts().sort_index().tail())
        st.write("dow dağılımı:", _tmp["ts_sf"].dt.dayofweek.value_counts().sort_index())

with st.sidebar:
    if HAS_REPORTS:
        sekme = st.radio("Sekme", ["Operasyon", "Raporlar"], index=0, horizontal=True, label_visibility="collapsed")
    else:
        sekme = "Operasyon"

    engine = st.radio("Harita motoru", ["Folium", "pydeck"], index=0, horizontal=True)

    st.markdown("**Harita katmanları**")
    show_popups = st.checkbox("Hücre popup'larını (en olası 3 suç) göster", value=True)

    st.markdown("**Grafik kapsamı**")
    scope = st.radio("Grafik kapsamı", ["Tüm şehir", "Seçili hücre"], index=0, label_visibility="collapsed")

    show_hotspot = True
    show_temp_hotspot = True
    hotspot_cat = st.selectbox("Hotspot kategorisi", ["(Tüm suçlar)"] + CATEGORIES, index=0)

    use_hot_hours = st.checkbox("Geçici hotspot için gün içi saat filtresi", value=False)
    if use_hot_hours:
        hot_hours_rng = st.slider("Saat aralığı (hotspot)", 0, 24, (0, 24))
    else:
        hot_hours_rng = (0, 24)

    current_time = datetime.now().strftime('%H:%M')
    current_date = datetime.now().strftime('%Y-%m-%d')
    ufuk_label = f"Zaman Aralığı (from {current_time}, today, {current_date})"
    ufuk = st.radio(ufuk_label, ["24s", "48s", "7g"], index=0, horizontal=True)
    max_h, step = (24, 1) if ufuk == "24s" else (48, 3) if ufuk == "48s" else (7*24, 24)
    start_h, end_h = st.slider("Saat filtresi", 0, max_h, (0, max_h), step=step)

    sel_categories = st.multiselect("Kategori", ["(Hepsi)"] + CATEGORIES, default=[])
    filters = {"cats": CATEGORIES if sel_categories and "(Hepsi)" in sel_categories else (sel_categories or None)}

    st.markdown("### Devriye Planı")
    K_planned = st.number_input("Planlanan devriye sayısı (K)", 1, 50, 6, 1)
    duty_minutes = st.number_input("Devriye görev süresi (dk)", 15, 600, 120, 15)
    cell_minutes = st.number_input("Hücre başına ort. kontrol (dk)", 2, 30, 6, 1)

    run_hourly_heatmap = False
    
    colA, colB = st.columns(2)
    btn_predict = colA.button("Tahmin et")
    btn_patrol  = colB.button("Devriye öner")

# state
st.session_state.setdefault("agg", None)
st.session_state.setdefault("agg_long", None)
st.session_state.setdefault("patrol", None)
st.session_state.setdefault("start_iso", None)
st.session_state.setdefault("horizon_h", None)
st.session_state.setdefault("explain", {})

# ── main
if sekme == "Operasyon":
    col1, col2 = st.columns([2.4, 1.0])

    with col1:
        if btn_predict or st.session_state["agg"] is None:
            agg, agg_long, start_iso, horizon_h = run_prediction(start_h, end_h, filters, GEO_DF, BASE_INT)
            st.session_state.update({"agg": agg, "agg_long": agg_long, "patrol": None,
                                     "start_iso": start_iso, "horizon_h": horizon_h,
                                     "events": st.session_state.get("events_df")})

        agg = st.session_state["agg"]
        # >>> YENİ: Devriye planını hesapla ve state'e yaz
        if btn_patrol:
            if isinstance(agg, pd.DataFrame) and not agg.empty:
                try:
                    plan = allocate_patrols(
                        df_agg=agg,
                        geo_df=GEO_DF,
                        k_planned=int(K_planned),     # ← doğru argüman adı
                        duty_minutes=int(duty_minutes),
                        cell_minutes=int(cell_minutes),
                        # travel_overhead=0.4  # varsa opsiyonel
                    )     
                    st.session_state["patrol"] = _normalize_patrol(plan, GEO_DF, agg)
                    st.experimental_rerun()  # rotayı hemen çizmek için yeniden çalıştır
                except Exception as e:
                    st.error(f"Devriye planı oluşturulamadı: {e}")
            else:
                st.warning("Önce ‘Tahmin et’ ile risk haritasını üretin.")

        
        events_all = st.session_state.get("events")
        lookback_h = int(np.clip(2 * (st.session_state.get("horizon_h") or 24), 24, 72))
        ev_recent_df = recent_events(events_all if isinstance(events_all, pd.DataFrame) else pd.DataFrame(),
                                     lookback_h, hotspot_cat)

        if scope == "Seçili hücre" and st.session_state.get("explain", {}).get("geoid") and KEY_COL in ev_recent_df.columns:
            gid = str(st.session_state["explain"]["geoid"])
            ev_recent_df = ev_recent_df[ev_recent_df[KEY_COL].astype(str) == gid]

        temp_points = ev_recent_df[["latitude", "longitude", "weight"]] if not ev_recent_df.empty \
                      else pd.DataFrame(columns=["latitude", "longitude", "weight"])
        if use_hot_hours and not temp_points.empty and "ts" in ev_recent_df.columns:
            h1, h2 = hot_hours_rng[0], (hot_hours_rng[1] - 1) % 24
            temp_points = ev_recent_df[ev_recent_df["ts"].dt.hour.between(h1, h2)][["latitude", "longitude", "weight"]]
        if temp_points.empty and isinstance(agg, pd.DataFrame) and not agg.empty:
            temp_points = make_temp_hotspot_from_agg(agg, GEO_DF, topn=80)
        st.sidebar.caption(f"Geçici hotspot noktası: {len(temp_points)}")

        if isinstance(agg, pd.DataFrame):
            if "neighborhood" not in agg.columns and "neighborhood" in GEO_DF.columns:
                try:
                    from utils.geo import join_neighborhood
                    agg = join_neighborhood(agg, GEO_DF)
                except Exception:
                    pass

        # map
        if agg is not None:
            if engine == "Folium":
                try:
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
                except TypeError:
                    m = build_map_fast(
                        df_agg=agg, geo_features=GEO_FEATURES, geo_df=GEO_DF,
                        show_popups=show_popups, patrol=st.session_state.get("patrol"),
                        show_hotspot=True, perm_hotspot_mode="heat",
                        show_temp_hotspot=True, temp_hotspot_points=temp_points,
                    )
                # remove internal LC, add our LC
                for k, ch in list(m._children.items()):
                    if isinstance(ch, folium.map.LayerControl):
                        del m._children[k]
                folium.TileLayer(
                    tiles="CartoDB positron", name="cartodbpositron", control=True,
                    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
                         'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                ).add_to(m)

                _sel = st.session_state.get("explain", {}).get("geoid")
                if _sel:
                    try:
                        row = GEO_DF[GEO_DF[KEY_COL].astype(str) == str(_sel)].head(1)
                        if not row.empty and {"centroid_lat","centroid_lon"}.issubset(row.columns):
                            lat = float(row["centroid_lat"].iloc[0]); lon = float(row["centroid_lon"].iloc[0])
                            folium.CircleMarker(location=(lat, lon), radius=10, weight=3, color="#111",
                                                fill=False, tooltip=f"Seçili GEOID: {str(_sel)}").add_to(m)
                    except Exception:
                        pass

                folium.LayerControl(position="topright", collapsed=True, autoZIndex=True).add_to(m)
                ret = st_folium(m, key="riskmap", height=540, width=800,
                                returned_objects=["last_object_clicked", "last_clicked"])
                if ret:
                    gid, _ = resolve_clicked_gid(GEO_DF, ret)
                    if gid:
                        st.session_state["explain"] = {"geoid": gid}
            else:
                if build_map_fast_deck is None:
                    st.error("Pydeck harita modülü bulunamadı (utils/deck.py). Lütfen Folium motorunu seçin.")
                else:
                    deck = build_map_fast_deck(
                        df_agg=agg, geo_df=GEO_DF, show_hotspot=True, show_temp_hotspot=True,
                        temp_hotspot_points=temp_points, show_risk_layer=True,
                        map_style=("mapbox://styles/mapbox/dark-v11" if st.session_state.get("dark_mode")
                                   else "mapbox://styles/mapbox/light-v11"),
                        initial_view={"lat": 37.7749, "lon": -122.4194, "zoom": 11.8},
                    )
                    st.pydeck_chart(deck)
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

        # result card
        start_iso = st.session_state.get("start_iso")
        horizon_h = st.session_state.get("horizon_h")
        info = st.session_state.get("explain")
        if info and info.get("geoid"):
            render_result_card(agg, info["geoid"], start_iso, horizon_h)
        else:
            st.info("Haritada bir hücreye tıklayın; kart burada görünecek.")

    with col2:
        st.subheader("Risk Özeti", anchor=False)
        a = st.session_state.get("agg")
        if isinstance(a, pd.DataFrame) and not a.empty:
            kpi_expected = round(float(a["expected"].sum()), 2)
            cnts = {
                "Çok Yüksek": int((a.get("tier", pd.Series([], dtype="string")) == "Çok Yüksek").sum()),
                "Yüksek":     int((a.get("tier", pd.Series([], dtype="string")) == "Yüksek").sum()),
                "Orta":       int((a.get("tier", pd.Series([], dtype="string")) == "Orta").sum()),
                "Düşük":      int((a.get("tier", pd.Series([], dtype="string")) == "Düşük").sum()),
                "Çok Düşük":  int((a.get("tier", pd.Series([], dtype="string")) == "Çok Düşük").sum()),
            }
            render_kpi_row([
                ("Beklenen olay (ufuk)", kpi_expected, "Seçili zaman ufkunda toplam beklenen olay sayısı"),
                ("Çok Yüksek", cnts["Çok Yüksek"], "En yüksek riskli hücre sayısı (üst %20)"),
                ("Yüksek", cnts["Yüksek"], "Yüksek kademe riskli hücre sayısı"),
                ("Orta", cnts["Orta"], "Orta kademe riskli hücre sayısı"),
                ("Düşük", cnts["Düşük"], "Düşük kademe riskli hücre sayısı"),
                ("Çok Düşük", cnts["Çok Düşük"], "En düşük riskli hücre sayısı (alt %20)"),
            ])
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

        st.subheader("Top-5 kritik GEOID")
        if isinstance(a, pd.DataFrame) and not a.empty:
            tab = top_risky_table(a, n=5, show_ci=True,
                                  start_iso=st.session_state.get("start_iso"),
                                  horizon_h=int(st.session_state.get("horizon_h") or 0))
            st.dataframe(tab, use_container_width=True)
            st.markdown("Seç / odağı haritada göster:")
            cols = st.columns(len(tab))
            for i, row in enumerate(tab.itertuples()):
                with cols[i]:
                    if st.button(str(row.geoid)):
                        st.session_state["explain"] = {"geoid": str(row.geoid)}
                        st.experimental_rerun()
            st.caption("Butona tıklayınca haritada centroid işaretlenir ve açıklama kartı güncellenir.")
            
        st.subheader("Devriye özeti")
        patrol = st.session_state.get("patrol")
        if patrol and patrol.get("zones"):
            rows = [{
                "zone": z["id"], "cells_planned": z["planned_cells"], "capacity_cells": z["capacity_cells"],
                "eta_minutes": z["eta_minutes"], "utilization_%": z["utilization_pct"],
                "avg_risk(E[olay])": round(z["expected_risk"], 2),
            } for z in patrol["zones"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

        st.subheader("Gün × Saat Isı Matrisi")
        if st.session_state.get("agg") is not None and st.session_state.get("start_iso"):
            H = int(st.session_state.get("horizon_h") or 24)
            events_all = st.session_state.get("events")
        
            # utils/heatmap varsa gerçek saatlik; yoksa ui.py fallback'i zaten import logic ile devreye girer
            render_day_hour_heatmap(
                agg=st.session_state["agg"],
                start_iso=st.session_state["start_iso"],
                horizon_h=H,
            )
        else:
            st.caption("Isı matrisi, bir tahmin üretildiğinde gösterilir.")

# ── reports tab
elif sekme == "Raporlar":
    agg_current = st.session_state.get("agg")
    agg_long = st.session_state.get("agg_long")
    events_src = st.session_state.get("events")
    if not isinstance(events_src, pd.DataFrame) or events_src.empty:
        events_src = st.session_state.get("events_df")
    render_reports(events_df=events_src, agg_current=agg_current, agg_long_term=agg_long)
