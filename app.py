from __future__ import annotations

# =============================================================================
# SUTAM: Suç Tahmin Modeli — T A M   R E V I Z E  (tek dosya, çalıştırılabilir)
# -----------------------------------------------------------------------------
# Bu sürüm, utils/* ve başka modüller eksik olsa dahi fallback (yedek) 
# fonksiyonlarla uygulamayı ayağa kaldırır. Kendi modüllerin eklendikçe 
# fallback'ler otomatik olarak devre dışı kalır.
# =============================================================================

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────────────────────
# Proje kökü ve import yolu
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Sabitler (constants) — yoksa güvenli varsayılanlar
# ─────────────────────────────────────────────────────────────────────────────
try:
    from utils.constants import (
        SF_TZ_OFFSET,
        KEY_COL,
        MODEL_VERSION,
        MODEL_LAST_TRAIN,
        CATEGORIES,
    )
except Exception:
    SF_TZ_OFFSET = -7  # PDT varsayılan; kışın -8 olabilir (ileride zoneinfo önerilir)
    KEY_COL = "geoid"
    MODEL_VERSION = "v0.0.0-fallback"
    MODEL_LAST_TRAIN = "(bilinmiyor)"
    CATEGORIES = [
        "theft",
        "assault",
        "burglary",
        "robbery",
        "vehicle",
        "vandalism",
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Opsiyonel modüller (varsa kullan, yoksa zarifçe degrade et)
# ─────────────────────────────────────────────────────────────────────────────
# 1) Rapor sayfası
try:
    from components.report_view import render_reports  # type: ignore
    HAS_REPORTS = True
except Exception:
    HAS_REPORTS = False

    def render_reports(**kwargs):
        st.info("Raporlar modülü bulunamadı (components/report_view.py).")

# 2) Isı matrisi (istatistik)
try:
    from utils.heatmap import render_day_hour_heatmap  # type: ignore
except Exception:
    def render_day_hour_heatmap(*args, **kwargs):
        st.caption("Isı matrisi bileşeni bulunamadı (fallback).")

# 3) Pydeck harita
try:
    from utils.deck import build_map_fast_deck  # type: ignore
except Exception:
    build_map_fast_deck = None

# 4) Olay veri yükleyici (rapor/olay dosyaları)
try:
    from utils.reports import load_events  # type: ignore
except Exception:
    def load_events(path: str) -> pd.DataFrame:
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
        df = df.dropna(subset=["ts"]).copy()
        if "latitude" not in df.columns and "lat" in df.columns:
            df = df.rename(columns={"lat": "latitude"})
        if "longitude" not in df.columns and "lon" in df.columns:
            df = df.rename(columns={"lon": "longitude"})
        return df

# ─────────────────────────────────────────────────────────────────────────────
# UI yardımcıları — yoksa fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    from utils.ui import (
        SMALL_UI_CSS,
        render_result_card,
        build_map_fast,
        render_kpi_row,
    )
except Exception:
    SMALL_UI_CSS = """
    <style>
      .small {font-size:0.9rem}
      .muted {color:#6b7280}
    </style>
    """

    def render_result_card(df: pd.DataFrame, geoid: str, start_iso: Optional[str], horizon_h: Optional[int]):
        st.subheader("Seçili Hücre")
        st.caption(f"GEOID: {geoid} • Ufuk: {horizon_h} saat • Başlangıç (SF): {start_iso}")
        if isinstance(df, pd.DataFrame) and not df.empty:
            row = df[df.get(KEY_COL, pd.Series([], dtype=str)).astype(str) == str(geoid)].head(1)
            if not row.empty:
                exp = float(row.get("expected", pd.Series([0.0])).iloc[0])
                tier = str(row.get("tier", pd.Series(["-"])).iloc[0])
                neighborhood = str(row.get("neighborhood", pd.Series(["-"])).iloc[0])
                c1, c2, c3 = st.columns(3)
                c1.metric("E[olay] (λ)", f"{exp:.2f}")
                c2.metric("Seviye", tier)
                c3.metric("Mahalle", neighborhood)

    def render_kpi_row(items: list[tuple[str, object, str]]):
        cols = st.columns(len(items))
        for i, (title, val, helptext) in enumerate(items):
            with cols[i]:
                st.metric(title, val, help=helptext)

    def build_map_fast(
        df_agg: pd.DataFrame,
        geo_features: dict,
        geo_df: pd.DataFrame,
        show_popups: bool = True,
        patrol: Optional[dict] = None,
        show_hotspot: bool = True,
        perm_hotspot_mode: str = "heat",
        show_temp_hotspot: bool = True,
        temp_hotspot_points: Optional[pd.DataFrame] = None,
        add_layer_control: bool = True,
        risk_layer_show: bool = True,
        perm_hotspot_show: bool = True,
        temp_hotspot_show: bool = True,
        risk_layer_name: str = "Tahmin (risk)",
        perm_hotspot_layer_name: str = "Hotspot (kalıcı)",
        temp_hotspot_layer_name: str = "Hotspot (geçici)",
        **kwargs,
    ):
        m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles=None)
        folium.TileLayer("cartodbpositron", control=True).add_to(m)

        # Risk katmanı (basit choropleth)
        if isinstance(df_agg, pd.DataFrame) and not df_agg.empty:
            df = df_agg.copy()
            df[KEY_COL] = df[KEY_COL].astype(str)
            vmin = float(df["expected"].min()) if "expected" in df.columns else 0.0
            vmax = float(df["expected"].max()) if "expected" in df.columns else 1.0
            if vmin == vmax:
                vmax = vmin + 1.0
            cmap = folium.LinearColormap(
                ["#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8"],
                vmin=vmin,
                vmax=vmax,
            )

            def style_fn(feat):
                gid = str(feat.get("properties", {}).get(KEY_COL))
                val = float(df.set_index(KEY_COL)["expected"].get(gid, vmin)) if "expected" in df.columns else vmin
                return {
                    "fillColor": cmap(val),
                    "color": "#555",
                    "weight": 0.4,
                    "fillOpacity": 0.6,
                }

            gj = folium.GeoJson(
                geo_features,
                name=risk_layer_name,
                show=risk_layer_show,
                style_function=style_fn,
                highlight_function=lambda f: {"weight": 1.2, "color": "#111"},
            )
            if show_popups:
                fields = [KEY_COL]
                aliases = ["GEOID"]
                if "expected" in df.columns:
                    fields.append("expected")
                    aliases.append("E[olay] (λ)")
                if "tier" in df.columns:
                    fields.append("tier")
                    aliases.append("Seviye")
                gj.add_child(
                    folium.features.GeoJsonTooltip(
                        fields=fields, aliases=aliases, sticky=True
                    )
                )
            gj.add_to(m)

        # Geçici hotspot noktaları
        if temp_hotspot_show and isinstance(temp_hotspot_points, pd.DataFrame) and not temp_hotspot_points.empty:
            fg = folium.FeatureGroup(name=temp_hotspot_layer_name, show=True)
            for _, r in temp_hotspot_points.iterrows():
                lat = float(r.get("latitude"))
                lon = float(r.get("longitude"))
                folium.CircleMarker(location=[lat, lon], radius=3, fill=True).add_to(fg)
            fg.add_to(m)

        if add_layer_control:
            folium.LayerControl(collapsed=True).add_to(m)
        return m

# ─────────────────────────────────────────────────────────────────────────────
# GEO yardımcıları — yoksa fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    from utils.geo import load_geoid_layer, resolve_clicked_gid, join_neighborhood  # type: ignore
except Exception:
    def load_geoid_layer(path: str):
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        rows = []
        for ft in gj.get("features", []):
            props = ft.get("properties", {})
            gid = str(
                props.get(KEY_COL)
                or props.get("geoid")
                or props.get("GEOID")
                or props.get("GeoID")
                or ""
            )
            bbox = ft.get("bbox")
            if bbox and len(bbox) == 4:
                lon = (bbox[0] + bbox[2]) / 2
                lat = (bbox[1] + bbox[3]) / 2
            else:
                # Çok kaba merkez (fallback)
                lon, lat = -122.4194, 37.7749
            rows.append({KEY_COL: gid, "centroid_lon": lon, "centroid_lat": lat})
        gdf = pd.DataFrame(rows)
        return gdf, gj

    def resolve_clicked_gid(geo_df: pd.DataFrame, folium_ret: dict):
        if not folium_ret:
            return None, None
        data = folium_ret.get("last_object_clicked") or folium_ret.get("last_clicked")
        if not data:
            return None, None
        gid = data.get("tooltip") or data.get("properties", {}).get(KEY_COL)
        return (str(gid) if gid else None), data

    def join_neighborhood(df_agg: pd.DataFrame, geo_df: pd.DataFrame):
        out = df_agg.copy()
        if "neighborhood" not in out.columns:
            out["neighborhood"] = "-"
        return out

# ─────────────────────────────────────────────────────────────────────────────
# Tahmin & planlama yardımcıları — yoksa fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    from utils.forecast import precompute_base_intensity, aggregate_fast, prob_ge_k  # type: ignore
except Exception:
    def precompute_base_intensity(geo_df: pd.DataFrame) -> pd.DataFrame:
        base = geo_df[[KEY_COL]].copy()
        base["lambda_base"] = 0.05
        return base

    def aggregate_fast(
        start_iso: str,
        horizon_h: int,
        geo_df: pd.DataFrame,
        base_int: pd.DataFrame,
        events: Optional[pd.DataFrame] = None,
        near_repeat_alpha: float = 0.0,
        nr_lookback_h: int = 24,
        nr_radius_m: int = 400,
        nr_decay_h: float = 12.0,
        filters: Optional[dict] = None,
    ) -> pd.DataFrame:
        df = geo_df[[KEY_COL]].copy()
        df = df.merge(base_int, on=KEY_COL, how="left")
        lam = df["lambda_base"].fillna(0.05).astype(float)
        df["expected"] = (lam * max(horizon_h, 1)).astype(float)
        try:
            df["tier"] = pd.qcut(
                df["expected"], q=5,
                labels=["Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek"],
                duplicates="drop",
            ).astype(str)
        except Exception:
            df["tier"] = "Düşük"
        return df

    def prob_ge_k(lam: float, k: int) -> float:
        import math
        cdf = sum(math.exp(-lam) * lam**i / math.factorial(i) for i in range(k))
        return max(0.0, 1.0 - cdf)

try:
    from utils.patrol import allocate_patrols  # type: ignore
except Exception:
    def allocate_patrols(
        agg: pd.DataFrame,
        geo_df: pd.DataFrame,
        k_planned: int = 6,
        duty_minutes: int = 120,
        cell_minutes: int = 6,
        travel_overhead: float = 0.4,
    ) -> dict:
        df = agg.sort_values("expected", ascending=False).head(5 * k_planned).copy()
        zones = []
        cap = int(duty_minutes / (cell_minutes * (1 + travel_overhead))) or 1
        for i in range(k_planned):
            cells = df.iloc[i::k_planned][KEY_COL].astype(str).tolist()[:cap]
            zones.append(
                {
                    "id": f"Devriye {i + 1}",
                    "planned_cells": len(cells),
                    "capacity_cells": cap,
                    "eta_minutes": duty_minutes,
                    "utilization_pct": int(100 * len(cells) / max(1, cap)),
                    "expected_risk": float(
                        agg[agg[KEY_COL].isin(cells)]["expected"].mean() if cells else 0.0
                    ),
                }
            )
        return {"zones": zones}

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def ensure_keycol(df: pd.DataFrame, want: str = KEY_COL) -> pd.DataFrame:
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
    if df is None or df.empty:
        return df
    out = df.copy()
    rn: dict[str, str] = {}
    if "centroid_lat" not in out.columns:
        for alt in ("Centroid_Lat", "CENTROID_LAT"):
            if alt in out.columns:
                rn[alt] = "centroid_lat"
        if "lat" in out.columns and "centroid_lon" in out.columns:
            rn["lat"] = "centroid_lat"
    if "centroid_lon" not in out.columns:
        for alt in ("Centroid_Lon", "CENTROID_LON"):
            if alt in out.columns:
                rn[alt] = "centroid_lon"
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
        pts = tmp.rename(columns={"centroid_lat": "latitude", "centroid_lon": "longitude"})[
            ["latitude", "longitude"]
        ]
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

    # Kademe sınıflayıcı güvenli
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
        if x.nunique(dropna=True) < 5 or x.count() < 5:
            out["tier"] = "Çok Düşük"
            return out
        try:
            out["tier"] = pd.qcut(x, q=5, labels=labels5, duplicates="drop").astype(str)
            if out["tier"].isna().all():
                raise ValueError("qcut collapsed")
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

    agg = assign_tier_safe(agg)
    agg = ensure_keycol(agg, KEY_COL)

    # Uzun ufuk referansı (son 30 gün)
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
# UI: Sayfa
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli", layout="wide")
st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)
st.title("SUTAM: Suç Tahmin Modeli")

# Üst bant (tek kaynak): Son güncelleme (SF)
LAST_UPDATE_ISO_SF = now_sf_iso()
render_top_badge(
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
    last_update_iso=LAST_UPDATE_ISO_SF,
    daily_time_label="19:00",
)

# Geo katmanı (zorunlu dosya)
geojson_path = os.path.join(PROJECT_ROOT, "data", "sf_cells.geojson")
if not os.path.exists(geojson_path):
    st.error("'data/sf_cells.geojson' bulunamadı. Lütfen dosyayı ekleyin.")
    st.stop()

GEO_DF, GEO_FEATURES = load_geoid_layer(geojson_path)
GEO_DF = ensure_keycol(ensure_centroid_cols(GEO_DF), KEY_COL)
if GEO_DF.empty:
    st.error("GEOJSON yüklendi ancak satır bulunamadı.")
    st.stop()

# Model tabanı
BASE_INT = precompute_base_intensity(GEO_DF)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("Görünüm & Katmanlar")
engine = st.sidebar.radio("Harita motoru", ["Folium", "pydeck"], index=0, horizontal=True)
show_popups = st.sidebar.checkbox("Hücre popup'ları", value=True)

st.sidebar.subheader("Hotspot ayarları")
hotspot_cat = st.sidebar.selectbox(
    "Hotspot kategorisi",
    ["(Tüm suçlar)"] + CATEGORIES,
    index=0,
    help="Kalıcı/Geçici hotspot katmanları bu kategoriye göre gösterilir.",
)
use_hot_hours = st.sidebar.checkbox("Geçici hotspot için gün içi saat filtresi", value=False)
hot_hours_rng = st.sidebar.slider("Saat aralığı (hotspot)", 0, 24, (0, 24), disabled=not use_hot_hours)

# Zaman ufku etiketi (SF saatine göre)
_now_local = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET))
current_time = _now_local.strftime('%H:%M')
current_date = _now_local.strftime('%Y-%m-%d')
ufuk_label = f"Zaman Aralığı (from {current_time}, today, {current_date})"

ufuk = st.sidebar.radio(ufuk_label, ["24s", "48s", "7g"], index=0, horizontal=True)
max_h, step = (24, 1) if ufuk == "24s" else (48, 3) if ufuk == "48s" else (7 * 24, 24)
start_h, end_h = st.sidebar.slider("Saat filtresi", min_value=0, max_value=max_h, value=(0, max_h), step=step)

sel_categories = st.sidebar.multiselect("Kategori filtresi", ["(Hepsi)"] + CATEGORIES, default=[])
filters = {"cats": CATEGORIES if sel_categories and "(Hepsi)" in sel_categories else (sel_categories or None)}

st.sidebar.divider()
st.sidebar.header("Devriye Parametreleri")
K_planned = st.sidebar.number_input("Planlanan devriye sayısı (K)", 1, 50, 6, 1)
duty_minutes = st.sidebar.number_input("Devriye görev süresi (dk)", 15, 600, 120, 15)
cell_minutes = st.sidebar.number_input("Hücre başına ort. kontrol (dk)", 2, 30, 6, 1)
colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et")
btn_patrol = colB.button("Devriye öner")

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

st.session_state.setdefault("agg", None)
st.session_state.setdefault("agg_long", None)
st.session_state.setdefault("patrol", None)
st.session_state.setdefault("start_iso", None)
st.session_state.setdefault("horizon_h", None)
st.session_state.setdefault("explain", {})

# ─────────────────────────────────────────────────────────────────────────────
# Sekmeler
# ─────────────────────────────────────────────────────────────────────────────

sekme_options = ["Operasyon"] + (["Raporlar"] if HAS_REPORTS else [])
sekme = st.tabs(sekme_options)[0] if len(sekme_options) == 1 else None

# Eğer tek sekmeyse, aşağıdaki mantığı doğrudan çalıştırıyoruz.
# Birden fazla sekme varsa, st.tabs döndürdüğü container'ları kullanmak gerekir.

# Kolaylık için tek/çok durumunu ayırt edelim:
if HAS_REPORTS:
    t_operasyon, t_raporlar = st.tabs(["Operasyon", "Raporlar"])
else:
    t_operasyon = st.container()
    t_raporlar = None

# ─────────────────────────────────────────────────────────────────────────────
# SEKME: Operasyon
# ─────────────────────────────────────────────────────────────────────────────

with t_operasyon:
    col1, col2 = st.columns([2.4, 1.0])

    with col1:
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

        agg = st.session_state.get("agg")
        events_all = st.session_state.get("events")
        lookback_h = int(np.clip(2 * (st.session_state.get("horizon_h") or 24), 24, 72))
        ev_recent_df = recent_events(
            events_all if isinstance(events_all, pd.DataFrame) else pd.DataFrame(),
            lookback_h,
            hotspot_cat,
        )

        # Geçici hotspot için saat filtresi
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

        # Mahalle adı ekle (varsa)
        if isinstance(agg, pd.DataFrame):
            if "neighborhood" not in agg.columns and "neighborhood" in GEO_DF.columns:
                try:
                    agg = join_neighborhood(agg, GEO_DF)
                except Exception:
                    pass

        # HARİTA
        if isinstance(agg, pd.DataFrame) and not agg.empty:
            if engine == "Folium":
                try:
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
                        add_layer_control=True,
                        risk_layer_show=True,
                        perm_hotspot_show=True,
                        temp_hotspot_show=True,
                    )
                except TypeError:
                    # Eski imza uyumu
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

                ret = st_folium(
                    m,
                    key="riskmap",
                    height=560,
                    width=None,
                    returned_objects=["last_object_clicked", "last_clicked"],
                )
                if ret:
                    gid, _ = resolve_clicked_gid(GEO_DF, ret)
                    if gid:
                        st.session_state["explain"] = {"geoid": gid}
            else:
                if build_map_fast_deck is None:
                    st.error("Pydeck harita modülü bulunamadı (utils/deck.py). Lütfen Folium motorunu seçin.")
                else:
                    deck = build_map_fast_deck(
                        df_agg=agg,
                        geo_df=GEO_DF,
                        show_hotspot=True,
                        show_temp_hotspot=True,
                        temp_hotspot_points=temp_points,
                        show_risk_layer=True,
                        map_style=(
                            "mapbox://styles/mapbox/dark-v11" if st.session_state.get("dark_mode")
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
            render_result_card(st.session_state["agg"], info["geoid"], start_iso, horizon_h)
        else:
            st.info("Haritada bir hücreye tıklayın; kart burada görünecek.")

    # Sağ panel
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
                show_ci=True,
                start_iso=st.session_state.get("start_iso"),
                horizon_h=int(st.session_state.get("horizon_h") or 0),
            )
            st.dataframe(tab, use_container_width=True, height=300)
            st.caption("P(≥1)%: Seçilen ufukta en az bir olay olma olasılığı. • 95% GA: λ ± 1.96·√λ")

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
                    "devriye": z["id"],
                    "planlanan_hücre": z["planned_cells"],
                    "kapasite_hücre": z["capacity_cells"],
                    "süre_dk": z["eta_minutes"],
                    "kullanım_%": z["utilization_pct"],
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

# ─────────────────────────────────────────────────────────────────────────────
# SEKME: Raporlar
# ─────────────────────────────────────────────────────────────────────────────

if HAS_REPORTS and t_raporlar is not None:
    with t_raporlar:
        agg_current = st.session_state.get("agg")
        agg_long = st.session_state.get("agg_long")
        events_src = st.session_state.get("events")
        if not isinstance(events_src, pd.DataFrame) or events_src.empty:
            events_src = st.session_state.get("events_df")
        render_reports(events_df=events_src, agg_current=agg_current, agg_long_term=agg_long)

# ─────────────────────────────────────────────────────────────────────────────
# NOT: İlk gün için önerilen requirements (örnek):
# streamlit==1.37.1
# pandas>=2.0
# numpy>=1.24
# folium>=0.16.0
# streamlit-folium==0.20.0
# pydeck>=0.9.1
# shapely>=2.0
# requests>=2.32
# =============================================================================
