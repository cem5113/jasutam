# utils/deck.py
from __future__ import annotations
import json
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import pydeck as pdk

try:
    from utils.constants import KEY_COL
except Exception:
    KEY_COL = "GEOID"

# ───────────────────────────── Renk paleti (dark-mode uyumlu) ─────────────────────────────
_COLOR: Dict[str, list[int]] = {
    "Çok Hafif":  [120, 150, 170, 180],
    "Hafif":      [90, 160, 200, 190],
    "Düşük":      [60, 130, 190, 200],
    "Orta":       [255, 170, 0, 210],
    "Yüksek":     [255, 80, 40, 220],
    "Çok Yüksek": [255, 40, 80, 230],
    # İsteğe bağlı: projede "Çok Düşük" kullanılıyorsa renk eşlemesi
    "Çok Düşük":  [166, 206, 227, 180],
}
_DEF_COLOR = [90, 120, 140, 180]

# Etiket alias’ları (aksan/boşluk/küçük-büyük farkını tolere eder)
_TIER_ALIASES: Dict[str, str] = {
    "cok yuksek": "Çok Yüksek", "çok yüksek": "Çok Yüksek",
    "yuksek": "Yüksek", "yüksek": "Yüksek",
    "orta": "Orta",
    "dusuk": "Düşük", "düşük": "Düşük",
    "hafif": "Hafif",
    "cok hafif": "Çok Hafif", "çok hafif": "Çok Hafif",
    "cok dusuk": "Çok Düşük", "çok düşük": "Çok Düşük",
}

def _norm_level(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    low = s.lower()
    return _TIER_ALIASES.get(low, s)

def _color_for(level: Optional[str]) -> list[int]:
    key = _norm_level(level)
    return _COLOR.get(key or "", _DEF_COLOR)

# ───────────────────────────── Yardımcılar ─────────────────────────────
def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aşağıdakileri garanti eder:
    - pred_expected (yoksa expected'tan)
    - risk_level (yoksa tier'dan; o da yoksa kantil tabanlı)
    - pred_expected_fmt (tooltip'ler için)
    - neighborhood (yoksa boş string)
    - risk_level normalize edilir (alias tablosundan geçer)
    """
    d = df.copy()

    # E[olay]
    if "pred_expected" not in d.columns:
        if "expected" in d.columns:
            d["pred_expected"] = pd.to_numeric(d["expected"], errors="coerce").fillna(0.0)
        else:
            d["pred_expected"] = 0.0

    # risk seviyesi
    if "risk_level" not in d.columns:
        if "tier" in d.columns:
            d["risk_level"] = d["tier"].map(_norm_level).fillna("Çok Hafif")
        else:
            x = d["pred_expected"].to_numpy(dtype=float)
            finite = np.isfinite(x)
            if finite.sum() >= 5:
                q20, q40, q60, q80 = np.quantile(x[finite], [0.20, 0.40, 0.60, 0.80])
                def to_lvl(v: float) -> str:
                    if v <= q20: return "Çok Hafif"
                    if v <= q40: return "Hafif"
                    if v <= q60: return "Düşük"
                    if v <= q80: return "Orta"
                    return "Çok Yüksek"
                d["risk_level"] = [to_lvl(float(v)) for v in x]
            else:
                d["risk_level"] = "Çok Hafif"
    else:
        d["risk_level"] = d["risk_level"].map(_norm_level).fillna("Çok Hafif")

    # Tooltip ve yardımcı alanlar
    d["pred_expected_fmt"] = pd.to_numeric(d["pred_expected"], errors="coerce").fillna(0.0).round(2)
    if "neighborhood" not in d.columns:
        d["neighborhood"] = ""

    # KEY_COL tip standardizasyonu
    d[KEY_COL] = d[KEY_COL].astype(str)

    return d

def _try_build_geojson(data: pd.DataFrame) -> tuple[bool, Dict[str, Any]]:
    """
    df['geometry'] varsa ve parse edilebiliyorsa GeoJSON FeatureCollection üretir.
    properties içine:
      - KEY_COL, neighborhood, pred_expected, pred_expected_fmt, risk_level, _color eklenir.
    """
    if "geometry" not in data.columns:
        return False, {}
    feats = []
    try:
        for _, r in data.iterrows():
            geom = r["geometry"]
            geom_obj = geom if isinstance(geom, dict) else json.loads(geom)
            lvl = _norm_level(r.get("risk_level", ""))
            feats.append({
                "type": "Feature",
                "properties": {
                    KEY_COL: r[KEY_COL],
                    "geoid": r[KEY_COL],
                    "neighborhood": r.get("neighborhood", ""),
                    "pred_expected": float(r.get("pred_expected", 0.0)),
                    "pred_expected_fmt": float(r.get("pred_expected_fmt", 0.0)),
                    "risk_level": lvl or "",
                    "_color": _color_for(lvl),
                },
                "geometry": geom_obj
            })
        return True, {"type": "FeatureCollection", "features": feats}
    except Exception:
        return False, {}

# ───────────────────────────── Ana API ─────────────────────────────
def build_map_fast_deck(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    show_poi: bool = False,                 # kullanılmıyor (imza uyumu için)
    show_transit: bool = False,             # kullanılmıyor (imza uyumu için)
    patrol=None,                            # kullanılmıyor (imza uyumu için)
    show_hotspot: bool = True,              # kalıcı hotspot (%90 üstü)
    show_temp_hotspot: bool = True,         # geçici hotspot (HeatmapLayer)
    temp_hotspot_points: pd.DataFrame | None = None,
    show_risk_layer: bool = True,           # poligon/centroid risk katmanı
    map_style: str = "mapbox://styles/mapbox/light-v11",  # "dark-v11" için koyu tema
    initial_view: Optional[Dict[str, float]] = None       # {"lat":.., "lon":.., "zoom":..}
) -> pdk.Deck:

    # Boş veri durumunda dahi bir Deck döndür
    if df_agg is None or df_agg.empty:
        return pdk.Deck(
            map_style=map_style,
            initial_view_state=pdk.ViewState(
                latitude=(initial_view or {}).get("lat", 37.7749),
                longitude=(initial_view or {}).get("lon", -122.4194),
                zoom=(initial_view or {}).get("zoom", 11.8),
            ),
            layers=[]
        )

    data = _ensure_cols(df_agg)

    # GeoJSON mu, centroid mi?
    has_geojson, data_gj = _try_build_geojson(data)
    layers: list[pdk.Layer] = []

    # ── Risk katmanı
    if show_risk_layer:
        if has_geojson:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    data=data_gj,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    get_fill_color="properties._color",  # RGBA listesi
                    get_line_color=[80, 80, 80, 120],
                    get_line_width=1,
                    extruded=False,
                    parameters={"depthTest": False},
                )
            )
        else:
            # centroid yolu
            centers = (
                data.merge(
                    geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]],
                    on=KEY_COL, how="left"
                )
                .dropna(subset=["centroid_lat", "centroid_lon"])
                .copy()
            )
            centers["_color"] = centers["risk_level"].apply(_color_for)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=centers,
                    pickable=True,
                    get_position="[centroid_lon, centroid_lat]",
                    get_radius=80,
                    radius_min_pixels=2,
                    radius_max_pixels=80,
                    get_fill_color="_color",          # RGBA listesi
                )
            )

    # ── Geçici hotspot (HeatmapLayer)
    if show_temp_hotspot and isinstance(temp_hotspot_points, pd.DataFrame) and not temp_hotspot_points.empty:
        pts = temp_hotspot_points.copy()
        cols_lower = {c.lower(): c for c in pts.columns}
        lat = cols_lower.get("latitude") or cols_lower.get("lat")
        lon = cols_lower.get("longitude") or cols_lower.get("lon")
        if lat and lon:
            pts = pts.rename(columns={lat: "lat", lon: "lon"})
            if "weight" not in pts.columns:
                pts["weight"] = 1.0
            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=pts,
                    get_position="[lon, lat]",
                    get_weight="weight",
                    radius_pixels=40,
                    aggregation="SUM",
                )
            )

    # ── Kalıcı hotspot (üst %10, Scatterplot)
    if show_hotspot:
        metric = "pred_expected" if "pred_expected" in data.columns else None
        if metric:
            x = pd.to_numeric(data[metric], errors="coerce").fillna(0.0).to_numpy()
            if np.isfinite(x).sum() >= 1:
                thr = float(np.quantile(x[np.isfinite(x)], 0.90))
                strong = (
                    data[data[metric] >= thr]
                    .merge(geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]],
                           on=KEY_COL, how="left")
                    .dropna(subset=["centroid_lat", "centroid_lon"])
                    .copy()
                )
                if not strong.empty:
                    layers.append(
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=strong,
                            pickable=False,
                            get_position="[centroid_lon, centroid_lat]",
                            get_radius=120,
                            get_fill_color=[139, 0, 0, 200],  # koyu kırmızı marker
                        )
                    )

    # ── Tooltip
    tooltip = {
        "html": (
            f"<b>{KEY_COL}:</b> {{{KEY_COL}}}<br>"
            "<b>Mahalle:</b> {neighborhood}<br>"
            "<b>E[olay]:</b> {pred_expected_fmt}<br>"
            "<b>Risk seviyesi:</b> {risk_level}"
        ),
        "style": {"backgroundColor": "rgba(0,0,0,0.78)", "color": "white"},
    }

    # ── Görünüm
    view_state = pdk.ViewState(
        latitude=(initial_view or {}).get("lat", 37.7749),
        longitude=(initial_view or {}).get("lon", -122.4194),
        zoom=(initial_view or {}).get("zoom", 11.8),
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=map_style,
        tooltip=tooltip,
    )
