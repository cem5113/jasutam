# utils/deck.py
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pydeck as pdk

try:
    from utils.constants import KEY_COL
except Exception:
    KEY_COL = "GEOID"

# Renk paleti (risk seviyesi → RGBA)
_COLOR = {
    "Çok Hafif":  [198, 219, 239, 120],
    "Hafif":      [158, 202, 225, 140],
    "Düşük":      [107, 174, 214, 160],
    "Orta":       [49, 130, 189, 190],
    "Yüksek":     [239, 59, 44, 200],     # bazı modellerde 'Yüksek' olabilir
    "Çok Yüksek": [255, 102, 102, 210],,
}
_DEF_COLOR = [166, 206, 227, 140]

def _color_for(level: str):
    return _COLOR.get(str(level), _DEF_COLOR)

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """pred_expected, risk_level (tier fallback) ve basit kantil tabanlı tier üretimi."""
    d = df.copy()
    # E[olay]
    if "pred_expected" not in d.columns:
        if "expected" in d.columns:
            d["pred_expected"] = d["expected"].astype(float)
        else:
            d["pred_expected"] = 0.0

    # risk seviyesi
    if "risk_level" not in d.columns:
        if "tier" in d.columns:
            d["risk_level"] = d["tier"].astype(str)
        else:
            # Kantil tabanlı otomatik seviye (q20/40/60/80)
            x = d["pred_expected"].to_numpy()
            if len(x) >= 5 and np.any(np.isfinite(x)):
                q20, q40, q60, q80 = np.quantile(x[np.isfinite(x)], [0.20, 0.40, 0.60, 0.80])
                def to_lvl(v):
                    if v <= q20: return "Çok Hafif"
                    if v <= q40: return "Hafif"
                    if v <= q60: return "Düşük"
                    if v <= q80: return "Orta"
                    return "Çok Yüksek"
                d["risk_level"] = [to_lvl(float(v)) for v in d["pred_expected"].to_numpy()]
            else:
                d["risk_level"] = "Çok Hafif"

    # Gösterim için 2 ondalık
    d["pred_expected_fmt"] = d["pred_expected"].astype(float).round(2)

    # Tooltip alan adlarını sabitle
    if "neighborhood" not in d.columns:
        d["neighborhood"] = ""

    return d

def build_map_fast_deck(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    show_poi: bool = False,             # şimdilik kullanılmıyor
    show_transit: bool = False,         # şimdilik kullanılmıyor
    patrol=None,                        # opsiyonel
    show_hotspot: bool = True,          # kalıcı hotspot
    show_temp_hotspot: bool = True,     # geçici hotspot
    temp_hotspot_points: pd.DataFrame | None = None,
    show_risk_layer: bool = True        # ← app.py'deki yeni bayrak
) -> pdk.Deck:

    # Boş durumda da bir deck döndür
    if df_agg is None or df_agg.empty:
        return pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v11",
            initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=11.8),
            layers=[]
        )

    data = _ensure_cols(df_agg)
    data[KEY_COL] = data[KEY_COL].astype(str)

    # GeoJSON var mı?
    has_geojson = False
    if "geometry" in data.columns:
        try:
            feats = []
            for _, r in data.iterrows():
                geom = r["geometry"]
                feats.append({
                    "type": "Feature",
                    "properties": {
                        KEY_COL: r[KEY_COL],
                        "geoid": r[KEY_COL],
                        "neighborhood": r.get("neighborhood", ""),
                        "pred_expected": float(r.get("pred_expected", 0.0)),
                        "pred_expected_fmt": float(r.get("pred_expected_fmt", 0.0)),
                        "risk_level": r.get("risk_level", ""),
                        "_color": _color_for(r.get("risk_level", "")),
                    },
                    "geometry": geom if isinstance(geom, dict) else json.loads(geom)
                })
            data_gj = {"type": "FeatureCollection", "features": feats}
            has_geojson = True
        except Exception:
            has_geojson = False

    layers = []

    # --- RISK LAYER ---
    if show_risk_layer:
        if has_geojson:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    data=data_gj,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    get_fill_color="properties._color",
                    get_line_color=[80, 80, 80, 120],
                    get_line_width=1,
                    extruded=False,
                    parameters={"depthTest": False},
                )
            )
        else:
            # Centroid üzerinden (geo_df ile birleştir)
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
                    get_fill_color="_color",
                )
            )

    # --- TEMP HOTSPOT (geçici) ---
    if show_temp_hotspot and isinstance(temp_hotspot_points, pd.DataFrame) and not temp_hotspot_points.empty:
        pts = temp_hotspot_points.copy()
        # kolonları normalize et
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

    # --- PERM HOTSPOT (kalıcı, üst %10 marker) ---
    if show_hotspot:
        metric = "pred_expected" if "pred_expected" in data.columns else None
        if metric:
            thr = float(np.quantile(data[metric].to_numpy(), 0.90))
            strong = (
                data[data[metric] >= thr]
                .merge(geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL, how="left")
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
                        get_fill_color=[139, 0, 0, 200],
                    )
                )

    # --- TOOLTIP ---
    # Not: q10/q90 bilgisi eklenmiyor (istenmedi).
    tooltip = {
        "html": (
            f"<b>{KEY_COL}:</b> {{{KEY_COL}}}<br>"
            "<b>Mahalle:</b> {neighborhood}<br>"
            "<b>E[olay]:</b> {pred_expected_fmt}<br>"
            "<b>Risk seviyesi:</b> {risk_level}"
        ),
        "style": {"backgroundColor": "rgba(0,0,0,0.78)", "color": "white"}
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=37.7749, longitude=-122.4194, zoom=11.8),
        map_style="mapbox://styles/mapbox/light-v11",
        tooltip=tooltip
    )
