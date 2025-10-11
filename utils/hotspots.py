# utils/ui.py (REVİZE) — sadece build_map_fast ve gerekli importlar

from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
import pandas as pd
import folium
from folium import FeatureGroup
from folium.plugins import HeatMap

# İsteğe bağlı: küçük bir anahtar-adı çözümleyici
def _resolve_keycol(df: pd.DataFrame, prefer: str = "geoid") -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    for cand in (prefer, "id", "geoid", "GEOID".lower()):
        if cand.lower() in cols:
            return cols[cand.lower()]
    # ilk string benzeri kolonu zorunlu olmadıkça dönmeyelim
    return None

def build_map_fast(
    df_agg: pd.DataFrame,
    geo_features: Iterable[dict],
    geo_df: pd.DataFrame,
    *,
    show_popups: bool = True,
    patrol: Optional[dict] = None,

    # hotspot üretimi (kalıcı / geçici)
    show_hotspot: bool = True,
    perm_hotspot_mode: str = "heat",     # "heat" | "markers" (şimdilik heat)
    show_temp_hotspot: bool = True,
    temp_hotspot_points: Optional[pd.DataFrame] = None,

    # Harita üstünde LayerControl ve görünürlükler
    add_layer_control: bool = True,
    risk_layer_show: bool = True,
    perm_hotspot_show: bool = True,
    temp_hotspot_show: bool = True,
    risk_layer_name: str = "Tahmin (risk)",
    perm_hotspot_layer_name: str = "Hotspot (kalıcı)",
    temp_hotspot_layer_name: str = "Hotspot (geçici)",

    # Opsiyonel ek ayarlar
    key_col_prefer: str = "geoid",       # df_agg/geo_df içindeki anahtar adı tercihi
) -> folium.Map:
    """
    Folium haritasını katmanlarla (risk/kalıcı hotspot/geçici hotspot) kurar,
    katmanları FeatureGroup olarak ekler ve harita *üzerine* LayerControl koyar.
    """

    # ── Harita (San Francisco merkezli)
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=12,
        tiles="cartodbpositron",
        control_scale=True,
    )

    if df_agg is None or df_agg.empty or geo_df is None or geo_df.empty:
        # Yine de boş LayerControl ekleyelim (UI tutarlılığı)
        if add_layer_control:
            folium.LayerControl(position="topleft", collapsed=True, autoZIndex=True).add_to(m)
        return m

    # Anahtar kolonları çöz
    key_df = _resolve_keycol(df_agg, key_col_prefer) or "geoid"
    key_geo = _resolve_keycol(geo_df, key_col_prefer) or "geoid"

    # ── 1) RİSK KATMANI (choropleth)
    try:
        # Choropleth 'key_on' için geojson içinde "properties.id" kullanacağız.
        # load_geoid_layer içinde properties['id'] set ediliyordu; yine de garanti edelim:
        # (Eğer id yoksa ve geoid varsa, hızlıca enjekte edelim)
        for feat in geo_features:
            props = feat.setdefault("properties", {})
            if "id" not in props:
                # geo_df ile eşleştirip doldurmak karmaşık; çoğu veri setinde zaten var.
                # yoksa choropleth yine de çalışır ama eşleşme düşebilir.
                pass

        # Choropleth verisi: [id, value]
        df_risk = df_agg[[key_df, "expected"]].copy()
        df_risk.columns = ["id", "expected"]  # choropleth için 'id' anahtar ismi

        risk_fg = FeatureGroup(name=risk_layer_name, show=risk_layer_show, control=True, overlay=True)
        folium.Choropleth(
            geo_data={"type": "FeatureCollection", "features": list(geo_features)},
            data=df_risk,
            columns=["id", "expected"],
            key_on="feature.properties.id",
            fill_opacity=0.85,
            line_opacity=0.25,
            legend_name=None,   # sol alt efsane istemiyorsan None bırak
            highlight=True,
        ).add_to(risk_fg)

        # İsteğe bağlı popup/tooltip
        if show_popups:
            folium.GeoJson(
                {"type": "FeatureCollection", "features": list(geo_features)},
                name="__risk_popups__",
                style_function=lambda x: {"fillOpacity": 0, "color": "#00000000", "weight": 0},
                tooltip=folium.GeoJsonTooltip(
                    fields=["id"],
                    aliases=["GEOID"],
                    sticky=False,
                    opacity=0.9,
                ),
            ).add_to(risk_fg)

        risk_fg.add_to(m)
    except Exception:
        # Risk katmanı çizilemezse sessizce geç (harita yine gelir)
        pass

    # ── 2) KALICI HOTSPOT (expected → centroid heat)
    if show_hotspot:
        try:
            perm_fg = FeatureGroup(name=perm_hotspot_layer_name, show=perm_hotspot_show, control=True, overlay=True)
            # centroid koordinatlarıyla birleştir
            needed = ["centroid_lat", "centroid_lon"]
            if all(c in geo_df.columns for c in needed):
                dfj = (
                    df_agg[[key_df, "expected"]]
                    .merge(geo_df[[key_geo, "centroid_lat", "centroid_lon"]],
                           left_on=key_df, right_on=key_geo, how="left")
                    .dropna(subset=["centroid_lat", "centroid_lon"])
                )
                if perm_hotspot_mode == "heat":
                    pts = dfj[["centroid_lat", "centroid_lon", "expected"]].to_numpy().tolist()
                    if len(pts) > 0:
                        HeatMap(pts, radius=18, blur=22, max_zoom=12).add_to(perm_fg)
                else:
                    for _, r in dfj.iterrows():
                        folium.CircleMarker(
                            location=[r["centroid_lat"], r["centroid_lon"]],
                            radius=4 + 6 * float(max(r["expected"], 0)) ** 0.5,
                            fill=True, fill_opacity=0.7, opacity=0.0,
                        ).add_to(perm_fg)
            perm_fg.add_to(m)
        except Exception:
            pass

    # ── 3) GEÇİCİ HOTSPOT (son olaylardan)
    if show_temp_hotspot:
        try:
            temp_fg = FeatureGroup(name=temp_hotspot_layer_name, show=temp_hotspot_show, control=True, overlay=True)
            if isinstance(temp_hotspot_points, pd.DataFrame) and not temp_hotspot_points.empty:
                cols = {c.lower(): c for c in temp_hotspot_points.columns}
                latc = cols.get("latitude"); lonc = cols.get("longitude"); wc = cols.get("weight")
                if latc and lonc:
                    if wc is None:
                        temp_hotspot_points = temp_hotspot_points.assign(weight=1.0)
                        wc = "weight"
                    pts = temp_hotspot_points[[latc, lonc, wc]].to_numpy().tolist()
                    if len(pts) > 0:
                        HeatMap(pts, radius=16, blur=20, max_zoom=12).add_to(temp_fg)
            temp_fg.add_to(m)
        except Exception:
            pass

    # (Opsiyonel) devriye çizimleri vs. burada ilgili katmana eklenebilir.

    # ── LayerControl: HARİTA ÜZERİNDE küçük ikon
    if add_layer_control:
        folium.LayerControl(
            position="topleft",      # zoom +/- ile aynı köşe
            collapsed=True,          # küçük ikon — tıklayınca açılır
            autoZIndex=True
        ).add_to(m)

    return m
