# utils/deck.py
from __future__ import annotations
import pydeck as pdk
import pandas as pd

try:
    from utils.constants import KEY_COL, CRIME_TYPES
except Exception:
    KEY_COL = "GEOID"
    CRIME_TYPES = []

def build_map_fast_deck(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    # Overlayler
    show_poi: bool = False,             # (şimdilik kullanılmıyor)
    show_transit: bool = False,         # (şimdilik kullanılmıyor)
    patrol: dict | None = None,         # (opsiyonel, ileride çizilebilir)
    # Hotspot ve katman görünürlüğü
    show_hotspot: bool = True,
    show_temp_hotspot: bool = True,
    temp_hotspot_points: pd.DataFrame | None = None,
    # Yeni: Folium ile hizalı kontroller
    risk_layer_show: bool = True,
    perm_hotspot_show: bool = True,
    temp_hotspot_show: bool = True,
    selected_type: str | None = None,   # kalıcı hotspot ağırlığı için
    perm_hotspot_mode: str = "heat",    # "heat" | "markers"
):
    """
    Pydeck hızlı harita; Folium'daki build_map_fast ile parametre olarak hizalıdır.
    - risk_layer_show: hücre noktaları (tier renklendirmesi)
    - perm_hotspot_show: kalıcı hotspot (ısı/marker)
    - temp_hotspot_show: geçici hotspot (olay ısısı)
    - selected_type varsa kalıcı hotspot ağırlığı bu sütundan alınır, yoksa 'expected'.
    """
    center = [37.7749, -122.4194]
    layers = []

    # Hücre centroid’leri + renk (tier)
    if isinstance(df_agg, pd.DataFrame) and not df_agg.empty:
        cells = (
            df_agg.merge(
                geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]],
                on=KEY_COL, how="left"
            )
            .dropna(subset=["centroid_lat", "centroid_lon"])
            .copy()
        )

        # Renk haritası
        color_map = {"Yüksek": [214, 39, 40], "Orta": [255, 127, 14], "Hafif": [31, 119, 180]}
        cells["color"] = cells["tier"].apply(lambda t: color_map.get(t, [31, 119, 180]))

        # Tooltip için kısa tip listesi
        def _top_types(row, k=3):
            if not CRIME_TYPES:
                return ""
            items = []
            for t in CRIME_TYPES:
                v = float(row.get(t, 0.0))
                if v > 0:
                    items.append((t, v))
            items.sort(key=lambda x: x[1], reverse=True)
            return ", ".join([f"{t}:{v:.2f}" for t, v in items[:k]])

        cells["top_types"] = cells.apply(_top_types, axis=1)

        # Risk katmanı (nokta/centroid)
        if risk_layer_show:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=cells,
                get_position="[centroid_lon, centroid_lat]",
                get_fill_color="color",
                get_radius=60,
                pickable=True,
                stroked=False,
                opacity=0.35,
            ))

        # Kalıcı hotspot (ısı veya marker)
        if show_hotspot and perm_hotspot_show:
            weight_col = None
            if selected_type and selected_type in cells.columns:
                weight_col = selected_type
            elif "expected" in cells.columns:
                weight_col = "expected"

            if weight_col:
                hm = cells.rename(columns={"centroid_lat": "lat", "centroid_lon": "lon"})
                hm["weight"] = hm[weight_col].clip(lower=0)

                if perm_hotspot_mode == "markers":
                    # Üst %10 marker
                    thr = float(hm["weight"].quantile(0.90)) if not hm["weight"].empty else 0.0
                    strong = hm[hm["weight"] >= thr].copy()
                    if not strong.empty:
                        layers.append(pdk.Layer(
                            "ScatterplotLayer",
                            data=strong,
                            get_position="[lon, lat]",
                            get_fill_color="[139,0,0, 180]",
                            get_radius=80,
                            pickable=False,
                            stroked=False,
                            opacity=0.6,
                        ))
                else:
                    # Isı haritası
                    layers.append(pdk.Layer(
                        "HeatmapLayer",
                        data=hm[["lon", "lat", "weight"]],
                        get_position="[lon, lat]",
                        get_weight="weight",
                        radius_pixels=60,
                    ))

    # Geçici hotspot: olay noktaları ısı
    if show_temp_hotspot and temp_hotspot_show and isinstance(temp_hotspot_points, pd.DataFrame) and not temp_hotspot_points.empty:
        tmp = temp_hotspot_points.copy()
        # Kolon isimlerini normalize et
        if "lat" not in tmp.columns and "latitude" in tmp.columns:
            tmp = tmp.rename(columns={"latitude": "lat"})
        if "lon" not in tmp.columns and "longitude" in tmp.columns:
            tmp = tmp.rename(columns={"longitude": "lon"})
        if "weight" not in tmp.columns:
            tmp["weight"] = 1.0
        # Geçerli kolonlar var mı?
        if {"lon", "lat", "weight"}.issubset(tmp.columns):
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=tmp[["lon", "lat", "weight"]],
                get_position="[lon, lat]",
                get_weight="weight",
                radius_pixels=40,
            ))

    # (Opsiyonel) Devriye rotaları ileride burada çizilebilir
    # if patrol and "zones" in patrol: ...

    # Görünüm ve tooltip
    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=11.5)

    # Tooltip: hücre katmanı tıklanınca çalışır; veri alanları scatter layer'da mevcut
    tooltip = {
        "html": f"<b>{KEY_COL}</b>: {{{KEY_COL}}}<br/>"
                "Öncelik: {tier}<br/>"
                "E[olay]: {expected}<br/>"
                "<i>{top_types}</i>",
        "style": {"backgroundColor": "rgba(255,255,255,0.9)", "color": "black"}
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
        tooltip=tooltip,
    )
    return deck
