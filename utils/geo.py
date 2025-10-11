# utils/geo.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import json
import numpy as np
import pandas as pd

from utils.constants import KEY_COL

# ── Varsayılan harita başlangıcı (San Francisco) ─────────────────────────────
SF_CENTER: Tuple[float, float] = (37.7749, -122.4194)  # (lat, lon)
SF_ZOOM: int = 12

def join_neighborhood(df_agg: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    """df_agg + geo_df → neighborhood ekler (varsa). KEY_COL ile eşler."""
    if df_agg is None or df_agg.empty:
        return df_agg
    if "neighborhood" in df_agg.columns:
        return df_agg
    if geo_df is None or geo_df.empty or "neighborhood" not in geo_df.columns:
        return df_agg
    a = df_agg.copy()
    a[KEY_COL] = a[KEY_COL].astype(str)
    g = geo_df[[KEY_COL, "neighborhood"]].copy()
    g[KEY_COL] = g[KEY_COL].astype(str)
    return a.merge(g, on=KEY_COL, how="left")

def polygon_centroid(lonlat_loop: List[List[float]] | List[Tuple[float, float]]):
    """Basit poligon centroid (lon, lat) → (Cx, Cy). İlk/son nokta kapalıysa sonu atla."""
    x, y = zip(*lonlat_loop)
    A = Cx = Cy = 0.0
    # Kapanmış halkada son nokta == ilk nokta olur; -1'e kadar dönmek yeterli
    rng = range(len(lonlat_loop) - 1) if lonlat_loop[0] == lonlat_loop[-1] else range(len(lonlat_loop))
    for i in rng:
        j = (i + 1) % len(lonlat_loop)
        cross = x[i]*y[j] - x[j]*y[i]
        A  += cross
        Cx += (x[i] + x[j]) * cross
        Cy += (y[i] + y[j]) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        return float(sum(x)/len(x)), float(sum(y)/len(y))
    return float(Cx/(6*A)), float(Cy/(6*A))

def load_geoid_layer(path: str = "data/sf_cells.geojson", key_field: str = KEY_COL):
    """
    GeoJSON katmanını oku → (DataFrame, FeatureList).
    - DataFrame: [key_field, centroid_lon, centroid_lat]
    - Features:  properties.id garanti edilir (str)
    """
    p = Path(path)
    if not p.exists():
        # Boş dönüş: app/map güvenli çalışır
        return pd.DataFrame(columns=[key_field, "centroid_lon", "centroid_lat"]), []

    gj = json.loads(p.read_text(encoding="utf-8"))
    rows, feats_out = [], []

    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        # GEOID alanı olası isimlerden toplanır
        geoid = str(
            props.get(key_field)
            or props.get(key_field.upper())
            or props.get("GEOID")
            or props.get("geoid")
            or props.get("id")
            or ""
        ).strip()
        if not geoid:
            continue

        # Centroid varsa kullan; yoksa geometri'den hesapla
        lon = props.get("centroid_lon")
        lat = props.get("centroid_lat")
        if lon is None or lat is None:
            geom = feat.get("geometry", {}) or {}
            gtype = geom.get("type")
            if gtype == "Polygon":
                ring = geom.get("coordinates", [[]])[0]
            elif gtype == "MultiPolygon":
                ring = geom.get("coordinates", [[[]]])[0][0]
            else:
                # Desteklenmeyen geometri—atla
                continue
            lon, lat = polygon_centroid(ring)

        # Kayıt satırı
        rows.append({key_field: geoid, "centroid_lon": float(lon), "centroid_lat": float(lat)})

        # Feature properties.id garanti et + centroid'leri ekle (popup/tooltip için faydalı)
        props["id"] = geoid
        props.setdefault("centroid_lon", float(lon))
        props.setdefault("centroid_lat", float(lat))
        feat["properties"] = props
        feats_out.append(feat)

    df = pd.DataFrame(rows)
    return df, feats_out

def nearest_geoid(geo_df: pd.DataFrame, lat: float, lon: float) -> str | None:
    """Verilen (lat,lon) için centroid’e en yakın GEOID."""
    if geo_df is None or geo_df.empty:
        return None
    la = geo_df["centroid_lat"].to_numpy()
    lo = geo_df["centroid_lon"].to_numpy()
    d2 = (la - float(lat)) ** 2 + (lo - float(lon)) ** 2
    i = int(np.argmin(d2))
    return str(geo_df.iloc[i][KEY_COL])

def _extract_latlon_from_ret(ret) -> Tuple[float, float] | None:
    """st_folium dönüşünden güvenli lat/lon çıkarma."""
    if not ret:
        return None
    lc = ret.get("last_clicked")
    if lc is None:
        return None

    # Çeşitli olası şekiller:
    # 1) [lat, lon]
    if isinstance(lc, (list, tuple)) and len(lc) >= 2:
        return float(lc[0]), float(lc[1])

    # 2) {"lat":..., "lng":...} veya {"lat":..., "lon":...}
    if isinstance(lc, dict):
        if "lat" in lc and ("lng" in lc or "lon" in lc):
            return float(lc["lat"]), float(lc.get("lng", lc.get("lon")))
        # 3) {"latlng": {"lat":..., "lng":...}}
        ll = lc.get("latlng")
        if isinstance(ll, (list, tuple)) and len(ll) >= 2:
            return float(ll[0]), float(ll[1])
        if isinstance(ll, dict) and "lat" in ll and ("lng" in ll or "lon" in ll):
            return float(ll["lat"]), float(ll.get("lng", ll.get("lon")))
    return None

def resolve_clicked_gid(geo_df: pd.DataFrame, ret: dict) -> tuple[str | None, tuple[float, float] | None]:
    """
    st_folium ret sözlüğünden tıklanan GEOID'i ve (lat,lon)'u çıkart.
    Dönüş: (geoid | None, (lat, lon) | None)
    """
    gid, latlon = None, None
    obj = ret.get("last_object_clicked") if isinstance(ret, dict) else None

    if isinstance(obj, dict):
        props = obj.get("properties", {}) or obj.get("feature", {}).get("properties", {}) or {}
        gid = str(
            obj.get("id")
            or props.get("id")
            or props.get(KEY_COL)
            or props.get(KEY_COL.upper(), props.get("GEOID"))
            or ""
        ).strip() or None

        # Geometri/koordinat yakalamaya çalış
        geom = obj.get("geometry") or obj.get("feature", {}).get("geometry")
        if isinstance(geom, dict) and geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                latlon = (float(coords[1]), float(coords[0]))
        if not latlon:
            lat = obj.get("lat") or obj.get("latlng", {}).get("lat") or obj.get("location", {}).get("lat")
            lng = obj.get("lng") or obj.get("latlng", {}).get("lng") or obj.get("location", {}).get("lng") or obj.get("lon")
            if lat is not None and lng is not None:
                latlon = (float(lat), float(lng))

    if not gid:
        if not latlon:
            latlon = _extract_latlon_from_ret(ret)
        if latlon:
            gid = nearest_geoid(geo_df, latlon[0], latlon[1])

    return gid, latlon

# ── Harita başlangıç parametresi (lat, lon, zoom) ─────────────────────────────
def get_map_init(geo_df: pd.DataFrame | None = None) -> tuple[float, float, int]:
    """
    Harita ilk açılış merkezi ve zoom.
    - Geo veri varsa: centroid’lerin ortalaması
    - Yoksa: San Francisco (SF_CENTER) + SF_ZOOM
    """
    if isinstance(geo_df, pd.DataFrame) and not geo_df.empty:
        try:
            lat = float(geo_df["centroid_lat"].mean())
            lon = float(geo_df["centroid_lon"].mean())
            if np.isfinite(lat) and np.isfinite(lon):
                return lat, lon, SF_ZOOM
        except Exception:
            pass
    return SF_CENTER[0], SF_CENTER[1], SF_ZOOM
