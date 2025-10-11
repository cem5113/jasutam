# utils/hotspots.py
from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
import pandas as pd

# BallTree opsiyonel: yoksa numpy fallback
try:
    from sklearn.neighbors import BallTree
    _HAS_SK = True
except Exception:
    BallTree = None
    _HAS_SK = False


def _auto_cols(df: pd.DataFrame) -> Tuple[str | None, str | None, str | None]:
    """events DataFrame'inden (lat, lon, ts) kolon adlarını esnekçe yakala."""
    if df is None or df.empty:
        return None, None, None
    lower = {str(c).lower(): c for c in df.columns}
    lat = lower.get("latitude") or lower.get("lat")
    lon = lower.get("longitude") or lower.get("lon")
    ts  = lower.get("timestamp") or lower.get("ts")
    return lat, lon, ts


def _hour_mask(ts: pd.Series, hours_filter: Tuple[int, int]) -> pd.Series:
    """Saat filtresi (start,end). Gece saran aralıkları da destekler (örn. 22–4)."""
    h1, h2 = int(hours_filter[0]) % 24, int(hours_filter[1]) % 24
    hh = ts.dt.hour
    if h1 == h2:
        # tüm gün
        return pd.Series(True, index=ts.index)
    if h1 < h2:
        return (hh >= h1) & (hh < h2)
    # wrap-around: örn. 22–04
    return (hh >= h1) | (hh < h2)


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vektörize haversine; metre döndürür."""
    R = 6_371_000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat/2.0)**2 +
         np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2)
    return 2.0 * R * np.arcsin(np.sqrt(a))


def temp_hotspot_scores(
    events: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    lookback_h: int = 48,
    sigma_m: int = 500,
    half_life_h: int = 24,
    lat_col: str | None = "latitude",
    lon_col: str | None = "longitude",
    ts_col: str | None = "timestamp",
    key_col: str = "GEOID",
    type_col: str = "type",
    category: str | Iterable[str] | None = None,       # tek kategori veya liste
    hours_filter: tuple[int, int] | None = None        # (start,end) [start,end)
) -> pd.DataFrame:
    """
    Son 'lookback_h' saatlik olaylardan, GEOID centroid’lerine uzaysal (σ=sigma_m, Gaussian)
    ve zamansal (half-life) ağırlıklarla **geçici hotspot puanı** üretir.

    Döndürür: [key_col, hotspot_raw, hotspot_score] (score ∈ [0,1], p99 normalizasyonu)
    """
    out = geo_df[[key_col]].copy()
    out["hotspot_raw"] = 0.0
    out["hotspot_score"] = 0.0

    # Girdi kontrolleri
    if events is None or events.empty or geo_df is None or geo_df.empty:
        return out
    if ("centroid_lat" not in geo_df.columns) or ("centroid_lon" not in geo_df.columns):
        # Güvenli tarafta kal: centroid yoksa üretim yapmayalım
        return out

    # Kolon adları: otomatik bul (gelen parametreler yoksa)
    if lat_col is None or lon_col is None or ts_col is None:
        auto_lat, auto_lon, auto_ts = _auto_cols(events)
        lat_col = lat_col or auto_lat
        lon_col = lon_col or auto_lon
        ts_col  = ts_col  or auto_ts

    if not (lat_col and lon_col and ts_col and lat_col in events.columns and lon_col in events.columns and ts_col in events.columns):
        return out

    # Kopya + temizleme
    ev = events[[lat_col, lon_col, ts_col] + ([type_col] if type_col in events.columns else [])].copy()
    ev[ts_col]  = pd.to_datetime(ev[ts_col], utc=True, errors="coerce")
    ev[lat_col] = pd.to_numeric(ev[lat_col], errors="coerce")
    ev[lon_col] = pd.to_numeric(ev[lon_col], errors="coerce")
    ev = ev.dropna(subset=[lat_col, lon_col, ts_col])
    if ev.empty:
        return out

    # Kategori filtresi (tek string veya iterable)
    if category and str(category).lower() not in ("all", "tüm", "tum"):
        if type_col in ev.columns:
            if isinstance(category, (list, tuple, set, np.ndarray, pd.Series)):
                cats = {str(x).strip().lower() for x in category}
                ev = ev[ev[type_col].astype(str).str.lower().isin(cats)]
            else:
                ev = ev[ev[type_col].astype(str).str.lower() == str(category).strip().lower()]

    # Zaman filtresi: lookback + opsiyonel saat bandı
    now = pd.Timestamp.utcnow()
    ev = ev[(now - ev[ts_col]) <= pd.Timedelta(hours=int(lookback_h))]
    if hours_filter:
        ev = ev[_hour_mask(ev[ts_col], hours_filter)]
    if ev.empty:
        return out

    # Zamansal ağırlık (half-life)
    dt_h = (now - ev[ts_col]).dt.total_seconds() / 3600.0
    w_t = np.power(2.0, -dt_h / float(max(half_life_h, 1e-6)))

    # GEO centroid’leri
    glat = pd.to_numeric(geo_df["centroid_lat"], errors="coerce").to_numpy()
    glon = pd.to_numeric(geo_df["centroid_lon"], errors="coerce").to_numpy()

    # Uzaysal yakınlık ve skor
    hotspot_raw = np.zeros(len(geo_df), dtype=float)

    if _HAS_SK:
        # BallTree (haversine)
        R = 6_371_000.0
        ev_rad = np.radians(ev[[lat_col, lon_col]].to_numpy())
        geo_rad = np.radians(np.c_[glat, glon])
        tree = BallTree(ev_rad, metric="haversine")
        query_r = 3.0 * (float(sigma_m) / R)  # 3σ yarıçap (radyan)

        ind = tree.query_radius(geo_rad, r=query_r, return_distance=True)
        for gi, (idxs, dists_rad) in enumerate(zip(*ind)):
            if len(idxs) == 0:
                continue
            d_m = dists_rad * R
            w_s = np.exp(-0.5 * (d_m / float(sigma_m)) ** 2)
            hotspot_raw[gi] = float((w_s * w_t.iloc[idxs].to_numpy()).sum())
    else:
        # Fallback: numpy ile brute-force (küçük veri için yeterli)
        ev_lat = ev[lat_col].to_numpy()
        ev_lon = ev[lon_col].to_numpy()
        for gi in range(len(geo_df)):
            d_m = _haversine_m(glat[gi], glon[gi], ev_lat, ev_lon)
            mask = d_m <= 3.0 * float(sigma_m)  # 3σ
            if not mask.any():
                continue
            w_s = np.exp(-0.5 * (d_m[mask] / float(sigma_m)) ** 2)
            hotspot_raw[gi] = float((w_s * w_t.to_numpy()[mask]).sum())

    # p99 normalizasyonu
    p99 = float(np.percentile(hotspot_raw, 99)) if hotspot_raw.max() > 0 else 1.0
    out["hotspot_raw"] = hotspot_raw
    out["hotspot_score"] = np.clip(hotspot_raw / (p99 + 1e-12), 0.0, 1.0)
    return out
