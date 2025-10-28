# utils/travel.py
from __future__ import annotations
import os, math, time, json, hashlib
from typing import List, Tuple, Callable, Optional

import numpy as np
import requests

# ---- Basit file-cache
_CACHE_DIR = os.environ.get("TRAVEL_CACHE_DIR", "data/_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

def _cache_path(tag: str) -> str:
    return os.path.join(_CACHE_DIR, f"{tag}.json")

def _load_cache(tag: str) -> Optional[dict]:
    p = _cache_path(tag)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_cache(tag: str, data: dict) -> None:
    try:
        with open(_cache_path(tag), "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass

# ---- Haversine (fallback, dakika tahmini)
def _haversine_minutes(a: Tuple[float,float], b: Tuple[float,float], avg_speed_kmh=25.0) -> float:
    # kaba fallback: 25 km/s şehir içi
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    d = 2 * R * math.asin(math.sqrt(h))  # km
    minutes = (d / max(1e-9, avg_speed_kmh)) * 60.0
    # ışıklar/kavşak vs. için küçük bir katsayı
    return minutes * 1.2

# ---- Google Distance Matrix
def _google_matrix(coords: List[Tuple[float,float]]) -> np.ndarray:
    import googlemaps  # pip install googlemaps

    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY yok.")

    gmaps = googlemaps.Client(key=key)
    n = len(coords)
    M = np.zeros((n, n), dtype=float)

    # Google limitleri: element sayısı, batch…
    # Basit: tüm i,j’yi parça parça dolduralım
    max_origins = 25  # güvenli batch
    max_dest    = 25
    for i0 in range(0, n, max_origins):
        origins = coords[i0:i0+max_origins]
        for j0 in range(0, n, max_dest):
            dests = coords[j0:j0+max_dest]
            # API çağrısı
            resp = gmaps.distance_matrix(
                origins=origins,
                destinations=dests,
                mode="driving",
                departure_time="now",            # trafik için önemli
                traffic_model="best_guess"       # best_guess / optimistic / pessimistic
            )
            rows = resp.get("rows", [])
            for oi, row in enumerate(rows):
                elements = row.get("elements", [])
                for dj, el in enumerate(elements):
                    # saniye → dakika
                    if el.get("status") == "OK":
                        dur = el.get("duration_in_traffic") or el.get("duration")
                        sec = dur.get("value", 0)
                        M[i0+oi, j0+dj] = max(0.0, float(sec) / 60.0)
                    else:
                        # fallback: haversine
                        M[i0+oi, j0+dj] = _haversine_minutes(origins[oi], dests[dj])
            # nazik hız
            time.sleep(0.1)
    return M

# ---- OSRM Table (trafiksiz)
def _osrm_matrix(coords: List[Tuple[float,float]], base_url="http://localhost:5000") -> np.ndarray:
    n = len(coords)
    M = np.zeros((n, n), dtype=float)
    # OSRM /table v1/driving/{lon,lat;...}?annotations=duration
    # Not: OSRM koordinat sırası lon,lat
    pts = ";".join([f"{c[1]:.6f},{c[0]:.6f}" for c in coords])
    url = f"{base_url}/table/v1/driving/{pts}?annotations=duration"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    D = js.get("durations") or []
    for i in range(n):
        for j in range(n):
            sec = D[i][j] if D[i][j] is not None else 0.0
            M[i, j] = max(0.0, float(sec) / 60.0)
    return M

def build_travel_time_matrix(coords: List[Tuple[float,float]], provider="google") -> np.ndarray:
    # cache anahtarı
    h = hashlib.sha1()
    h.update(provider.encode("utf-8"))
    for (lat, lon) in coords:
        h.update(f"{lat:.6f},{lon:.6f};".encode("utf-8"))
    tag = f"ttm_{provider}_{h.hexdigest()}"
    cached = _load_cache(tag)
    if cached and "M" in cached:
        return np.array(cached["M"], dtype=float)

    try:
        if provider == "google":
            M = _google_matrix(coords)
        elif provider == "osrm":
            M = _osrm_matrix(coords)
        else:
            raise ValueError("Bilinmeyen provider")
    except Exception:
        # tam fallback: haversine
        n = len(coords)
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                M[i, j] = 0.0 if i == j else _haversine_minutes(coords[i], coords[j])

    _save_cache(tag, {"M": M.tolist()})
    return M

def build_travel_time_fn(coords: List[Tuple[float,float]], provider="google") -> Callable[[int,int], float]:
    M = build_travel_time_matrix(coords, provider=provider)
    def _fn(i: int, j: int) -> float:
        return float(M[i, j])
    return _fn
