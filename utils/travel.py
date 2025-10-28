# utils/travel.py
from __future__ import annotations
import os, math, time, json, hashlib
from typing import List, Tuple, Callable, Optional
import numpy as np
import requests

# =====================================================================
# 🚦 Yol Süresi Hesaplayıcı — Google (trafikli) + OSRM (ücretsiz) destekli
# =====================================================================

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


# =====================================================================
# 🔑 Anahtar / ortam değişkeni yönetimi
# =====================================================================
def _get_google_key() -> str:
    """API anahtarını güvenli biçimde getirir."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_MAPS_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_MAPS_API_KEY"]
    except Exception:
        pass
    return os.getenv("GOOGLE_MAPS_API_KEY", "")


# =====================================================================
# 🗺️ Sağlayıcılar
# =====================================================================
def _osrm_table(coords: List[Tuple[float, float]]) -> np.ndarray:
    """OSRM table endpoint (ücretsiz, trafiksiz)"""
    if not coords:
        return np.zeros((0, 0))
    url = "http://router.project-osrm.org/table/v1/driving/" + ";".join(
        [f"{lon},{lat}" for lat, lon in coords]
    )
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        js = r.json()
        m = np.array(js.get("durations", []), dtype=float)
        return np.nan_to_num(m / 60.0, nan=999.0)  # dakika cinsinden
    except Exception:
        return np.ones((len(coords), len(coords))) * 10.0  # fallback


def _google_matrix(coords: List[Tuple[float, float]], api_key: str) -> np.ndarray:
    """Google Distance Matrix API (canlı trafik, ücretli)"""
    if not coords or not api_key:
        return np.ones((len(coords), len(coords))) * 10.0
    n = len(coords)
    origins = "|".join([f"{lat},{lon}" for lat, lon in coords])
    destinations = origins
    url = (
        f"https://maps.googleapis.com/maps/api/distancematrix/json"
        f"?origins={origins}&destinations={destinations}"
        f"&departure_time=now&key={api_key}"
    )
    tag = hashlib.sha1(url.encode()).hexdigest()
    cached = _load_cache(tag)
    if cached:
        m = np.array(cached.get("durations", []), dtype=float)
        if m.shape == (n, n):
            return m
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        js = r.json()
        rows = js.get("rows", [])
        m = np.zeros((n, n))
        for i, row in enumerate(rows):
            elements = row.get("elements", [])
            for j, el in enumerate(elements):
                if el.get("status") == "OK":
                    dur = el["duration"].get("value", 0)
                    m[i, j] = dur / 60.0  # saniye→dakika
                else:
                    m[i, j] = 999.0
        _save_cache(tag, {"durations": m.tolist()})
        return m
    except Exception as e:
        print("Google API hata:", e)
        return np.ones((n, n)) * 10.0


# =====================================================================
# 🧠 Ana Fonksiyon
# =====================================================================
def build_travel_time_fn(coords: List[Tuple[float, float]], provider: str = "osrm") -> Callable[[Tuple[float, float], Tuple[float, float]], float]:
    """
    Sağlayıcıya göre yol süresi fonksiyonu döner.
    - provider='google' → canlı trafik (ücretli)
    - provider='osrm' → ücretsiz, trafiksiz
    Geri dönen fonksiyon: travel_time(origin, dest) → dakika
    """
    if not coords:
        return lambda a, b: 0.0

    provider = provider.lower().strip()
    tag = hashlib.sha1((provider + str(len(coords))).encode()).hexdigest()
    cached = _load_cache(tag)
    M = None

    if provider == "google":
        api_key = _get_google_key()
        if not api_key:
            raise ValueError("Google Maps API anahtarı bulunamadı. secrets.toml veya ortam değişkenine ekleyin.")
        M = _google_matrix(coords, api_key)
    else:
        M = _osrm_table(coords)

    if M is None or not isinstance(M, np.ndarray) or M.shape[0] != len(coords):
        raise RuntimeError("Rota matrisi oluşturulamadı.")

    # Hızlı erişim için (coord index map)
    idx_map = {tuple(coords[i]): i for i in range(len(coords))}

    def _travel_time(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Dakika cinsinden süre döner (yaklaşık)."""
        i, j = idx_map.get(tuple(a)), idx_map.get(tuple(b))
        if i is None or j is None:
            # haversine fallback
            lat1, lon1 = map(math.radians, a)
            lat2, lon2 = map(math.radians, b)
            d = 6371 * math.acos(
                math.sin(lat1) * math.sin(lat2)
                + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
            )
            return max(1.0, d / 0.5)  # 0.5 km/dk ~ 30 km/h varsayımı
        return float(M[i, j])

    return _travel_time
