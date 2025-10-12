# utils/forecast.py
from __future__ import annotations
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Iterable

import numpy as np
import pandas as pd

from utils.constants import CRIME_TYPES, KEY_COL, CATEGORY_TO_KEYS, SF_TZ_OFFSET

# ---- küçük yardımcılar ----
def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

# -------------------- Baz yoğunluk: normalize --------------------
def precompute_base_intensity(geo_df: pd.DataFrame) -> np.ndarray:
    lon = pd.to_numeric(geo_df["centroid_lon"], errors="coerce").to_numpy()
    lat = pd.to_numeric(geo_df["centroid_lat"], errors="coerce").to_numpy()

    # 2 sentetik tepe + küçük zemin (veri yoksa fallback yüzeyi)
    peak1 = np.exp(-(((lon + 122.41) ** 2) / 0.0008 + ((lat - 37.78) ** 2) / 0.0005))
    peak2 = np.exp(-(((lon + 122.42) ** 2) / 0.0006 + ((lat - 37.76) ** 2) / 0.0006))
    raw = 0.2 + 0.8 * (peak1 + peak2) + 0.07

    raw = raw - np.nanmin(raw)
    denom = np.nanmax(raw) + 1e-9
    return (raw / denom).astype(float)

# -------------------- Near-Repeat (NR) skor vektörü --------------------
def _near_repeat_score(
    geo_df: pd.DataFrame,
    events: pd.DataFrame,
    start_iso: str,
    *,
    lookback_h: int = 24,
    spatial_radius_m: int = 400,
    temporal_decay_h: float = 12.0,
) -> np.ndarray:
    if events is None or len(events) == 0:
        return np.zeros(len(geo_df), dtype=float)

    # kolon adlarını esnek yakala
    cols = {c.lower(): c for c in events.columns}
    lat_col = cols.get("lat") or cols.get("latitude")
    lon_col = cols.get("lon") or cols.get("longitude")
    ts_col  = cols.get("ts")  or cols.get("timestamp")
    if not (lat_col and lon_col and ts_col):
        return np.zeros(len(geo_df), dtype=float)

    ev = events[[ts_col, lat_col, lon_col]].copy()
    ev[ts_col]  = pd.to_datetime(ev[ts_col], utc=True, errors="coerce")
    ev[lat_col] = pd.to_numeric(ev[lat_col], errors="coerce")
    ev[lon_col] = pd.to_numeric(ev[lon_col], errors="coerce")
    ev = ev.dropna()

    start = datetime.fromisoformat(start_iso)
    t0 = start - timedelta(hours=int(lookback_h))
    ev = ev[(ev[ts_col] >= t0) & (ev[ts_col] <= start)]
    if ev.empty:
        return np.zeros(len(geo_df), dtype=float)

    cent_lat = pd.to_numeric(geo_df["centroid_lat"], errors="coerce").to_numpy()
    cent_lon = pd.to_numeric(geo_df["centroid_lon"], errors="coerce").to_numpy()

    nr = np.zeros(len(geo_df), dtype=float)
    tau = max(float(temporal_decay_h), 1e-6)

    for _, r in ev.iterrows():
        d = _haversine_m(cent_lat, cent_lon, float(r[lat_col]), float(r[lon_col]))
        w_space = np.exp(-np.maximum(d, 0.0) / float(spatial_radius_m))
        dt_h = (start - r[ts_col].to_pydatetime()).total_seconds() / 3600.0
        w_time = np.exp(-max(dt_h, 0.0) / tau)
        nr += w_space * w_time

    nr = nr - nr.min()
    maxv = nr.max()
    return (nr / maxv) if maxv > 1e-9 else np.zeros_like(nr)

# -------------------- Events'ten (24,7) zaman ağırlıkları --------------------
def _build_time_weights_from_events(events: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """
    events -> (hour_of_day weights [24], day_of_week weights [7]).
    UTC ts'leri SF yereline çevirir. Veri yoksa makul fallback döner.
    """
    # Fallback: sinüs bazlı 24 saat profili ve hafif hafta etkisi
    h = np.arange(24, dtype=float)
    diurnal24 = 1.0 + 0.4 * np.sin(((h - 18) / 24.0) * 2.0 * np.pi)
    diurnal24 = np.clip(diurnal24, 1e-6, None)
    diurnal24 = diurnal24 / diurnal24.sum()

    weekly7 = np.array([0.95, 1.00, 1.00, 1.00, 1.05, 1.10, 1.10], dtype=float)
    weekly7 = weekly7 / weekly7.sum()

    if not isinstance(events, pd.DataFrame) or events.empty:
        return diurnal24, weekly7

    cols = {c.lower(): c for c in events.columns}
    ts_col = cols.get("ts") or cols.get("timestamp")
    if not ts_col:
        return diurnal24, weekly7

    tmp = events[[ts_col]].copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], utc=True, errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return diurnal24, weekly7

    # UTC → SF
    tmp["ts_sf"] = tmp[ts_col] + pd.Timedelta(hours=SF_TZ_OFFSET)

    # Saat (0..23)
    hours_cnt = tmp["ts_sf"].dt.hour.value_counts().reindex(range(24), fill_value=0).astype(float)
    if hours_cnt.sum() > 0:
        diurnal24 = (hours_cnt / hours_cnt.sum()).to_numpy()

    # Gün (Mon=0..Sun=6)
    dow_cnt = tmp["ts_sf"].dt.dayofweek.value_counts().reindex(range(7), fill_value=0).astype(float)
    if dow_cnt.sum() > 0:
        weekly7 = (dow_cnt / dow_cnt.sum()).to_numpy()

    return diurnal24, weekly7

# -------------------- Hızlı agregasyon (NR + filtre + 5'li tier) --------------------
def aggregate_fast(
    start_iso: str,
    horizon_h: int,
    geo_df: pd.DataFrame,
    base_int: np.ndarray,
    *,
    k_lambda: float = 0.12,
    events: pd.DataFrame | None = None,
    near_repeat_alpha: float = 0.35,
    nr_lookback_h: int = 24,
    nr_radius_m: int = 400,
    nr_decay_h: float = 12.0,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Not: Toplam beklenenin ufukla (H) doğru orantılı artması için saat ağırlıkları
    önce 1'e normalize edilir, sonra H ile ÇARPILIR.
    """
    # --- 1) Zaman ağırlıkları (sadece diurnal) ---
    start = datetime.fromisoformat(start_iso)
    H = max(int(horizon_h), 1)
    hours = np.arange(H)
    # gün içi dalga (tepe ≈ 18:00)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24.0) * 2.0 * np.pi)
    # normalize et, sonra H ile ölçekle → toplam ağırlık = H
    w = diurnal / (diurnal.sum() + 1e-12)
    w = H * w  # <<< kritik: ufuk çarpanı

    # --- 2) Near-repeat skoru ---
    nr = _near_repeat_score(
        geo_df,
        events,
        start_iso,
        lookback_h=nr_lookback_h,
        spatial_radius_m=nr_radius_m,
        temporal_decay_h=nr_decay_h,
    )

    # --- 3) Saatlik lambda ve kırpma ---
    base_per_hour = k_lambda * base_int  # hücre başına saatlik baz oran
    lam_hour = base_per_hour[:, None] * w[None, :]
    lam_hour *= (1.0 + near_repeat_alpha * nr[:, None])  # NR etkisi
    lam_hour = np.clip(lam_hour, 0.0, 0.9)

    # --- 4) Özetler ---
    expected = lam_hour.sum(axis=1)  # ufuk toplamı
    p_hour = 1.0 - np.exp(-lam_hour)
    q10 = np.quantile(p_hour, 0.10, axis=1)
    q90 = np.quantile(p_hour, 0.90, axis=1)

    # --- 5) Tür dağılımı ---
    rng = np.random.default_rng(42)
    if CRIME_TYPES and len(CRIME_TYPES) >= 1:
        alpha = np.full(len(CRIME_TYPES), 1.2)
        W = rng.dirichlet(alpha, size=len(geo_df))
        type_cols = {t: expected * W[:, i] for i, t in enumerate(CRIME_TYPES)}
    else:
        alpha = np.array([1.5, 1.2, 2.0, 1.0, 1.3])
        W = rng.dirichlet(alpha, size=len(geo_df))
        base_types = ["assault", "burglary", "theft", "robbery", "vandalism"]
        type_cols = {t: expected * W[:, i] for i, t in enumerate(base_types)}

    out = pd.DataFrame({
        KEY_COL: geo_df[KEY_COL].astype(str).to_numpy(),
        "expected": expected.astype(float),
        "q10": q10.astype(float),
        "q90": q90.astype(float),
        "nr_boost": nr.astype(float),
        **type_cols,
    })

    # --- 6) Kategori filtresi istenmişse expected'i yeniden hesapla ---
    if filters:
        cats: Optional[Iterable[str]] = filters.get("cats")
        if cats:
            wanted_keys: list[str] = []
            for c in cats:
                wanted_keys += CATEGORY_TO_KEYS.get(c, [])
            wanted_cols = [c for c in wanted_keys if c in out.columns]
            if wanted_cols:
                out["expected"] = out[wanted_cols].sum(axis=1)

    # --- 7) 5 kademeli tier ---
    if out["expected"].gt(0).any():
        q95 = float(out["expected"].quantile(0.95))
        q75 = float(out["expected"].quantile(0.75))
        q50 = float(out["expected"].quantile(0.50))
        q25 = float(out["expected"].quantile(0.25))
    else:
        q95 = q75 = q50 = q25 = 0.0

    out["tier"] = np.select(
        [
            out["expected"] >= q95,
            out["expected"] >= q75,
            out["expected"] >= q50,
            out["expected"] >= q25,
        ],
        ["Çok Yüksek", "Yüksek", "Orta", "Düşük"],
        default="Çok Düşük",
    )

    return out

# -------------------- Poisson yardımcıları --------------------
def p_to_lambda_array(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, 0.999999)
    return -np.log1p(-p)

def p_to_lambda(p):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 0.999999)
    return -np.log(1.0 - p)

def pois_cdf(k: int, lam: float) -> float:
    s = 0.0
    for i in range(k + 1):
        s += (lam ** i) / math.factorial(i)
    return math.exp(-lam) * s

def prob_ge_k(lam: float, k: int) -> float:
    return 1.0 - pois_cdf(k - 1, lam)

def pois_quantile(lam: float, q: float) -> int:
    k = 0
    while pois_cdf(k, lam) < q and k < 10_000:
        k += 1
    return k

def pois_pi90(lam: float) -> tuple[int, int]:
    lo = pois_quantile(lam, 0.05)
    hi = pois_quantile(lam, 0.95)
    return lo, hi
