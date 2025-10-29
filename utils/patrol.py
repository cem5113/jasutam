# utils/patrol.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from utils.constants import KEY_COL

# ---- Tier tanımı
TIER_ORDER = ["Çok Yüksek", "Yüksek", "Orta", "Düşük", "Çok Düşük"]
TIER_RANK  = {t: i for i, t in enumerate(TIER_ORDER)}  # 0 en yüksek öncelik

# Dengeli (balanced) mod için varsayılan kotalar (toplamı 1.0 olmalı)
TIER_QUOTA_DEFAULT: Dict[str, float] = {
    "Çok Yüksek": 0.30,
    "Yüksek":     0.25,
    "Orta":       0.20,
    "Düşük":      0.15,
    "Çok Düşük":  0.10,
}

# ---------------------------
# Yardımcı: Tier garantisi
# ---------------------------
def _ensure_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Tier yoksa expected'a göre 5'li kademe atar (qcut→cut fallback)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["expected"] = (
        pd.to_numeric(out.get("expected", 0), errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    if "tier" in out.columns:
        return out

    x = out["expected"].to_numpy()
    try:
        out["tier"] = pd.qcut(out["expected"], q=5, labels=TIER_ORDER, duplicates="drop").astype(str)
    except Exception:
        q = np.quantile(x, [0.20, 0.40, 0.60, 0.80]) if len(out) >= 5 else [0, 0, 0, 0]
        bins = np.concatenate(([-np.inf], q, [np.inf]))
        out["tier"] = pd.cut(out["expected"], bins=bins, labels=TIER_ORDER, include_lowest=True).astype(str)
    return out

# ---------------------------
# K-means benzeri kümeleme
# ---------------------------
def _init_centroids_farthest(coords: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Farthest-point başlatma (ilk merkez en ağır nokta)."""
    n = len(coords)
    if k == 0 or n == 0:
        return np.empty((0, 2))

    idx0 = int(np.argmax(weights))
    centers = [coords[idx0]]

    for _ in range(1, min(k, n)):
        # her noktanın mevcut merkezlere en yakın uzaklığı
        d = np.min(np.linalg.norm(coords[:, None, :] - np.array(centers)[None, :, :], axis=2), axis=1)
        # uzak ve ağır olanı tercih eden skor
        score = d * (1.0 + (weights / (weights.max() + 1e-9)))
        nxt = int(np.argmax(score))
        centers.append(coords[nxt])

    return np.array(centers)


def kmeans_like(
    coords: np.ndarray,
    weights: np.ndarray,
    k: int,
    iters: int = 20,
    init: str = "farthest",            # "farthest" | "topk"
    random_state: Optional[int] = None,
    jitter_scale: float = 0.0           # >0 verilirse varyant üretmek için küçük gürültü
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Basit, ağırlıklı k-means benzeri kümeleme.
    - init="farthest": merkezler yayılır (önerilir)
    - init="topk": en ağır k noktayla başlatır (eski davranışa yakın)
    """
    n = len(coords)
    k = min(k, n) if n > 0 else 0
    if k == 0:
        return np.empty((0, 2)), np.array([], dtype=int)

    rng = np.random.default_rng(random_state)

    X = coords.copy()
    if jitter_scale > 0:
        X = X + rng.normal(0.0, jitter_scale, size=X.shape)

    if init == "topk":
        idx_sorted = np.argsort(-weights)
        centroids = X[idx_sorted[:k]].copy()
    else:
        centroids = _init_centroids_farthest(X, weights, k, rng)

    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        for c in range(k):
            m = assign == c
            if not np.any(m):
                # boş küme: uzak+agir stratejisi
                d = np.min(np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2), axis=1)
                score = d * (1.0 + (weights / (weights.max() + 1e-9)))
                centroids[c] = X[int(np.argmax(score))]
            else:
                w = weights[m][:, None]
                centroids[c] = (X[m] * w).sum(axis=0) / max(1e-6, w.sum())

    return centroids, assign

# ---------------------------
# Dengeli seçim (kotalı)
# ---------------------------
def _select_with_tier_quota(
    sub: pd.DataFrame,
    cap_cells: int,
    cz_lonlat: np.ndarray,                 # (lon, lat)
    tier_quota: Dict[str, float],
    distance_push_alpha: float = 0.15
) -> pd.DataFrame:
    """
    Her tier için kota uygular; kota dolmazsa kalan kapasiteyi tier önceliğine göre doldurur.
    Aynı tier içinde 'expected' + merkezden uzaklığa küçük bir ağırlık vererek yayılımı artırır.
    """
    sub = sub.copy()

    # merkezden uzaklık (derece)
    dist = np.sqrt((sub["centroid_lon"] - cz_lonlat[0])**2 + (sub["centroid_lat"] - cz_lonlat[1])**2)
    dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-9)
    sub["score_mixed"] = sub["expected"] * (1.0 + distance_push_alpha * dist_norm)

    sub["tier_rank"] = sub["tier"].map(TIER_RANK).fillna(999).astype(int)
    # Önce tier (yüksek öncelik), sonra score_mixed (yayılım), sonra expected
    sub = sub.sort_values(["tier_rank", "score_mixed", "expected"], ascending=[True, False, False])

    picks: List[pd.DataFrame] = []
    used = set()

    # 1) kotaları uygula
    for tier, pct in tier_quota.items():
        need = int(np.floor(cap_cells * pct))
        cand = sub[(sub["tier"] == tier) & (~sub.index.isin(used))].head(need)
        picks.append(cand)
        used |= set(cand.index)

    # 2) kalan kapasite (tier önceliği korunur)
    taken = sum(len(p) for p in picks)
    if taken < cap_cells:
        rest = sub[~sub.index.isin(used)].head(cap_cells - taken)
        picks.append(rest)

    out = pd.concat(picks, ignore_index=False) if picks else sub.head(0)
    return out.head(cap_cells).copy()

# ---------------------------
# Ana: Tek plan üret
# ---------------------------
def allocate_patrols(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int = 6,
    travel_overhead: float = 0.40,
    strategy: str = "priority",                         # "priority" | "balanced"
    tier_quota: Optional[Dict[str, float]] = None,      # balanced için
    distance_push_alpha: float = 0.15,                  # balanced için
    init: str = "farthest",
    random_state: Optional[int] = None,
    jitter_scale: float = 0.0
) -> Dict:
    """
    Tek devriye planı döndürür.
    strategy="priority": saf risk önceliği (tier→expected)
    strategy="balanced": kotalı (tier_quota) + yayılım (distance_push_alpha)
    """
    # güvenlik ve tier
    if df_agg is None or df_agg.empty:
        return {"zones": []}

    df = _ensure_tier(df_agg.copy())
    df[KEY_COL] = df[KEY_COL].astype(str)

    cand = df[df["tier"].isin(TIER_ORDER)].copy()
    if cand.empty:
        return {"zones": []}

    # GEO birleşimi
    need_cols = [KEY_COL, "expected", "tier"]
    merged = cand[need_cols].merge(
        geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]].copy(),
        on=KEY_COL, how="inner"
    )
    merged["centroid_lat"] = pd.to_numeric(merged["centroid_lat"], errors="coerce")
    merged["centroid_lon"] = pd.to_numeric(merged["centroid_lon"], errors="coerce")
    merged = merged.dropna(subset=["centroid_lat", "centroid_lon"])
    if merged.empty:
        return {"zones": []}

    coords  = merged[["centroid_lon", "centroid_lat"]].to_numpy()
    weights = merged["expected"].to_numpy(dtype=float)

    # Küme sayısı
    k = max(1, min(int(k_planned), 50))
    cents, assign = kmeans_like(
        coords, weights, k,
        iters=20, init=init,
        random_state=random_state,
        jitter_scale=jitter_scale
    )

    # Kapasite
    cap_cells = max(1, int(duty_minutes / (cell_minutes * (1.0 + travel_overhead))))

    zones = []
    for z in range(len(cents)):
        m = assign == z
        if not np.any(m):
            continue
        sub = merged[m].copy()
        sub["tier_rank"] = sub["tier"].map(TIER_RANK).fillna(999).astype(int)

        cz = cents[z]  # (lon, lat)

        # Hücre seçimi
        if strategy == "balanced":
            tq = tier_quota or TIER_QUOTA_DEFAULT
            sub_planned = _select_with_tier_quota(
                sub=sub,
                cap_cells=cap_cells,
                cz_lonlat=cz,
                tier_quota=tq,
                distance_push_alpha=distance_push_alpha
            )
        else:
            # priority: saf risk öncelikli
            sub = sub.sort_values(["tier_rank", "expected"], ascending=[True, False])
            sub_planned = sub.head(cap_cells).copy()

        # Rota (açıya göre)
        angles = np.arctan2(sub_planned["centroid_lat"] - cz[1], sub_planned["centroid_lon"] - cz[0])
        sub_planned = sub_planned.assign(angle=angles).sort_values("angle")
        route = sub_planned[["centroid_lat", "centroid_lon"]].to_numpy().tolist()

        n_cells = len(sub_planned)
        eta_minutes = int(round(n_cells * cell_minutes * (1.0 + travel_overhead)))
        util = min(100, int(round(100 * eta_minutes / max(1, duty_minutes))))

        zones.append({
            "id": f"Z{z+1}",
            "centroid": {"lat": float(cz[1]), "lon": float(cz[0])},
            "cells": sub_planned[KEY_COL].astype(str).tolist(),
            "route": route,
            "expected_risk": float(sub_planned["expected"].mean() if n_cells else 0.0),
            "planned_cells": int(n_cells),
            "eta_minutes": int(eta_minutes),
            "utilization_pct": int(util),
            "capacity_cells": int(cap_cells),
        })

    return {"zones": zones}

# ---------------------------
# Çoklu öneri üreticileri
# ---------------------------
def _seed_grid(n: int, base: int = 42) -> List[int]:
    """Deterministik farklı tohumlar."""
    return [base + i * 131 for i in range(n)]

def suggest_multiple_plans(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    *,
    n_priority: int = 3,
    n_balanced: int = 6,
    cell_minutes: int = 6,
    travel_overhead: float = 0.40,
    tier_quota: Optional[Dict[str, float]] = None,
    distance_push_alpha: float = 0.15,
    init_priority: str = "topk",     # risk öncelikli varyantlarda çekirdeğe odaklı başlatma
    init_balanced: str = "farthest", # dengeli varyantlarda yayılım
    jitter_scale: float = 2e-4       # küçük gürültü ile farklı kümeler
) -> Dict[str, List[Dict]]:
    """
    Birden fazla plan önerir:
    - priority: n_priority adet
    - balanced: n_balanced adet
    """
    pri_plans: List[Dict] = []
    bal_plans: List[Dict] = []

    # Risk-öncelikli varyantlar
    for seed in _seed_grid(n_priority, base=101):
        pri_plans.append(
            allocate_patrols(
                df_agg=df_agg,
                geo_df=geo_df,
                k_planned=k_planned,
                duty_minutes=duty_minutes,
                cell_minutes=cell_minutes,
                travel_overhead=travel_overhead,
                strategy="priority",
                init=init_priority,
                random_state=seed,
                jitter_scale=jitter_scale
            )
        )

    # Dengeli (kotalı) varyantlar
    tq = tier_quota or TIER_QUOTA_DEFAULT
    # Küçük quota-jitter (toplamı 1.0 tutmak için normalize edeceğiz)
    tq_vec = np.array([tq[t] for t in TIER_ORDER], dtype=float)

    seeds = _seed_grid(n_balanced, base=707)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        # %5'e kadar hafif sars
        jitter = rng.normal(0, 0.02, size=len(TIER_ORDER))
        tq_j = np.clip(tq_vec + jitter, 0.0, None)
        tq_j = tq_j / max(1e-9, tq_j.sum())
        tq_dict = {t: float(v) for t, v in zip(TIER_ORDER, tq_j)}

        bal_plans.append(
            allocate_patrols(
                df_agg=df_agg,
                geo_df=geo_df,
                k_planned=k_planned,
                duty_minutes=duty_minutes,
                cell_minutes=cell_minutes,
                travel_overhead=travel_overhead,
                strategy="balanced",
                tier_quota=tq_dict,
                distance_push_alpha=distance_push_alpha,
                init=init_balanced,
                random_state=seed,
                jitter_scale=jitter_scale
            )
        )

    return {"priority": pri_plans, "balanced": bal_plans}

# ---------------------------
# Kolay sarmalayıcılar (UI'de kullanışlı)
# ---------------------------
# -----------------------------------------
# Geriye-uyumlu sarmalayıcılar (deprecated)
# -----------------------------------------
def make_balanced_plans(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    *,
    cell_minutes: int = 6,
    travel_overhead: float = 0.40,
    tier_quota: Optional[Dict[str, float]] = None,
    distance_push_alpha: float = 0.15,
    init: str = "farthest",
    random_state: Optional[int] = None,
    jitter_scale: float = 0.0,
    travel_time_fn=None,   # kabul et ama şimdilik kullanma
    **kwargs               # gelecekteki ek argümanları da yut
) -> Dict:
    """
    DEPRECATED: Eski API'yi desteklemek için.
    travel_time_fn verilirse şu an kullanılmıyor; ileride entegrasyon yapılabilir.
    """
    return allocate_patrols(
        df_agg=df_agg,
        geo_df=geo_df,
        k_planned=k_planned,
        duty_minutes=duty_minutes,
        cell_minutes=cell_minutes,
        travel_overhead=travel_overhead,
        strategy="balanced",
        tier_quota=tier_quota,
        distance_push_alpha=distance_push_alpha,
        init=init,
        random_state=random_state,
        jitter_scale=jitter_scale,
    )

def make_priority_plans(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    *,
    cell_minutes: int = 6,
    travel_overhead: float = 0.40,
    init: str = "topk",
    random_state: Optional[int] = None,
    jitter_scale: float = 0.0,
    **kwargs
) -> Dict:
    """
    DEPRECATED: Eski API’yi desteklemek için.
    """
    return allocate_patrols(
        df_agg=df_agg,
        geo_df=geo_df,
        k_planned=k_planned,
        duty_minutes=duty_minutes,
        cell_minutes=cell_minutes,
        travel_overhead=travel_overhead,
        strategy="priority",
        init=init,
        random_state=random_state,
        jitter_scale=jitter_scale,
    )

def suggest_dual_plans_single(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    **kwargs
) -> Dict[str, Dict]:
    """
    Tek ekranda iki plan: {"priority": {...}, "balanced": {...}}
    """
    p = allocate_patrols(df_agg, geo_df, k_planned, duty_minutes, strategy="priority", **kwargs)
    b = allocate_patrols(df_agg, geo_df, k_planned, duty_minutes, strategy="balanced", **kwargs)
    return {"priority": p, "balanced": b}
