# utils/patrol.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from utils.constants import KEY_COL

TIER_ORDER = ["Çok Yüksek", "Yüksek", "Orta", "Düşük", "Çok Düşük"]
TIER_RANK = {t:i for i,t in enumerate(TIER_ORDER)}  # 0 en yüksek öncelik

def _ensure_tier(df: pd.DataFrame) -> pd.DataFrame:
    """tier yoksa expected'a göre 5'li kademe atar (quantile→fallback)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["expected"] = pd.to_numeric(out.get("expected", 0), errors="coerce").fillna(0).clip(lower=0)
    if "tier" in out.columns:
        return out
    x = out["expected"].to_numpy()
    labels = TIER_ORDER[::-1]  # düşükten yükseğe giden sıralama verirsek, bins ile uyumlu olur
    try:
        # qcut dene
        out["tier"] = pd.qcut(out["expected"], q=5, labels=TIER_ORDER, duplicates="drop").astype(str)
    except Exception:
        # manuel eşik
        q = np.quantile(x, [0.20, 0.40, 0.60, 0.80]) if len(out) >= 5 else [0,0,0,0]
        bins = np.concatenate(([-np.inf], q, [np.inf]))
        out["tier"] = pd.cut(out["expected"], bins=bins, labels=TIER_ORDER, include_lowest=True).astype(str)
    return out

def kmeans_like(coords: np.ndarray, weights: np.ndarray, k: int, iters: int = 20):
    n = len(coords); k = min(k, n) if n > 0 else 0
    if k == 0:
        return np.empty((0,2)), np.array([], dtype=int)
    idx_sorted = np.argsort(-weights)
    centroids = coords[idx_sorted[:k]].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        for c in range(k):
            m = assign == c
            if not np.any(m):
                centroids[c] = coords[idx_sorted[0]]
            else:
                w = weights[m][:, None]
                centroids[c] = (coords[m] * w).sum(axis=0) / max(1e-6, w.sum())
    return centroids, assign

def allocate_patrols(df_agg: pd.DataFrame, geo_df: pd.DataFrame,
                     k_planned: int, duty_minutes: int,
                     cell_minutes: int = 6, travel_overhead: float = 0.40) -> Dict:
    # --- güvenlik: kolonlar & tier
    if df_agg is None or df_agg.empty:
        return {"zones": []}
    df = _ensure_tier(df_agg.copy())
    df[KEY_COL] = df[KEY_COL].astype(str)

    cand = df[df["tier"].isin(["Çok Yüksek", "Yüksek", "Orta",  "Düşük", "Çok Düşük"])].copy()
    if cand.empty:
        return {"zones": []}

    # GEO ile birleşim (centroid zorunlu)
    need_cols = [KEY_COL, "expected", "tier"]
    merged = (cand[need_cols]
              .merge(geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]].copy(), on=KEY_COL, how="inner"))
    merged["centroid_lat"] = pd.to_numeric(merged["centroid_lat"], errors="coerce")
    merged["centroid_lon"] = pd.to_numeric(merged["centroid_lon"], errors="coerce")
    merged = merged.dropna(subset=["centroid_lat", "centroid_lon"])
    if merged.empty:
        return {"zones": []}

    coords  = merged[["centroid_lon", "centroid_lat"]].to_numpy()
    weights = merged["expected"].to_numpy(dtype=float)

    k = max(1, min(int(k_planned), 50))
    cents, assign = kmeans_like(coords, weights, k)

    # Kapasite (hücre sayısı)
    cap_cells = max(1, int(duty_minutes / (cell_minutes * (1.0 + travel_overhead))))

    zones = []
    for z in range(len(cents)):
        m = assign == z
        if not np.any(m):
            continue
        sub = merged[m].copy()

        # Hücre seçimi: önce kademe önceliği, sonra expected
        sub["tier_rank"] = sub["tier"].map(TIER_RANK).fillna(999).astype(int)
        sub = sub.sort_values(["tier_rank", "expected"], ascending=[True, False])
        sub_planned = sub.head(cap_cells).copy()

        # Rota sıralaması (açıya göre)
        cz = cents[z]
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
