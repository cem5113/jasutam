# utils/patrol_planner.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Dict, List
from functools import partial
import numpy as np
import pandas as pd

# ── constants (güvenli import; fallback)
try:
    from utils.constants import KEY_COL  # örn. "GEOID"
except Exception:
    KEY_COL = "GEOID"

# Varsayılan tier dağılımı (toplam 1.0): 30/25/20/15/10
DEFAULT_DIST: Dict[str, float] = {
    "Çok Yüksek": 0.30,
    "Yüksek":     0.25,
    "Orta":       0.20,
    "Düşük":      0.15,
    "Çok Düşük":  0.10,
}

# ───────────────────────────── helpers ─────────────────────────────
def _ensure_keycol(df: pd.DataFrame, key_col: str = KEY_COL) -> pd.DataFrame:
    """KEY_COL yoksa olası varyantlardan eşler; stringe çevirir."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if key_col not in out.columns:
        alts = {key_col.upper(), key_col.lower(), "GEOID", "geoid", "GeoID"}
        hit = next((c for c in out.columns if c in alts), None)
        if hit:
            out = out.rename(columns={hit: key_col})
    if key_col in out.columns:
        out[key_col] = out[key_col].astype(str).str.strip()
    return out


def _ensure_expected_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """expected sütununu sayısal hale getirir, NaN/inf temizlenir."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "expected" in out.columns:
        x = pd.to_numeric(out["expected"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["expected"] = x.clip(lower=0.0)
    return out


# ───────────────────────────── tier/kademe ataması ─────────────────────────────
def assign_tier_safe(df: pd.DataFrame) -> pd.DataFrame:
    """expected'a göre 5-kademe 'tier' sütunu atar."""
    if df is None or df.empty or "expected" not in df.columns:
        return df
    out = _ensure_expected_numeric(df)
    x = out["expected"].to_numpy(dtype=float)
    labels5 = ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek"]

    if np.unique(x[~np.isnan(x)]).size < 5 or np.count_nonzero(~np.isnan(x)) < 5:
        out["tier"] = "Çok Düşük"
        return out

    try:
        out["tier"] = pd.qcut(out["expected"], q=5, labels=labels5, duplicates="drop").astype(str)
        if out["tier"].isna().all():
            raise ValueError
        return out
    except Exception:
        q = np.quantile(x, [0.20, 0.40, 0.60, 0.80]).astype(float)
        eps = max(1e-9, 1e-6 * (float(np.nanmax(x)) - float(np.nanmin(x))))
        for i in range(1, len(q)):
            if q[i] <= q[i - 1]:
                q[i] = q[i - 1] + eps
        bins = np.concatenate(([-np.inf], q, [np.inf]))
        out["tier"] = pd.cut(out["expected"], bins=bins, labels=labels5, include_lowest=True).astype(str)
        return out


# ───────────────────────────── geçmiş plan kayıtları ─────────────────────────────
def load_saved_plans(path: str = "data/patrol_logs.json") -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_plan(plan: dict, meta: dict | None = None, path: str = "data/patrol_logs.json") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    logs = load_saved_plans(path)
    logs.append({"ts": int(time.time()), "meta": meta or {}, "plan": plan or {}})
    p.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")


def recent_patrolled_geoids(logs: List[dict], lookback_hours: int = 24) -> set[str]:
    cutoff = int(time.time()) - int(lookback_hours) * 3600
    gids: set[str] = set()
    for item in logs:
        try:
            if int(item.get("ts", 0)) < cutoff:
                continue
            zones = (item.get("plan") or {}).get("zones", [])
            for z in zones:
                for g in z.get("cells", []):
                    gids.add(str(g))
        except Exception:
            continue
    return gids


def apply_fatigue_penalty(df: pd.DataFrame, penalized_geoids: Iterable[str], alpha: float = 0.25, key_col: str = KEY_COL) -> pd.DataFrame:
    """Son X saatte devriye yapılmış hücrelerin expected değerini düşürür."""
    if df is None or df.empty:
        return df
    out = _ensure_keycol(_ensure_expected_numeric(df), key_col)
    if key_col not in out.columns or "expected" not in out.columns:
        return out
    penalized = set(map(str, penalized_geoids or []))
    if not penalized:
        return out
    mask = out[key_col].astype(str).isin(penalized)
    out.loc[mask, "expected"] = out.loc[mask, "expected"] * (1.0 - float(alpha))
    return out


# ───────────────────────────── ana plan üretimi ─────────────────────────────
def propose_patrol_plans(
    base_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int,
    allocate_fn: Callable[..., dict],
    *,
    key_col: str = KEY_COL,
    n_plans: int = 5,
    dist: Dict[str, float] = DEFAULT_DIST,
    logs_path: str = "data/patrol_logs.json",
    fatigue_hours: int = 24,
    fatigue_alpha: float = 0.25,
    diversify: bool = True,
    diversify_top_n: int = 60,
    diversify_gamma: float = 0.35,
    travel_overhead: float | None = None,
    travel_time_fn=None,  # ✅ Google/OSRM rota süresi
    seed: int | None = None,
) -> List[dict]:
    """Çoklu devriye planı üretir (yorgunluk, çeşitlilik ve rota süresi destekli)."""
    if seed is not None:
        np.random.seed(int(seed))
    base_df = _ensure_keycol(_ensure_expected_numeric(base_df), key_col)
    geo_df = _ensure_keycol(geo_df, key_col)

    logs = load_saved_plans(logs_path)
    penalized = recent_patrolled_geoids(logs, lookback_hours=fatigue_hours)

    plans: List[dict] = []
    already_top: set[str] = set()

    for i in range(int(n_plans)):
        df_i = apply_fatigue_penalty(base_df, penalized_geoids=penalized, alpha=fatigue_alpha, key_col=key_col)
        df_i = assign_tier_safe(df_i)
        df_i["expected"] = df_i["expected"] * (1.0 + np.random.uniform(-0.05, 0.05, len(df_i)))

        if diversify and already_top:
            mask = df_i[key_col].astype(str).isin(already_top)
            df_i.loc[mask, "expected"] = df_i.loc[mask, "expected"] * (1.0 - diversify_gamma)

        kwargs = dict(
            df_agg=df_i,
            geo_df=geo_df,
            k_planned=int(k_planned),
            duty_minutes=int(duty_minutes),
            cell_minutes=int(cell_minutes),
            travel_time_fn=travel_time_fn,  # ✅ rota fonksiyonu eklendi
        )
        if travel_overhead is not None:
            kwargs["travel_overhead"] = float(travel_overhead)

        plan = allocate_fn(**kwargs)

        if isinstance(plan, dict):
            plan.setdefault("meta", {})
            plan["meta"].update({
                "planner": "patrol_planner.py",
                "plan_no": i + 1,
                "tier_distribution": dist,
                "fatigue_hours": fatigue_hours,
                "diversify": diversify,
            })
            cells = []
            for z in plan.get("zones", []):
                cells.extend(z.get("cells", []))
            already_top |= set(map(str, cells))

        plans.append(plan)
    return plans


# ───────────────────────────── kısa yol sarmalayıcılar ─────────────────────────────
def make_priority_plans(
    base_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int = 6,
    *,
    n_plans: int = 3,
    travel_overhead: float = 0.40,
    seed: Optional[int] = 101,
    travel_time_fn=None,
) -> List[dict]:
    """Risk-öncelikli çoklu plan (Google rota destekli)."""
    from utils.patrol import allocate_patrols as _alloc
    allocate_priority = partial(_alloc, strategy="priority", init="topk")
    return propose_patrol_plans(
        base_df=base_df,
        geo_df=geo_df,
        k_planned=k_planned,
        duty_minutes=duty_minutes,
        cell_minutes=cell_minutes,
        allocate_fn=allocate_priority,
        n_plans=n_plans,
        travel_overhead=travel_overhead,
        travel_time_fn=travel_time_fn,  # ✅
        seed=seed,
        dist={"Çok Yüksek": 1.0, "Yüksek": 0.7, "Orta": 0.4, "Düşük": 0.2, "Çok Düşük": 0.1},
    )


def make_balanced_plans(
    base_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int = 6,
    *,
    n_plans: int = 6,
    travel_overhead: float = 0.40,
    tier_quota: Optional[Dict[str, float]] = None,
    distance_push_alpha: float = 0.15,
    seed: Optional[int] = 707,
    travel_time_fn=None,  # ✅ EKLENDİ
) -> List[dict]:
    """Kotalı (balanced) çoklu plan üretir; rota süresiyle entegre."""
    from utils.patrol import allocate_patrols as _alloc
    if tier_quota is None:
        tier_quota = DEFAULT_DIST

    allocate_balanced = partial(
        _alloc,
        strategy="balanced",
        tier_quota=tier_quota,
        distance_push_alpha=distance_push_alpha,
        init="farthest",
        jitter_scale=2e-4,
    )

    return propose_patrol_plans(
        base_df=base_df,
        geo_df=geo_df,
        k_planned=k_planned,
        duty_minutes=duty_minutes,
        cell_minutes=cell_minutes,
        allocate_fn=allocate_balanced,
        n_plans=n_plans,
        travel_overhead=travel_overhead,
        travel_time_fn=travel_time_fn,  # ✅ rota fonksiyonu aktarılıyor
        seed=seed,
        dist=DEFAULT_DIST,
        diversify=True,
        diversify_top_n=80,
        diversify_gamma=0.35,
    )
