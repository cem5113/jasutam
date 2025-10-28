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
    """KEY_COL yoksa muhtemel varyantlardan eşler; stringe çevirir."""
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
    """expected sütununu sayısal, NaN/inf temizlenmiş ve 0 altı kırpılmış tutar."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "expected" in out.columns:
        x = pd.to_numeric(out["expected"], errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["expected"] = x.clip(lower=0.0)
    return out


# ───────────────────────────── tier/kademe ataması ─────────────────────────────
def assign_tier_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    expected'a göre 5-kademe "tier" sütunu atar.
    Veri azsa/bağıl ayrım yoksa graceful fallback uygular.
    """
    if df is None or df.empty or "expected" not in df.columns:
        return df
    out = _ensure_expected_numeric(df)
    x = out["expected"].to_numpy(dtype=float)

    labels5 = ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek"]

    # Çok az varyans varsa tek kademe bırak
    if np.unique(x[~np.isnan(x)]).size < 5 or np.count_nonzero(~np.isnan(x)) < 5:
        out["tier"] = "Çok Düşük"
        return out

    # qcut → olmazsa manuel quantile cut
    try:
        out["tier"] = pd.qcut(out["expected"], q=5, labels=labels5, duplicates="drop").astype(str)
        if out["tier"].isna().all():
            raise ValueError
        return out
    except Exception:
        try:
            q = np.quantile(x, [0.20, 0.40, 0.60, 0.80]).astype(float)
            eps = max(1e-9, 1e-6 * float(np.nanmax(x) - np.nanmin(x)))
            for i in range(1, len(q)):
                if q[i] <= q[i - 1]:
                    q[i] = q[i - 1] + eps
            bins = np.concatenate(([-np.inf], q, [np.inf]))
            out["tier"] = pd.cut(out["expected"], bins=bins, labels=labels5, include_lowest=True).astype(str)
            return out
        except Exception:
            # kaba fallback: medyan/quantile tabanlı
            med = float(np.nanmedian(x))
            p75 = float(np.nanquantile(x, 0.75))
            p90 = float(np.nanquantile(x, 0.90))

            def fb(v: float) -> str:
                if v <= med * 0.5:
                    return "Çok Düşük"
                if v <= med:
                    return "Düşük"
                if v <= p75:
                    return "Orta"
                if v <= p90:
                    return "Yüksek"
                return "Çok Yüksek"

            out["tier"] = [fb(float(v)) for v in out["expected"]]
            return out


# ───────────────────────────── geçmiş kayıt / yorgunluk ─────────────────────────────
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
    logs.append({
        "ts": int(time.time()),
        "meta": meta or {},
        "plan": plan or {},
    })
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
                # Sadece 'cells' liste; 'planned_cells' int'tir.
                for g in z.get("cells", []):
                    gids.add(str(g))
        except Exception:
            continue
    return gids


def apply_fatigue_penalty(
    df: pd.DataFrame,
    penalized_geoids: Iterable[str],
    alpha: float = 0.25,
    key_col: str = KEY_COL,
) -> pd.DataFrame:
    """
    Dün/son X saatte devriye yapılmış hücreleri %alpha kadar indir (coverage etkisi).
    """
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


# ───────────────────────────── ağırlıklandırma ve çeşitlilik ─────────────────────────────
def make_weighted_view(
    df: pd.DataFrame,
    dist: Dict[str, float] = DEFAULT_DIST,
    jitter: float = 0.05,
) -> pd.DataFrame:
    """
    Tier paylarını uygula: expected_w = expected * tier_weight * (1 ± jitter).
    Küçük jitter, alternatif planların birbirinden ayrışmasını sağlar.
    """
    if df is None or df.empty:
        return df
    out = assign_tier_safe(_ensure_expected_numeric(df))
    wmap = {**{k: 0.0 for k in ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Çok Yüksek"]}, **(dist or {})}
    rnd = (1.0 + np.random.uniform(-abs(jitter), abs(jitter), size=len(out)))
    tier_w = out["tier"].map(wmap).fillna(0.0).to_numpy(dtype=float)
    out["expected"] = out["expected"].astype(float) * tier_w * rnd
    # hepsi 0'a giderse, küçük taban sinyali
    if (out["expected"] <= 0).all():
        out["expected"] = out["expected"] + 1e-6
    return out


def diversify_against(
    df: pd.DataFrame,
    taken_geoids: Iterable[str],
    key_col: str = KEY_COL,
    gamma: float = 0.35,
) -> pd.DataFrame:
    """
    Daha önceki plan/adaylarda seçilmiş hücreleri ek olarak bastır:
      expected := expected * (1 - gamma)  (seçilmişlerde)
    """
    if df is None or df.empty:
        return df
    out = _ensure_keycol(_ensure_expected_numeric(df), key_col)
    if key_col not in out.columns or "expected" not in out.columns:
        return out
    taken = set(map(str, taken_geoids or []))
    if not taken:
        return out
    m = out[key_col].astype(str).isin(taken)
    out.loc[m, "expected"] = out.loc[m, "expected"] * (1.0 - float(gamma))
    return out


# ────────────── planlardan üst hücreleri toplayan yardımcı (çeşitlilik için) ──────────────
def _collect_top_geoids_from_plan(
    plan: dict,
    key_col: str = KEY_COL,
    fallback_df: Optional[pd.DataFrame] = None,
    top_n: int = 60
) -> set[str]:
    try:
        cells: List[str] = []
        for z in (plan or {}).get("zones", []):
            cells.extend(z.get("cells", []))  # DİKKAT: planned_cells int, liste değil
        if cells:
            return set(map(str, cells))
        if fallback_df is not None and len(fallback_df) > 0:
            return set(
                fallback_df.sort_values("expected", ascending=False)
                           .head(int(top_n))[key_col].astype(str).tolist()
            )
    except Exception:
        pass
    return set()


# ───────────────────────────── ana API ─────────────────────────────
def propose_patrol_plans(
    base_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int,
    allocate_fn: Callable[..., dict],     # örn. utils.patrol.allocate_patrols
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
    seed: int | None = None,
) -> List[dict]:
    """
    Çoklu devriye önerileri üretir (varsayılan 5 adet).

    Adımlar:
      1) Geçmiş devriyelerden (son `fatigue_hours`) gelen hücrelere ceza uygula.
      2) Tier dağılımına göre ağırlıklandır; jitter ile çeşitlilik ver.
      3) (opsiyonel) Önceki planlardaki top-N hücrelere çeşitlilik cezası uygula.
      4) allocate_fn ile planı üret, meta bilgisi ekle.
    """
    if seed is not None:
        np.random.seed(int(seed))

    base_df = _ensure_keycol(_ensure_expected_numeric(base_df), key_col)
    geo_df = _ensure_keycol(geo_df, key_col)

    # Geçmiş yorgunluk/coverage
    logs = load_saved_plans(logs_path)
    penalized = recent_patrolled_geoids(logs, lookback_hours=fatigue_hours)

    plans: List[dict] = []
    already_top: set[str] = set()

    for i in range(int(n_plans)):
        # 1) Yorgunluk cezası
        df_i = apply_fatigue_penalty(
            base_df, penalized_geoids=penalized, alpha=fatigue_alpha, key_col=key_col
        )

        # 2) Tier ağırlıkları + jitter
        df_i = make_weighted_view(df_i, dist=dist, jitter=0.08 + 0.02 * i)

        # 3) Çeşitlilik: önceki planların top-N hücrelerine gamma cezası
        if diversify and already_top:
            df_i = diversify_against(
                df_i, taken_geoids=already_top, key_col=key_col, gamma=diversify_gamma
            )

        # 4) Planı üret
        kwargs = dict(
            df_agg=df_i,
            geo_df=geo_df,
            k_planned=int(k_planned),
            duty_minutes=int(duty_minutes),
            cell_minutes=int(cell_minutes),
        )
        if travel_overhead is not None:
            kwargs["travel_overhead"] = float(travel_overhead)

        plan = allocate_fn(**kwargs)

        # 5) Meta + çeşitlilik güncellemesi
        if isinstance(plan, dict):
            plan.setdefault("meta", {})
            plan["meta"].update({
                "planner": "patrol_planner.py",
                "plan_no": i + 1,
                "tier_distribution": dist,
                "fatigue_hours": fatigue_hours,
                "fatigue_alpha": fatigue_alpha,
                "diversify": diversify,
                "diversify_top_n": diversify_top_n,
                "diversify_gamma": diversify_gamma,
            })

            # planın top hücrelerini biriktir (çeşitlilik için)
            top_ids = _collect_top_geoids_from_plan(plan, key_col=key_col, fallback_df=df_i, top_n=diversify_top_n)
            already_top |= top_ids

        plans.append(plan)

    return plans


# ───────────────────────────── seçim & kayıt yardımcıları ─────────────────────────────
def pick_plan(plans: List[dict], idx: int = 0) -> dict:
    """Plan listesinden güvenli seçim (out-of-range → ilk plan)."""
    if not plans:
        return {}
    i = int(idx)
    if i < 0 or i >= len(plans):
        i = 0
    return plans[i] or {}


def save_selected_plan(plans: List[dict], idx: int = 0, meta: dict | None = None, path: str = "data/patrol_logs.json") -> dict:
    """
    UI’de kullanıcı seçimi sonrası planı kalıcı kaydet.
    Dönüş: Kaydedilen plan (boşsa {}).
    """
    plan = pick_plan(plans, idx=idx)
    if plan:
        save_plan(plan, meta=meta or {}, path=path)
    return plan


# ───────────────────────────── kısa yol sarmalayıcılar (UI) ─────────────────────────────
def make_priority_plans(
    base_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int = 6,
    *,
    n_plans: int = 3,               # 2–3 öneri için 2 veya 3 yap
    travel_overhead: float = 0.40,
    seed: Optional[int] = 101,
) -> List[dict]:
    """Risk-öncelikli çoklu plan (kullanım: 2–3 adet)."""
    try:
        from utils.patrol import allocate_patrols as _alloc
    except Exception:
        raise ImportError("utils.patrol.allocate_patrols bulunamadı")

    allocate_priority = partial(
        _alloc,
        strategy="priority",
        init="topk",
        distance_push_alpha=0.0,  # risk öncelikli modda yayılma itişi kapalı
    )

    return propose_patrol_plans(
        base_df=base_df,
        geo_df=geo_df,
        k_planned=k_planned,
        duty_minutes=duty_minutes,
        cell_minutes=cell_minutes,
        allocate_fn=allocate_priority,
        n_plans=n_plans,
        travel_overhead=travel_overhead,
        seed=seed,
        # risk öncelikli: yüksekleri daha çok ödüllendir
        dist={"Çok Yüksek": 1.0, "Yüksek": 0.7, "Orta": 0.4, "Düşük": 0.2, "Çok Düşük": 0.1},
        diversify=True,
        diversify_top_n=60,
        diversify_gamma=0.25,
    )


def make_balanced_plans(
    base_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int = 6,
    *,
    n_plans: int = 6,               # 5–6 öneri için 5 veya 6 yap
    travel_overhead: float = 0.40,
    tier_quota: Optional[Dict[str, float]] = None,
    distance_push_alpha: float = 0.15,
    seed: Optional[int] = 707,
) -> List[dict]:
    """Kotalı-dengeli çoklu plan (kullanım: 5–6 adet)."""
    try:
        from utils.patrol import allocate_patrols as _alloc
    except Exception:
        raise ImportError("utils.patrol.allocate_patrols bulunamadı")

    if tier_quota is None:
        tier_quota = DEFAULT_DIST  # aynı yapıda; allocate içinde quota olarak kullanılabilir

    allocate_balanced = partial(
        _alloc,
        strategy="balanced",
        tier_quota=tier_quota,
        distance_push_alpha=distance_push_alpha,
        init="farthest",     # merkezler yayılır
        jitter_scale=2e-4,   # küçük geometri jitter
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
        seed=seed,
        dist=DEFAULT_DIST,
        diversify=True,
        diversify_top_n=80,
        diversify_gamma=0.35,
    )
