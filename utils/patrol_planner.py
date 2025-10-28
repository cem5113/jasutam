# utils/patrol_planner.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# ── constants (güvenli import; fallback)
try:
    from utils.constants import KEY_COL  # örn. "GEOID"
except Exception:
    KEY_COL = "GEOID"

# %50 / %30 / %10 / %10 – "Çok Düşük" bilinçli 0
DEFAULT_DIST = {
    "Çok Yüksek": 0.50,
    "Yüksek":     0.30,
    "Orta":       0.10,
    "Düşük":      0.10,
    "Çok Düşük":  0.00,
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
def load_saved_plans(path: str = "data/patrol_logs.json") -> list[dict]:
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


def recent_patrolled_geoids(logs: list[dict], lookback_hours: int = 24) -> set[str]:
    cutoff = int(time.time()) - int(lookback_hours) * 3600
    gids: set[str] = set()
    for item in logs:
        try:
            if int(item.get("ts", 0)) < cutoff:
                continue
            zones = (item.get("plan") or {}).get("zones", [])
            for z in zones:
                for g in z.get("planned_cells", []) or z.get("cells", []):
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
    dist: dict[str, float] = DEFAULT_DIST,
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
    dist: dict[str, float] = DEFAULT_DIST,
    logs_path: str = "data/patrol_logs.json",
    fatigue_hours: int = 24,
    fatigue_alpha: float = 0.25,
    diversify: bool = True,
    diversify_top_n: int = 60,
    diversify_gamma: float = 0.35,
    travel_overhead: float | None = None,
    seed: int | None = None,
) -> list[dict]:
    """
    Çoklu devriye önerileri üretir (varsayılan 5 adet).

    Adımlar:
      1) Geçmiş devriyelerden (son `fatigue_hours`) gelen hücrelere ceza uygula.
      2) 50/30/10/10 tier dağılımına göre ağırlıklandır; jitter ile çeşitlilik ver.
      3) (opsiyonel) Önceki planlardaki en üst N hücreye çeşitlilik cezası uygula.
      4) allocate_fn ile planı üret, meta bilgisi ekle.

    Dönüş: [plan_dict, ...]  (allocate_fn çıktıları + meta)
    """
    # RNG sabitleme (test edilebilirlik)
    if seed is not None:
        np.random.seed(int(seed))

    base_df = _ensure_keycol(_ensure_expected_numeric(base_df), key_col)
    geo_df = _ensure_keycol(geo_df, key_col)

    # Geçmiş yorgunluk/coverage
    logs = load_saved_plans(logs_path)
    penalized = recent_patrolled_geoids(logs, lookback_hours=fatigue_hours)

    plans: list[dict] = []
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

        # 5) Meta / defansif alanları ekle
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

            # planın top-N hücrelerini güncelle (çeşitlilik için)
            try:
                # Eğer plan "planned_cells" veriyorsa onu kullan,
                # yoksa df_i'de expected'e göre nlargest al.
                cells: list[str] = []
                zones = plan.get("zones", [])
                for z in zones:
                    cells.extend((z.get("planned_cells") or z.get("cells") or []))
                if cells:
                    top_ids = set(map(str, cells))
                else:
                    top_ids = set(
                        df_i.sort_values("expected", ascending=False)
                           .head(int(diversify_top_n))[key_col].astype(str).tolist()
                    )
                already_top |= top_ids
            except Exception:
                pass

        plans.append(plan)

    return plans


# ───────────────────────────── seçim & kayıt yardımcıları ─────────────────────────────
def pick_plan(plans: list[dict], idx: int = 0) -> dict:
    """Plan listesinden güvenli seçim (out-of-range → ilk plan)."""
    if not plans:
        return {}
    i = int(idx)
    if i < 0 or i >= len(plans):
        i = 0
    return plans[i] or {}


def save_selected_plan(plans: list[dict], idx: int = 0, meta: dict | None = None, path: str = "data/patrol_logs.json") -> dict:
    """
    UI’de kullanıcı seçimi sonrası planı kalıcı kaydet.
    Dönüş: Kaydedilen plan (boşsa {}).
    """
    plan = pick_plan(plans, idx=idx)
    if plan:
        save_plan(plan, meta=meta or {}, path=path)
    return plan
