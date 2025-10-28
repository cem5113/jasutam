# utils/patrol_planner.py (yeni dosya aç)
from __future__ import annotations
import json, time, random
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_DIST = {"Çok Yüksek":0.50, "Yüksek":0.30, "Orta":0.10, "Düşük":0.10}  # %50/%30/%10/%10
# "Çok Düşük" bilinçli olarak 0: hiç risk yok katmanı

def assign_tier_safe(df: pd.DataFrame) -> pd.DataFrame:
    """expected'a göre 5'li kademe; df['tier'] yazar. df boşsa aynen döner."""
    if df is None or df.empty or "expected" not in df.columns:
        return df
    out = df.copy()
    x = pd.to_numeric(out["expected"], errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
    labels5 = ["Çok Düşük","Düşük","Orta","Yüksek","Çok Yüksek"]
    try:
        out["tier"] = pd.qcut(x, q=5, labels=labels5, duplicates="drop").astype(str)
    except Exception:
        q = np.quantile(x.to_numpy(), [0.20,0.40,0.60,0.80])
        bins = np.concatenate(([-np.inf], q, [np.inf]))
        out["tier"] = pd.cut(x, bins=bins, labels=labels5, include_lowest=True).astype(str)
    return out

# ---- geçmiş kaydı
def load_saved_plans(path: str = "data/patrol_logs.json") -> list[dict]:
    p = Path(path)
    if not p.exists(): return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_plan(plan: dict, meta: dict, path: str = "data/patrol_logs.json") -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    logs = load_saved_plans(path)
    logs.append({"ts": int(time.time()), "meta": meta, "plan": plan})
    p.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")

def recent_patrolled_geoids(logs: list[dict], lookback_hours: int = 24) -> set[str]:
    cutoff = int(time.time()) - lookback_hours * 3600
    gids: set[str] = set()
    for item in logs:
        if int(item.get("ts",0)) < cutoff: continue
        zones = (item.get("plan") or {}).get("zones", [])
        for z in zones:
            for g in z.get("planned_cells", []):
                gids.add(str(g))
    return gids

def apply_fatigue_penalty(df: pd.DataFrame, penalized_geoids: set[str], alpha: float = 0.25, key_col: str = "GEOID") -> pd.DataFrame:
    """Dün devriye yapılan hücreleri %alpha kadar indir (coverage etkisi)."""
    if df is None or df.empty: return df
    out = df.copy()
    if key_col not in out.columns: return out
    mask = out[key_col].astype(str).isin(penalized_geoids)
    if "expected" in out.columns:
        out.loc[mask, "expected"] = out.loc[mask, "expected"] * (1.0 - alpha)
    return out

# ---- dağıtım ve jitter
def make_weighted_view(df: pd.DataFrame, dist: dict[str,float], jitter: float = 0.05) -> pd.DataFrame:
    """
    Tier paylarını uygula: expected_w = expected * tier_weight * (1 ± jitter).
    jitter: küçük rastgelelik; birden çok öneri için farklı rotalar çıkmasını sağlar.
    """
    if df is None or df.empty: return df
    out = df.copy()
    out = assign_tier_safe(out)
    wmap = {**{k:0.0 for k in ["Çok Düşük","Düşük","Orta","Yüksek","Çok Yüksek"]}, **dist}
    rnd = (1.0 + np.random.uniform(-jitter, jitter, size=len(out)))
    tier_w = out["tier"].map(wmap).fillna(0.0).to_numpy()
    out["expected"] = out["expected"].astype(float) * tier_w * rnd
    return out

def propose_patrol_plans(
    base_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int,
    allocate_fn,                     # allocate_patrols fonksiyonu (DI)
    key_col: str = "GEOID",
    n_plans: int = 5,
    dist: dict[str,float] = DEFAULT_DIST,
    logs_path: str = "data/patrol_logs.json",
    fatigue_hours: int = 24,
    fatigue_alpha: float = 0.25,
) -> list[dict]:
    """
    5 alternatif plan üretir. Her biri:
      - Son 24 saatte devriye yapılmış hücrelere ceza uygular
      - Tier paylarını 50/30/10/10'a göre ağırlıklandırır
      - Küçük jitter ile farklı çözüm üretir
      - allocate_patrols ile planı döndürür
    """
    logs = load_saved_plans(logs_path)
    penalized = recent_patrolled_geoids(logs, lookback_hours=fatigue_hours)

    plans = []
    for i in range(n_plans):
        df_i = apply_fatigue_penalty(base_df, penalized, alpha=fatigue_alpha, key_col=key_col)
        df_i = make_weighted_view(df_i, dist=dist, jitter=0.08 + 0.02*i)  # her planda jitter biraz farklı
        # expected tamamen 0 olabilir; güvenli küçük taban
        if "expected" in df_i.columns and (df_i["expected"]<=0).all():
            df_i["expected"] = base_df["expected"].astype(float) * 1e-6

        plan = allocate_fn(
            df_agg=df_i,
            geo_df=geo_df,
            k_planned=int(k_planned),
            duty_minutes=int(duty_minutes),
            cell_minutes=int(cell_minutes),
        )
        # Etiketle: plan_no ve dağıtım
        if isinstance(plan, dict):
            plan["meta"] = {
                "plan_no": i+1,
                "tier_distribution": dist,
                "fatigue_hours": fatigue_hours,
                "fatigue_alpha": fatigue_alpha,
            }
        plans.append(plan)
    return plans
