# models/xai.py
from __future__ import annotations
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# 0) Saha-dostu kısa etiketler (kural tabanlı açıklama için)
# ============================================================
_LABELS: Dict[str, str] = {
    "nr_7d": "Yakın tekrar (7g)",
    "nr_14d": "Yakın tekrar (14g)",
    "nei_7d_sum": "Komşu yoğunluğu (7g)",
    "911_request_count_hour_range": "911 yoğunluğu (saat)",
    "911_request_count_daily(before_24_hours)": "911 yoğunluğu (24s)",
    "poi_risk_score": "POI risk skoru",
    "poi_risk_score_range": "POI risk skoru (kategori)",
    "bus_stop_count": "Otobüs durağı sayısı",
    "train_stop_count": "Tren durağı sayısı",
    "distance_to_police": "Polise uzaklık",
    "distance_to_government_building": "Kamu binasına uzaklık",
    "precip": "Yağış",
    "wind_speed": "Rüzgâr",
    "event_hour": "Saat etkisi",
}

# 18–02 saatleri nispeten yüksek risk farz edilir (yer tutucu ağırlık)
_HOUR_WEIGHT: Dict[int, float] = {h: (1.0 if 18 <= h <= 23 or 0 <= h <= 2 else 0.4) for h in range(24)}

def _val(row: dict, k: str, default: float = 0.0) -> float:
    v = row.get(k, default)
    try:
        f = float(v)
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default

# ============================================================
# 1) Kural tabanlı saha açıklaması (model bağımsız, hızlı)
# ============================================================
def brief_xai_for_row(row: dict) -> List[Dict]:
    """
    Model olmasa bile sahaya okunur kısa açıklama üretir.
    Öncelik: yakın tekrar, komşu yoğunluğu, 911, POI, saat, hava.
    Dönen: ilk 3 faktör (score'a göre)
    """
    feats: List[Dict] = []

    # 1) Yakın tekrarlar
    nr7  = _val(row, "nr_7d")
    nr14 = _val(row, "nr_14d")
    if nr7 > 0:
        feats.append({"name": _LABELS["nr_7d"], "score": nr7, "why": "Son 7 günde benzer olay birikti."})
    if nr14 > 0:
        feats.append({"name": _LABELS["nr_14d"], "score": 0.6 * nr14, "why": "Son 14 günde tekrar sinyali var."})

    # 2) Komşu yoğunluğu
    nei = _val(row, "nei_7d_sum")
    if nei > 0:
        feats.append({"name": _LABELS["nei_7d_sum"], "score": 0.8 * nei, "why": "Komşu hücrelerde son 7 günde artış."})

    # 3) 911 yükleri
    r911h = _val(row, "911_request_count_hour_range")
    r911d = _val(row, "911_request_count_daily(before_24_hours)")
    if r911h > 0:
        feats.append({"name": _LABELS["911_request_count_hour_range"], "score": 1.2 * r911h, "why": "Saatlik 911 çağrıları yüksek."})
    if r911d > 0:
        feats.append({"name": _LABELS["911_request_count_daily(before_24_hours)"], "score": r911d, "why": "Son 24 saatte 911 çağrıları arttı."})

    # 4) POI / çevre
    poi = max(_val(row, "poi_risk_score"), _val(row, "poi_risk_score_range"))
    if poi > 0:
        feats.append({"name": _LABELS["poi_risk_score_range"], "score": 0.7 * poi, "why": "Riskli POI yoğunluğu etkili."})

    # 5) Toplu taşıma
    bus = _val(row, "bus_stop_count")
    train = _val(row, "train_stop_count")
    if bus > 0:
        feats.append({"name": _LABELS["bus_stop_count"], "score": 0.4 * bus, "why": "Yaya/hareketlilik artışı (otobüs)."})
    if train > 0:
        feats.append({"name": _LABELS["train_stop_count"], "score": 0.5 * train, "why": "Yaya/hareketlilik artışı (tren)."})

    # 6) Saat etkisi
    hr = int(_val(row, "event_hour", default=-1))
    if 0 <= hr <= 23:
        feats.append({"name": _LABELS["event_hour"], "score": 2.0 * _HOUR_WEIGHT.get(hr, 0.4), "why": f"Saat {hr:02d}:00 dilimi görece riskli."})

    # 7) Hava (opsiyonel)
    precip = _val(row, "precip")
    if precip > 0:
        feats.append({"name": _LABELS["precip"], "score": 0.3 * precip, "why": "Yağış, belirli suç tiplerini etkileyebilir."})

    # yumuşatma + sıralama
    for f in feats:
        f["score"] = round(math.log1p(max(f["score"], 0.0)), 4)
    feats = sorted(feats, key=lambda x: x["score"], reverse=True)
    return feats[:3]

# ============================================================
# 2) Global doğrusal XAI (tasarım matrisi + ridge)
# ============================================================
def _pick_category_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("category_grouped", "category", "subcategory_grouped", "subcategory", "crime_type"):
        if c in df.columns:
            return c
    return None

def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
    else:
        out["date"] = pd.to_datetime(out.get("date", pd.NaT), errors="coerce")
    if "event_hour" not in out.columns:
        out["event_hour"] = out["date"].dt.hour.fillna(0).astype(int)
    if "day_of_week" not in out.columns:
        out["day_of_week"] = out["date"].dt.dayofweek
    return out

def _target_series(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Hedef p (0..1):
      - varsa pred_p_occ
      - yoksa pred_expected → p ≈ 1 - exp(-λ)
      - yoksa crime_count normalize (0..1)
    """
    if "pred_p_occ" in df.columns:
        y = pd.to_numeric(df["pred_p_occ"], errors="coerce").clip(0, 1).fillna(0.0)
        return y, "pred_p_occ"
    if "pred_expected" in df.columns:
        lam = pd.to_numeric(df["pred_expected"], errors="coerce").fillna(0.0).clip(lower=0.0)
        p = (1.0 - np.exp(-lam)).clip(0, 1)
        return pd.Series(p, index=df.index), "pred_p_occ(λ)"
    if "crime_count" in df.columns:
        s = pd.to_numeric(df["crime_count"], errors="coerce").fillna(0.0)
        mx = float(s.max() or 1.0)
        return (s / (mx if mx > 0 else 1.0)).clip(0, 1), "count_norm"
    return pd.Series(np.zeros(len(df)), index=df.index), "zero"

def _topk_categories(df: pd.DataFrame, k: int = 8) -> List[str]:
    col = _pick_category_col(df)
    if not col:
        return []
    vc = (df[col].astype(str).value_counts(dropna=True).head(k)).index.tolist()
    return [str(x) for x in vc]

def _dow_name(i: int) -> str:
    names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
    if 0 <= int(i) <= 6:
        return names[int(i)]
    return str(i)

def _hour_bucket(h: int) -> str:
    # 0-5 gece, 6-11 sabah, 12-17 öğlen, 18-23 akşam
    if   0 <= h <= 5:   return "hour_night"
    if   6 <= h <= 11:  return "hour_morning"
    if  12 <= h <= 17:  return "hour_afternoon"
    return "hour_evening"

def build_design_matrix(df_raw: pd.DataFrame, cat_k: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Dönüş:
      X  : z-skorlanmış özellik matrisi
      meta : her sütun için {name, kind, label}
      y_name: hedef adı
    """
    df = _ensure_time(df_raw)
    y, y_name = _target_series(df)

    feats: Dict[str, pd.Series] = {}
    meta_rows: List[Dict] = []

    # Sürekli özellikler (varsa)
    if "temp_score" in df.columns:
        feats["temp_score"] = pd.to_numeric(df["temp_score"], errors="coerce").fillna(0.0)
        meta_rows.append({"name": "temp_score", "kind": "cont", "label": "Geçici hotspot (2g/30g)"})
    if "stable_score" in df.columns:
        feats["stable_score"] = pd.to_numeric(df["stable_score"], errors="coerce").fillna(0.0)
        meta_rows.append({"name": "stable_score", "kind": "cont", "label": "Kalıcı hotspot (90g)"})

    # Saat bucket (4’lü)
    hb = df["event_hour"].fillna(0).astype(int).map(_hour_bucket)
    for tag, lab in {
        "hour_night": "Gece (0–5)",
        "hour_morning": "Sabah (6–11)",
        "hour_afternoon": "Öğlen (12–17)",
        "hour_evening": "Akşam (18–23)",
    }.items():
        feats[tag] = (hb == tag).astype(int)
        meta_rows.append({"name": tag, "kind": "onehot", "label": lab})

    # Haftanın günü
    if "day_of_week" in df.columns:
        for d in range(7):
            name = f"dow_{d}"
            feats[name] = (df["day_of_week"].fillna(-1).astype(int) == d).astype(int)
            meta_rows.append({"name": name, "kind": "onehot", "label": f"Gün: {_dow_name(d)}"})

    # Kategori one-hot (ilk k)
    cat_top = _topk_categories(df, k=cat_k)
    col = _pick_category_col(df)
    if col and cat_top:
        catv = df[col].astype(str)
        for c in cat_top:
            nm = f"cat_{c}"
            feats[nm] = (catv == c).astype(int)
            meta_rows.append({"name": nm, "kind": "onehot", "label": f"Tür: {c}"})

    # Özellik matrisi
    if not feats:
        feats["bias"] = pd.Series(np.ones(len(df)), index=df.index)
        meta_rows.append({"name": "bias", "kind": "bias", "label": "Sabit terim"})
    X = pd.DataFrame(feats, index=df.index)

    # Z-skor
    means = X.mean(axis=0)
    stds = X.std(axis=0).replace(0, 1.0)
    Xz = (X - means) / stds

    meta = pd.DataFrame(meta_rows).drop_duplicates(subset=["name"]).reset_index(drop=True)

    # y’yi da ekleyelim (aynı index)
    Xz["__y__"] = y
    return Xz.drop(columns=["__y__"]), meta, y_name

def _ridge(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    # w = (X^T X + λI)^(-1) X^T y
    n_feats = X.shape[1]
    A = X.T @ X + l2 * np.eye(n_feats)
    b = X.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
    return w

def fit_global_explainer(df_raw: pd.DataFrame, cat_k: int = 8, l2: float = 1e-3) -> Dict:
    """
    Veri üzerinde tek bir global doğrusal açıklayıcı fit eder.
    Dönen:
      {
        "weights": pd.Series (feature->coef),
        "meta": pd.DataFrame(name, kind, label).set_index("name"),
        "y_name": str,
        "cat_k": int,
        "l2": float,
      }
    """
    Xz, meta, y_name = build_design_matrix(df_raw, cat_k=cat_k)
    X = np.nan_to_num(Xz.values, nan=0.0, posinf=0.0, neginf=0.0)

    # hedef p
    y_s, _ = _target_series(_ensure_time(df_raw))
    y = np.nan_to_num(y_s.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    w = _ridge(X, y, l2=l2)
    weights = pd.Series(w, index=Xz.columns)

    return {
        "weights": weights,
        "meta": meta.set_index("name"),
        "y_name": y_name,
        "cat_k": cat_k,
        "l2": l2,
    }

def _format_reason(name: str, meta: pd.DataFrame) -> str:
    if isinstance(meta, pd.DataFrame) and name in meta.index and isinstance(meta.loc[name, "label"], str):
        return meta.loc[name, "label"]
    if name.startswith("dow_"):
        return f"Gün: {_dow_name(int(name.split('_')[1]))}"
    if name.startswith("cat_"):
        return f"Tür: {name[4:]}"
    if name.startswith("hour_"):
        labels = {
            "hour_night": "Gece (0–5)",
            "hour_morning": "Sabah (6–11)",
            "hour_afternoon": "Öğlen (12–17)",
            "hour_evening": "Akşam (18–23)",
        }
        return labels.get(name, name)
    if name == "temp_score":   return "Geçici hotspot"
    if name == "stable_score": return "Kalıcı hotspot"
    if name == "bias":         return "Sabit terim"
    return name

def explain_rows(df_raw: pd.DataFrame, model: Dict, topk: int = 3) -> pd.DataFrame:
    """
    Global doğrusal açıklayıcıya göre her satır için en yüksek |katkı| veren top-k neden.
    Dönüş: df[['xai_reasons','xai_top_names','xai_top_vals']]
    """
    Xz, meta, _ = build_design_matrix(df_raw, cat_k=int(model.get("cat_k", 8)))
    X = np.nan_to_num(Xz.values, nan=0.0, posinf=0.0, neginf=0.0)
    w = model["weights"].reindex(Xz.columns).fillna(0.0).values

    contrib = X * w  # her satır, her özellik için katkı (w * z)
    abs_contrib = np.abs(contrib)

    top_names: List[List[str]] = []
    top_vals:  List[List[float]] = []
    reasons:   List[str] = []

    meta_idx = model.get("meta")
    meta_df = meta_idx if isinstance(meta_idx, pd.DataFrame) else pd.DataFrame()

    for i in range(X.shape[0]):
        idxs = np.argsort(-abs_contrib[i])[:max(1, int(topk))]
        names = Xz.columns[idxs].tolist()
        vals  = contrib[i, idxs].tolist()
        top_names.append(names)
        top_vals.append(vals)

        parts = []
        for nm, v in zip(names, vals):
            sign = "↑" if v >= 0 else "↓"
            parts.append(f"{_format_reason(nm, meta_df)} ({sign})")
        reasons.append(" • ".join(parts))

    out = pd.DataFrame({
        "xai_top_names": top_names,
        "xai_top_vals": top_vals,
        "xai_reasons": reasons,
    }, index=df_raw.index)
    return out

def attach_xai(df_raw: pd.DataFrame, topk: int = 3, l2: float = 1e-3, cat_k: int = 8) -> pd.DataFrame:
    """
    Tek satırlık kullanım:
      df_with_xai = attach_xai(df)
    """
    model = fit_global_explainer(df_raw, cat_k=cat_k, l2=l2)
    ex = explain_rows(df_raw, model, topk=topk)
    return pd.concat([df_raw.reset_index(drop=True), ex.reset_index(drop=True)], axis=1)

__all__ = [
    "brief_xai_for_row",
    "build_design_matrix",
    "fit_global_explainer",
    "explain_rows",
    "attach_xai",
]
