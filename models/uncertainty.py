# sutam/models/uncertainty.py
from __future__ import annotations
import math
from typing import Optional, Dict, Iterable, Tuple, List

import numpy as np
import pandas as pd

# ============================================================
# 0) Olasılık etiketi ve güven etiketi (UI ile uyumlu)
# ============================================================
def risk_level_from_prob(p: float, thresholds: Dict[str, float]) -> str:
    """
    p: gerçekleşme olasılığı (0..1)
    thresholds: {"low": 0.33, "mid": 0.66, ...}
    """
    if p is None:
        return "Bilinmiyor"
    try:
        p = float(p)
    except Exception:
        return "Bilinmiyor"
    if np.isnan(p):
        return "Bilinmiyor"
    low = float(thresholds.get("low", 0.33))
    mid = float(thresholds.get("mid", 0.66))
    return "Yüksek" if p > mid else ("Orta" if p > low else "Düşük")


def confidence_label(q10: Optional[float], q90: Optional[float]) -> str:
    """
    q10–q90 yayılımından basit güven etiketi.
    """
    if q10 is None or q90 is None:
        return "—"
    try:
        a = float(q10); b = float(q90)
    except Exception:
        return "—"
    if np.isnan(a) or np.isnan(b):
        return "—"
    spread = b - a
    if spread <= 0.5:
        return "Yüksek güven"
    if spread <= 1.5:
        return "Orta güven"
    return "Düşük güven"


def add_confidence_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    q10/q90 varsa 'confidence' sütunu ekler; yoksa dokunmaz.
    """
    if {"pred_q10", "pred_q90"}.issubset(df.columns):
        out = df.copy()
        out["confidence"] = [
            confidence_label(a, b) for a, b in zip(out["pred_q10"], out["pred_q90"])
        ]
        return out
    return df


# ============================================================
# 1) Poisson yardımcıları (baseline.py tarafından beklenen API)
# ============================================================
def prob_at_least_one(lmbda: float) -> float:
    """
    P(X >= 1) = 1 - P(X = 0) = 1 - e^{-λ}
    """
    try:
        lam = float(lmbda)
    except Exception:
        return np.nan
    if lam < 0:
        return np.nan
    return 1.0 - math.exp(-lam)


def _poisson_cdf(k: int, lam: float) -> float:
    """Kümülatif Poisson CDF (bağımlılıksız, basit hesap)."""
    s = 0.0
    # küçük lam'da sorun yok; büyük lam için performans istersen ileride scipy kullanılabilir.
    for i in range(0, k + 1):
        s += math.exp(-lam) * (lam ** i) / math.factorial(i)
    return s


def poisson_quantiles(
    lmbda: float,
    q_low: float = 0.10,
    q_high: float = 0.90
) -> Tuple[int, int]:
    """
    Poisson(λ) için yaklaşık alt/üst kuantil (tam-sayı).
    Varsayılanlar q_low=0.10, q_high=0.90; UI'daki q10/q90 ile uyumlu.
    """
    try:
        lam = float(lmbda)
        ql = float(q_low)
        qh = float(q_high)
    except Exception:
        return (np.nan, np.nan)  # type: ignore

    if lam < 0 or not (0.0 < ql < qh < 1.0):
        return (np.nan, np.nan)  # type: ignore

    # alt kuantil
    k_low = 0
    while _poisson_cdf(k_low, lam) < ql:
        k_low += 1

    # üst kuantil
    k_high = k_low
    while _poisson_cdf(k_high, lam) < qh:
        k_high += 1

    return int(k_low), int(k_high)


def lambda_to_p_occ(lmbda: float) -> float:
    """
    λ → en az bir olay olasılığına dönüştürür (P(X>=1)).
    baseline/predictor tarafından kullanılır.
    """
    return prob_at_least_one(lmbda)


def add_poisson_uncertainty(
    df: pd.DataFrame,
    lambda_col: str = "pred_mean",
    out_low_col: str = "pred_q10",
    out_high_col: str = "pred_q90",
    p_occ_col: str = "pred_p_occ",
    q_low: float = 0.10,
    q_high: float = 0.90,
    clip_zero: bool = True
) -> pd.DataFrame:
    """
    Bir DataFrame’e Poisson belirsizlik bandı ve gerçekleşme olasılığı sütunları ekler.
    - lambda_col: λ (beklenen olay sayısı)
    - out_low_col/out_high_col: kuantil sütun adları (q10/q90 default)
    - p_occ_col: P(X>=1)
    """
    if lambda_col not in df.columns:
        # Kolon yoksa dokunmadan dön
        return df

    out = df.copy()
    lows: List[float] = []
    highs: List[float] = []
    p_occs: List[float] = []

    for lam in out[lambda_col].tolist():
        try:
            lam_f = float(lam)
        except Exception:
            lam_f = np.nan

        if np.isnan(lam_f) or lam_f < 0:
            lows.append(np.nan)
            highs.append(np.nan)
            p_occs.append(np.nan)
            continue

        if clip_zero and lam_f < 0:
            lam_f = 0.0

        ql, qh = poisson_quantiles(lam_f, q_low=q_low, q_high=q_high)
        lows.append(ql)
        highs.append(qh)
        p_occs.append(lambda_to_p_occ(lam_f))

    out[out_low_col] = lows
    out[out_high_col] = highs
    out[p_occ_col] = p_occs
    return out

# ============================================================
# 1) Poisson → P(N>=1) yardımcıları
# ============================================================
def lambda_to_p_occ(lam: np.ndarray | pd.Series | float) -> np.ndarray:
    """Poisson varsayımı: P(N>=1) = 1 - exp(-λ)."""
    arr = np.asarray(lam, dtype=float)
    arr = np.clip(arr, 0.0, None)
    return 1.0 - np.exp(-arr)

# ============================================================
# 2) Hızlı (vektörel) normal yaklaşımı kantilleri
#    q ≈ λ + z * sqrt(λ)   (kontinü düzeltme ihmal)
# ============================================================
def poisson_normal_quantiles(
    lam: Iterable[float],
    qs: Tuple[float, ...] = (0.1, 0.5, 0.9)
) -> np.ndarray:
    """
    Poisson için hızlı normal yaklaşımı: q ≈ λ + z * sqrt(λ), λ>=0.
    SciPy olmadan hafif ve vektörel.
    Dönen: shape (n, len(qs))
    """
    lam = np.asarray(list(lam), dtype=float)
    lam = np.clip(lam, 0.0, None)

    # z-skorları
    try:
        from statistics import NormalDist
        zs = np.array([NormalDist().inv_cdf(q) for q in qs], dtype=float)
    except Exception:
        # Yedek: sık kullanılan quantile'lar için sabitler
        z_map = {0.1: -1.2815515655, 0.5: 0.0, 0.9: 1.2815515655}
        zs = np.array([z_map.get(float(q), 0.0) for q in qs], dtype=float)

    root = np.sqrt(np.maximum(lam, 1e-12))
    q = lam[:, None] + root[:, None] * zs[None, :]
    return np.maximum(0.0, q)

def attach_poisson_quantiles(df: pd.DataFrame, lambda_col: str = "pred_expected") -> pd.DataFrame:
    """
    df[lambda_col] → df['pred_q10','pred_q50','pred_q90'] ekler (normal yaklaşımı).
    """
    if lambda_col not in df.columns:
        return df
    qs = poisson_normal_quantiles(df[lambda_col].values, qs=(0.10, 0.50, 0.90))
    out = df.copy()
    out["pred_q10"] = qs[:, 0]
    out["pred_q50"] = qs[:, 1]
    out["pred_q90"] = qs[:, 2]
    return out

# ============================================================
# 3) Daha doğru (küçük λ’larda güvenli) kantiller
#    Küçük λ → kesin CDF; büyük λ → normal yaklaşımı
# ============================================================
_Z = {0.1: -1.2815515655446004, 0.5: 0.0, 0.9: 1.2815515655446004}

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def _poisson_cdf(k: int, lam: float) -> float:
    s = 0.0
    for i in range(0, max(0, k) + 1):
        s += _poisson_pmf(i, lam)
    return min(s, 1.0)

def poisson_quantile(lam: float, q: float) -> int:
    """
    Poisson(lam) için yaklaşık/yarı-kesin kantil.
    q tipik olarak {0.1, 0.5, 0.9}.
    Küçük λ’da CDF ile arama; λ>=50'de normal yaklaşımı (hızlı).
    """
    if lam <= 0:
        return 0
    if lam >= 50 and q in _Z:
        val = lam + _Z[q] * math.sqrt(lam)
        return max(0, int(round(val)))
    # artan arama (küçük λ güvenli)
    k = 0
    k_max = int(max(10, lam + 10 * math.sqrt(max(lam, 1.0))))
    while k < k_max and _poisson_cdf(k, lam) < q:
        k += 1
    return k

def poisson_quantiles(
    lam: Iterable[float],
    qs: Tuple[float, float, float] = (0.1, 0.5, 0.9)
) -> pd.DataFrame:
    """
    Çoklu λ için q10/q50/q90 kantilleri (DataFrame döner).
    """
    recs: List[dict] = []
    for L in lam:
        Lf = float(max(0.0, float(L)))
        q10 = poisson_quantile(Lf, qs[0])
        q50 = poisson_quantile(Lf, qs[1])
        q90 = poisson_quantile(Lf, qs[2])
        recs.append({"q10": q10, "q50": q50, "q90": q90})
    return pd.DataFrame(recs)

# ============================================================
# 4) DataFrame entegrasyon yardımcıları
# ============================================================
def add_poisson_uncertainty(df: pd.DataFrame, lam_col: str = "pred_expected") -> pd.DataFrame:
    """
    pred_expected (λ) → {pred_q10,pred_q50,pred_q90} ve (yoksa) pred_p_occ = 1 - e^{-λ}.
    Küçük λ'larda daha doğru olan poisson_quantiles kullanır.
    """
    out = df.copy()
    if lam_col not in out.columns:
        return out
    lam = pd.to_numeric(out[lam_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    qs = poisson_quantiles(lam.tolist())
    out["pred_q10"] = qs["q10"].values
    out["pred_q50"] = qs["q50"].values
    out["pred_q90"] = qs["q90"].values

    # olay gerçekleşme olasılığı: 1 - P(X=0) = 1 - e^{-λ}
    if "pred_p_occ" not in out.columns:
        out["pred_p_occ"] = np.nan
    m = out["pred_p_occ"].isna()
    out.loc[m, "pred_p_occ"] = (1.0 - np.exp(-lam[m])).clip(0, 1)
    return out

def add_uncertainty_and_labels(
    df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    lam_col: str = "pred_expected",
) -> pd.DataFrame:
    """
    - λ varsa: q10/q50/q90 + pred_p_occ (yoksa) ekler
    - pred_p_occ varsa: risk seviyesi ekler
    - q10/q90 varsa: confidence etiketi ekler
    """
    thresholds = thresholds or {"low": 0.33, "mid": 0.66}
    out = add_poisson_uncertainty(df, lam_col=lam_col)

    if "pred_p_occ" in out.columns:
        out = out.copy()
        out["risk_level"] = [
            risk_level_from_prob(float(p) if pd.notna(p) else np.nan, thresholds)
            for p in out["pred_p_occ"]
        ]

    out = add_confidence_cols(out)
    return out

# ============================================================
# 5) Dışa açılan simgeler
# ============================================================
__all__ = [
    # etiketler
    "risk_level_from_prob",
    "confidence_label",
    "add_confidence_cols",
    # olasılık ve kantiller
    "lambda_to_p_occ",
    "poisson_normal_quantiles",
    "attach_poisson_quantiles",
    "poisson_quantile",
    "poisson_quantiles",
    # DF entegrasyon
    "add_poisson_uncertainty",
    "add_uncertainty_and_labels",
]
