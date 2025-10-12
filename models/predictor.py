# sutam/models/predictor.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

# Paket içi importlar
from .baseline import (
    BaselineModel,
    fit_mean_by_groups,
    predict_mean_by_groups,
    fit_frequency_baseline,
    predict_expected_baseline,
)

# Basit Poisson baseline (opsiyonel)
try:
    from .baseline import baseline_expected as _baseline_expected
except Exception:
    _baseline_expected = None  # baseline_expected tanımlı değilse None ata

# Belirsizlik fonksiyonları (poisson quantiles vb.)
from .uncertainty import add_poisson_uncertainty, lambda_to_p_occ
# ---------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------
def _has_predictions(df: pd.DataFrame) -> bool:
    """Veride tahmin kolonlarından en az biri var mı?"""
    return any(c in df.columns for c in ("pred_p_occ", "pred_expected", "pred_q50"))

def _safe_clip_nonneg(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.fillna(0.0).clip(lower=0)

# ---------------------------------------------------------------------
# 1) Hafif "Hurdle-like" model
# ---------------------------------------------------------------------
class HurdleLikeModel:
    """
    Hafif/iskelet model:
      - Olasılık (occurrence) ~ grup ortalamaları (proxy, ikili hedef: y>0)
      - Pozitiflerde beklenen sayı ~ aynı grup ortalaması (count hedefi)

    Not: Bu sınıf, gerçek bir ML kitaplığı olmadan da çalışsın diye
    salt istatistiksel özetler kullanır.
    """

    def __init__(
        self,
        prob_groups: Optional[List[str]] = None,
        count_groups: Optional[List[str]] = None,
        y_col_prob: str = "crime_count",
        y_col_cnt: str = "crime_count",
    ):
        self.prob_groups = prob_groups or ["GEOID", "event_hour"]
        self.count_groups = count_groups or ["GEOID", "event_hour"]
        self.y_col_prob = y_col_prob
        self.y_col_cnt = y_col_cnt
        self._prob_model: Optional[Dict] = None
        self._cnt_model: Optional[Dict] = None

    def fit(self, df: pd.DataFrame) -> "HurdleLikeModel":
        dfx = df.copy()

        # olasılık hedefi: Y>0 ikili
        if self.y_col_prob in dfx.columns:
            y_occ = (pd.to_numeric(dfx[self.y_col_prob], errors="coerce").fillna(0) > 0).astype(int)
            dfx = dfx.assign(_y_occ=y_occ)
        else:
            dfx = dfx.assign(_y_occ=0)

        self._prob_model = fit_mean_by_groups(
            dfx.rename(columns={"_y_occ": "target_prob"}),
            y_col="target_prob",
            group_cols=[c for c in self.prob_groups if c in dfx.columns],
        )

        # pozitiflerde ortalama sayı
        dpos = dfx[dfx["_y_occ"] == 1].copy()
        if len(dpos) == 0:
            dpos = dfx.copy()
        self._cnt_model = fit_mean_by_groups(
            dpos.rename(columns={self.y_col_cnt: "target_cnt"}),
            y_col="target_cnt",
            group_cols=[c for c in self.count_groups if c in dpos.columns],
        )
        return self

    def predict(self, df_new: pd.DataFrame) -> pd.DataFrame:
        if self._prob_model is None or self._cnt_model is None:
            raise RuntimeError("Model fit edilmedi.")
        p = predict_mean_by_groups(self._prob_model, df_new).clip(0, 1)
        mu = predict_mean_by_groups(self._cnt_model, df_new).clip(lower=0)

        out = pd.DataFrame({
            "pred_p_occ": p.values.astype(float),
            "pred_q50":   _safe_clip_nonneg(mu).values.astype(float),  # median proxy
        }, index=df_new.index)
        out["pred_expected"] = (out["pred_p_occ"] * out["pred_q50"]).astype(float)

        # Basit belirsizlik bantları (placeholder)
        out["pred_q10"] = (out["pred_q50"] * 0.7).astype(float)
        out["pred_q90"] = (out["pred_q50"] * 1.3).astype(float)
        return out

# ---------------------------------------------------------------------
# 2) Tahmin kolonlarını güvenceye alan fonksiyon
# ---------------------------------------------------------------------
def ensure_predictions(
    df_raw: pd.DataFrame,
    model: Optional[Dict] = None,
    horizon_days: int = 90,
    slots_per_week: int = 7 * 24,
) -> pd.DataFrame:
    """
    Veri üzerinde tahmin kolonlarını **güvenle** üretir/tamamlar:

      - pred_expected (λ̂)  : yoksa frekans-baseline ya da basit Poisson baseline ile çıkarılır
      - pred_p_occ    (0..1): yoksa 1 - exp(-λ̂)
      - pred_q10/q50/q90    : yoksa Poisson kantilleri eklenir

    Zaten var olan kolonlara dokunmaz, eksikleri tamamlar.
    """
    df = df_raw.copy()

    # --- 1) λ̂ (pred_expected) üret/yamala ---
    if "pred_expected" not in df.columns:
        lam: Optional[pd.Series] = None

        # Önce frekans-baseline (GEOID/DOW/HOUR oranları) dene
        try:
            mdl = model or fit_frequency_baseline(df, horizon_days=horizon_days)
            lam = predict_expected_baseline(df, mdl, scale="per_slot", slots_per_week=slots_per_week)
        except Exception:
            lam = None

        # Olmadıysa basit Poisson baseline (GEOID×hour ort., fallback şehir/saat) dene
        if lam is None or lam.isna().all():
            if _baseline_expected is not None:
                try:
                    df_tmp = _baseline_expected(df, lookback_days=min(30, horizon_days))
                    lam = pd.to_numeric(df_tmp["pred_expected"], errors="coerce")
                except Exception:
                    lam = None

        # Hâlâ yoksa: 0 ver
        if lam is None:
            lam = pd.Series(np.zeros(len(df), dtype=float), index=df.index)

        df["pred_expected"] = _safe_clip_nonneg(lam)

    # --- 2) P(occurrence) yoksa λ̂ → p ---
    if "pred_p_occ" not in df.columns and "pred_expected" in df.columns:
        df["pred_p_occ"] = lambda_to_p_occ(df["pred_expected"].values)

    # --- 3) Kantiller/güvenlik yoksa ekle ---
    if not {"pred_q10", "pred_q50", "pred_q90"}.issubset(df.columns) and "pred_expected" in df.columns:
        df = add_poisson_uncertainty(df, lam_col="pred_expected")

    return df

# ---------------------------------------------------------------------
# 3) Genel Predictor (fallback'lı)
# ---------------------------------------------------------------------
class Predictor:
    """
    Gelecekte gerçek model (checkpoint/artifact) bağlanınca genişletilir.
    Şimdilik:
      - ensure_predictions(df) ile tahmin kolonlarını garanti eder.
      - Hiç tahmin kolonu yoksa BaselineModel ile p_proxy → pred_p_occ üretir.
    """
    def __init__(self) -> None:
        self._baseline = BaselineModel(window_days=7, use_hour=True)

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        # Girişte tahmin (veya λ̂) varsa → eksikleri tamamla
        if _has_predictions(df_raw) or ("pred_expected" in df_raw.columns):
            return ensure_predictions(df_raw)

        # Hiç tahmin yoksa: p_proxy (olasilik proxy) ile minimal çıktı
        self._baseline.fit(df_raw)
        out = self._baseline.predict(df_raw)
        if "p_proxy" in out.columns:
            out["pred_p_occ"] = out["p_proxy"].clip(0, 1)
        # λ̂ ve belirsizlik henüz yoksa, üret
        return ensure_predictions(out)
