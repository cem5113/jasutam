from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Iterable
import numpy as np
import pandas as pd
import importlib
import types

# ==== Uncertainty fonksiyonlarını sağlam şekilde içe al ====

def _import_uncertainty():
    """uncertainty modülünü hem göreli hem mutlak yollarla dener, bulunursa döndürür."""
    # 1) Göreli dene
    try:
        from . import uncertainty as _unc  # type: ignore
        return _unc
    except ImportError:
        pass

    # 2) Mutlak dene
    try:
        _unc = importlib.import_module("sutam.models.uncertainty")
        return _unc
    except ImportError:
        pass

    raise ImportError(
        "uncertainty modülü bulunamadı. Kontrol et:\n"
        " - sutam/models/uncertainty.py mevcut mu?\n"
        " - sutam/ ve sutam/models/ içinde __init__.py var mı?\n"
        " - Projeyi kökten çalıştırıyor musun? (cd /mount/src; streamlit run sutam/app.py)"
    )

_unc = _import_uncertainty()

# 3) İsimleri modülden çek ve doğrula
poisson_quantiles = getattr(_unc, "poisson_quantiles", None)
prob_at_least_one = getattr(_unc, "prob_at_least_one", None)

if not callable(poisson_quantiles) or not callable(prob_at_least_one):
    available = sorted([n for n in dir(_unc) if not n.startswith("_")])
    raise ImportError(
        "uncertainty modülü bulundu ama beklenen fonksiyon adları yok:\n"
        " - Beklenen: poisson_quantiles, prob_at_least_one\n"
        f" - Bulunan isimler: {available}\n"
        "Lütfen uncertainty.py içindeki fonksiyonların adlarını birebir bu şekilde tanımlayın."
    )

# ---------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------
def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    date/datetime kolonlarını güvenle datetime'a çevirir.
    'date' yoksa 'datetime' üzerinden üretmeyi dener.
    Ayrıca event_hour / day_of_week kolonlarını (yoksa) türetir.
    """
    out = df.copy()
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
    else:
        out["date"] = pd.to_datetime(out.get("date", pd.NaT), errors="coerce")

    if "event_hour" not in out.columns:
        if "date" in out.columns:
            out["event_hour"] = out["date"].dt.hour.fillna(0).astype(int)
        else:
            out["event_hour"] = 0

    if "day_of_week" not in out.columns:
        if "date" in out.columns:
            out["day_of_week"] = out["date"].dt.dayofweek
        else:
            out["day_of_week"] = 0

    return out


def _y_col(df: pd.DataFrame) -> str:
    """
    Sayım temelli hedef kolon; bulunamazsa her satırı 1 sayar.
    """
    for c in ("crime_count", "count", "y", "events"):
        if c in df.columns:
            return c
    df = df.copy()
    df["_ones"] = 1
    return "_ones"


# Eski adla geriye dönük uyumluluk
_ycol = _y_col


def _pick_y(df: pd.DataFrame) -> str:
    """
    Skor/etiket için tercih sırası:
      1) pred_expected
      2) pred_p_occ
      3) crime_count
      4) ilk sayısal kolon (fallback) ya da ilk kolon
    """
    for c in ("pred_expected", "pred_p_occ", "crime_count"):
        if c in df.columns:
            return c
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else df.columns[0]


# ---------------------------------------------------------------------
# 0) Kolay kullanım: Poisson tabanlı baseline (UI ile tam uyumlu)
# ---------------------------------------------------------------------
def baseline_expected(df_raw: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """
    Çok hafif Poisson baseline:
      - Son 'lookback_days' içinde GEOID×hour ortalama olay sayısı -> pred_expected
      - Fallback: GEOID ort. → (yoksa) şehir ort. + saat etkisi
      - pred_p_occ = 1 - exp(-pred_expected)
      - pred_q10/q50/q90 = Poisson kuantilleri
    """
    df = _ensure_time(df_raw)
    ycol = _y_col(df)

    # lookback penceresi
    if "date" in df.columns and df["date"].notna().any():
        dmax = df["date"].max()
        dmin = dmax - pd.Timedelta(days=lookback_days - 1)
        hist = df[(df["date"] >= dmin) & (df["date"] <= dmax)].copy()
    else:
        hist = df.copy()

    # GEOID×hour ortalamaları
    geo_hour = (
        hist.groupby(["GEOID", "event_hour"], as_index=False)[ycol]
        .mean()
        .rename(columns={ycol: "mu_geo_hour"})
    )
    # GEOID ortalamaları
    geo_mu = (
        hist.groupby("GEOID", as_index=False)[ycol]
        .mean()
        .rename(columns={ycol: "mu_geo"})
    )
    # şehir ortalaması
    g_city = float(pd.to_numeric(hist[ycol], errors="coerce").fillna(0).mean())

    # saat etkisi (varsa)
    by_hour = (
        hist.groupby("event_hour", as_index=False)[ycol]
        .mean()
        .rename(columns={ycol: "mu_hour"})
    )
    hour_map = by_hour.set_index("event_hour")["mu_hour"].to_dict() if len(by_hour) else {}

    out = df.copy()
    out = out.merge(geo_hour, on=["GEOID", "event_hour"], how="left")
    out = out.merge(geo_mu, on="GEOID", how="left")

    # hiyerarşik fallback
    out["pred_expected"] = np.where(
        out["mu_geo_hour"].notna(),
        out["mu_geo_hour"],
        np.where(
            out["mu_geo"].notna(),
            out["mu_geo"],
            pd.Series([hour_map.get(int(h), g_city) for h in out["event_hour"]], index=out.index),
        ),
    ).astype(float)

    # Olasılık & Poisson kuantilleri
    out["pred_p_occ"] = out["pred_expected"].map(lambda lam: prob_at_least_one(float(max(lam, 0.0))))
    qs = out["pred_expected"].map(lambda lam: poisson_quantiles(float(max(lam, 0.0))))
    out["pred_q10"] = qs.map(lambda d: d.get("q10", 0))
    out["pred_q50"] = qs.map(lambda d: d.get("q50", 0))
    out["pred_q90"] = qs.map(lambda d: d.get("q90", 0))

    # temizlik
    out.drop(columns=["mu_geo_hour", "mu_geo"], inplace=True, errors="ignore")
    return out


# ---------------------------------------------------------------------
# 1) Frekans-temelli baseline (λ haftalık oran)
# ---------------------------------------------------------------------
def fit_frequency_baseline(
    df_raw: pd.DataFrame,
    horizon_days: int = 90,
    min_obs_geoid: int = 10,
    alpha: float = 0.25,  # Laplace benzeri yumuşatma
) -> Dict:
    """
    Basit frekans temelli model:
      λ_weekly(GEOID, DOW, HOUR) ≈ (sayım + alpha) / (hafta_sayısı)
    Geri dönüş anahtarları:
      - 'rate3': {(geoid,dow,hour)->λ_weekly}
      - 'rate2': {(geoid,dow)->λ_weekly}
      - 'rate1': {geoid->λ_weekly}
      - 'rate0': float (şehir geneli, weekly)
      - 'weeks': float
      - 'horizon_days': int
    """
    df = _ensure_time(df_raw)

    if "date" in df.columns and df["date"].notna().any():
        dmax = pd.to_datetime(df["date"], errors="coerce").max()
        dmin = dmax - pd.Timedelta(days=horizon_days - 1)
        df = df[(df["date"] >= dmin) & (df["date"] <= dmax)].copy()

    y = _y_col(df)
    if y not in df.columns:
        df[y] = 1

    days_active = float(df["date"].dt.normalize().nunique()) if "date" in df.columns else float(horizon_days)
    weeks = max(1.0, days_active / 7.0)

    # şehir geneli weekly oran
    rate0 = float(df[y].sum() + alpha) / weeks

    def grp_rate(keys: List[str]) -> Dict[Tuple, float]:
        g = df.groupby(keys, as_index=False)[y].sum()
        out: Dict[Tuple, float] = {}
        for _, row in g.iterrows():
            key_vals = []
            for k in keys:
                v = row[k]
                key_vals.append(str(v) if k == "GEOID" and v is not None else v)
            out[tuple(key_vals)] = float(row[y] + alpha) / weeks
        return out

    rate1 = grp_rate(["GEOID"]) if "GEOID" in df.columns else {}
    rate2 = grp_rate(["GEOID", "day_of_week"]) if "GEOID" in df.columns else {}
    rate3 = grp_rate(["GEOID", "day_of_week", "event_hour"]) if "GEOID" in df.columns else {}

    # Az gözlemi olan GEOID'lere zayıf itme
    if "GEOID" in df.columns:
        counts_geoid = df.groupby("GEOID", as_index=False)[y].sum().set_index("GEOID")[y].to_dict()
        for g, cnt in counts_geoid.items():
            if cnt < min_obs_geoid and g in rate1:
                rate1[g] = 0.5 * rate1[g] + 0.5 * rate0

    return {
        "rate3": rate3,
        "rate2": rate2,
        "rate1": rate1,
        "rate0": rate0,
        "weeks": weeks,
        "horizon_days": int(horizon_days),
    }


def predict_expected_baseline(
    df_raw: pd.DataFrame,
    model: Dict,
    scale: str = "per_slot",
    slots_per_week: int = 7 * 24,
) -> pd.Series:
    """
    Her satır için beklenen olay sayısı λ̂ döndürür.
    Öncelik: (GEOID,DOW,HOUR) → (GEOID,DOW) → (GEOID) → şehir.

    scale:
      - "weekly": model oranını haftalık olarak döndürür
      - "per_slot": haftalık oranı 'slots_per_week' sayısına bölerek per-slot (≈ saatlik) beklenen üretir
    """
    df = _ensure_time(df_raw)
    r3, r2, r1 = model.get("rate3", {}), model.get("rate2", {}), model.get("rate1", {})
    r0 = float(model.get("rate0", 0.0))

    def one_row(i) -> float:
        geoid = str(df.at[i, "GEOID"]) if "GEOID" in df.columns else None
        dow = int(df.at[i, "day_of_week"]) if "day_of_week" in df.columns else None
        hour = int(df.at[i, "event_hour"]) if "event_hour" in df.columns else None
        lam_week = float(r3.get((geoid, dow, hour)) or r2.get((geoid, dow)) or r1.get(geoid) or r0)
        if scale == "weekly":
            return lam_week
        return lam_week / float(slots_per_week or 168)

    return pd.Series({i: one_row(i) for i in df.index}, index=df.index, dtype=float).rename("pred_expected")


# ---------------------------------------------------------------------
# 2) Pencere-temelli baseline (p_proxy; 0..1)
# ---------------------------------------------------------------------
class BaselineModel:
    """
    Çok hafif bir fallback:
      - Son N gün içinde GEOID bazında toplamı alır
      - Max-normalize ederek [0..1] 'p_proxy' üretir
      - (opsiyonel) Saat etkisi: event_hour varsa GEOID×hour ort. ile çarpar ve yeniden normalize eder
    Gerçek model gelene kadar UI’nin çalışmasını garanti eder.
    """

    def __init__(self, window_days: int = 7, use_hour: bool = True):
        self.window_days = int(window_days)
        self.use_hour = bool(use_hour)
        self.table_geo: Optional[pd.DataFrame] = None
        self.table_geo_hour: Optional[pd.DataFrame] = None

    def fit(self, df_raw: pd.DataFrame) -> "BaselineModel":
        df = _ensure_time(df_raw)
        y = _y_col(df)

        dmax = pd.to_datetime(df["date"], errors="coerce").dropna().max()
        if pd.isna(dmax):
            dsub = df.copy()
        else:
            dmin = dmax - pd.Timedelta(days=self.window_days - 1)
            d = pd.to_datetime(df["date"], errors="coerce")
            dsub = df[(d >= dmin) & (d <= dmax)].copy()

        if "GEOID" not in dsub.columns:
            self.table_geo = pd.DataFrame(columns=["GEOID", "p_proxy"])
            self.table_geo_hour = pd.DataFrame(columns=["GEOID", "event_hour", "hour_weight"])
            return self

        g = (
            dsub.groupby("GEOID", as_index=False)[y]
            .sum()
            .rename(columns={y: "recent_sum"})
        )
        mx = float(g["recent_sum"].max() or 1.0)
        g["p_proxy"] = (g["recent_sum"] / (mx if mx > 0 else 1.0)).clip(0, 1)
        self.table_geo = g[["GEOID", "p_proxy"]].copy()

        # Saat etkisi (opsiyonel)
        if self.use_hour and "event_hour" in dsub.columns:
            gh = (
                dsub.groupby(["GEOID", "event_hour"], as_index=False)[y]
                .mean()
                .rename(columns={y: "hour_mean"})
            )
            # GEOID içinde normalize
            gh["hour_weight"] = gh.groupby("GEOID")["hour_mean"].transform(
                lambda s: (s / (s.max() if s.max() > 0 else 1.0)).clip(0, 1)
            )
            self.table_geo_hour = gh[["GEOID", "event_hour", "hour_weight"]].copy()
        else:
            self.table_geo_hour = None

        return self

    def predict(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Girdi df’ye p_proxy ekler (ve varsa saat düzeltmesini uygular).
        Dönen DataFrame: df + ['p_proxy'] (0..1)
        """
        if self.table_geo is None:
            # fit edilmemişse hızlı bir fit yap
            self.fit(df_raw)

        df = _ensure_time(df_raw)
        out = df.merge(self.table_geo, on="GEOID", how="left") if "GEOID" in df.columns else df.copy()
        if "p_proxy" not in out.columns:
            out["p_proxy"] = 0.0

        if self.table_geo_hour is not None and "event_hour" in out.columns and "GEOID" in out.columns:
            out = out.merge(self.table_geo_hour, on=["GEOID", "event_hour"], how="left")
            out["p_proxy"] = (out["p_proxy"] * out["hour_weight"].fillna(1.0)).clip(0, 1)
            out.drop(columns=["hour_weight"], inplace=True, errors="ignore")

        return out


# ---------------------------------------------------------------------
# 3) Grup-ortalaması baseline (GEOID × event_hour vb.)
# ---------------------------------------------------------------------
def fit_mean_by_groups(
    df: pd.DataFrame,
    y_col: Optional[str] = None,
    group_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Basit baseline: grup ortalamaları (ör. GEOID × event_hour) + global fallback.
    Dönüş: {'y_col','groups','table','global_mean'}
    """
    if y_col is None:
        y_col = _pick_y(df)
    if group_cols is None:
        group_cols = [c for c in ["GEOID", "event_hour"] if c in df.columns]

    dfx = df.copy().dropna(subset=[y_col])
    if not group_cols:
        mu = float(np.nanmean(dfx[y_col])) if len(dfx) else 0.0
        return {"y_col": y_col, "groups": [], "table": pd.DataFrame(), "global_mean": mu}

    table = dfx.groupby(group_cols, as_index=False)[y_col].mean().rename(columns={y_col: "mean"})
    glob = float(np.nanmean(dfx[y_col])) if len(dfx) else 0.0
    return {"y_col": y_col, "groups": group_cols, "table": table, "global_mean": glob}


def predict_mean_by_groups(model: Dict, df_new: pd.DataFrame) -> pd.Series:
    """
    Eğitimdeki grup ortalamasını yeni veriye uygular; bulunamazsa global_mean.
    Dönüş: pd.Series (isim: baseline_{y_col})
    """
    y_col: str = model["y_col"]
    groups: List[str] = model["groups"]
    table: pd.DataFrame = model["table"]
    glob: float = model["global_mean"]

    if not groups or table.empty:
        return pd.Series([glob] * len(df_new), index=df_new.index, name=f"baseline_{y_col}")

    out = df_new.copy()
    out = out.merge(table, on=groups, how="left")
    pred = out["mean"].fillna(glob)
    return pd.to_numeric(pred, errors="coerce").fillna(glob).rename(f"baseline_{y_col}")
