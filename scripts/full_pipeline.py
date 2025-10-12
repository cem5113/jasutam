# scripts/full_pipeline.py
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────── Opsiyonel bağımlılıklar ───────────────────────────
# Bu modüller projede varsa kullanılacak; yoksa güvenli düşüş uygulanır.
try:
    from dataio.loaders import load_sf_crime_latest, _validate_schema, RESULTS_DIR as LOADER_RESULTS_DIR
except Exception:  # noqa: BLE001
    load_sf_crime_latest = None
    _validate_schema = None
    LOADER_RESULTS_DIR = None

try:
    from services.metrics import (
        build_basic_metadata as svc_build_basic_metadata,
        save_latest_metrics as svc_save_latest_metrics,
        update_from_csv as svc_update_from_csv,
    )
except Exception:  # noqa: BLE001
    svc_build_basic_metadata = None
    svc_save_latest_metrics = None
    svc_update_from_csv = None

# ────────────────────────────── Logging ayarı ──────────────────────────────────
LOG = logging.getLogger("full_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ───────────────────────────── Konfig & yardımcılar ────────────────────────────
@dataclass(frozen=True)
class Config:
    tz_offset_sf: int = -7  # SF DST: UTC-7 varsayılan
    out_dir: Path = Path("data")               # genel çıktı kökü
    raw_dir: Path = Path("data/raw")           # sentetik/ham olay deposu (opsiyonel)
    events_csv: Path = Path("data/events.csv") # sentetik/ham olay dosyası (opsiyonel)
    meta_file: Path = Path("data/_metadata.json")
    model_version: str = "v0.4.0"
    rng_seed: int = 42
    topk: int = 100

def try_import_settings(cfg: Config) -> Config:
    """config.settings varsa bazı alanları override eder."""
    try:
        from config.settings import TZ_OFFSET_SF, MODEL_VERSION  # type: ignore[attr-defined]
        return Config(
            tz_offset_sf=int(getattr(sys.modules["config.settings"], "TZ_OFFSET_SF", cfg.tz_offset_sf)),
            out_dir=cfg.out_dir,
            raw_dir=cfg.raw_dir,
            events_csv=cfg.events_csv,
            meta_file=cfg.meta_file,
            model_version=str(getattr(sys.modules["config.settings"], "MODEL_VERSION", cfg.model_version)),
            rng_seed=cfg.rng_seed,
            topk=cfg.topk,
        )
    except Exception:  # noqa: BLE001
        return cfg

def now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def now_sf_str(tz_off: int) -> str:
    dt = now_utc() + timedelta(hours=tz_off)
    return dt.replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        if p is not None:
            Path(p).mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: dict) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ───────────────────────────── Metrik yardımcıları ─────────────────────────────
def hit_rate_topk(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    k_eff = int(min(max(k, 1), len(y_true)))
    order = np.argsort(-y_score)
    topk_idx = order[:k_eff]
    return float(np.mean(y_true[topk_idx]))

def _auc_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc) if np.isfinite(auc) else 0.5

def _brier_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    return float(np.mean((y_score - y_true) ** 2))

# ─────────────────────────────── Veri kaynakları ───────────────────────────────
def fetch_latest_df(results_dir_hint: Optional[Path]) -> tuple[pd.DataFrame, str, Path]:
    """
    Öncelik:
      1) dataio.loaders.load_sf_crime_latest() → gerçek veri
      2) events.csv (sentetik/ham) → pipeline içi eğitim/örnek
    """
    # 1) Loader varsa kullan
    if callable(load_sf_crime_latest):
        df, src = load_sf_crime_latest()
        out_dir = results_dir_hint or Path("results")
        if LOADER_RESULTS_DIR is not None:
            out_dir = LOADER_RESULTS_DIR
        ensure_dirs(out_dir)
        return df, f"loader:{src}", out_dir

    # 2) Sentetik/ham olaylar
    out_dir = Path("data")
    ensure_dirs(out_dir)
    ev = out_dir / "events.csv"
    if ev.exists():
        df = pd.read_csv(ev)
        return df, f"csv:{ev}", out_dir
    # Yoksa boş döner (ileride sentetik üretilebilir)
    return pd.DataFrame(), "none", out_dir

def validate_schema_if_possible(df: pd.DataFrame) -> None:
    if callable(_validate_schema):
        ok, missing = _validate_schema(df)
        if not ok:
            LOG.warning("Şema uyarısı - eksik kolonlar: %s", missing)

# ───────────────────────────── Sentetik veri opsiyonu ──────────────────────────
def update_raw_data_synthetic(cfg: Config, since_days: int = 30) -> dict:
    """Eğitim/örnek amaçlı sentetik olay üretimi (opsiyonel)."""
    ensure_dirs(cfg.raw_dir, cfg.events_csv.parent)
    try:
        old = pd.read_csv(cfg.events_csv)
        old["ts"] = pd.to_datetime(old["ts"], utc=True, errors="coerce")
        old = old.dropna(subset=["ts"])
    except Exception:  # noqa: BLE001
        old = pd.DataFrame(columns=["ts", "latitude", "longitude", "type"])

    n_new = 800
    rng = np.random.default_rng(cfg.rng_seed)
    now_u = now_utc()
    start = now_u - timedelta(days=since_days)
    center_lat, center_lon = 37.7749, -122.4194
    lat = center_lat + rng.normal(scale=0.02, size=n_new)
    lon = center_lon + rng.normal(scale=0.025, size=n_new)
    ts = pd.to_datetime(rng.integers(int(start.timestamp()), int(now_u.timestamp()), size=n_new), unit="s", utc=True)
    types = rng.choice(["assault", "burglary", "theft", "robbery", "vandalism"], size=n_new,
                       p=[.18, .14, .35, .09, .24])
    new_df = pd.DataFrame({"ts": ts, "latitude": lat, "longitude": lon, "type": types})
    events = pd.concat([old, new_df], ignore_index=True)
    events = events.dropna(subset=["ts", "latitude", "longitude"]).sort_values("ts").reset_index(drop=True)
    events.to_csv(cfg.events_csv, index=False)
    data_upto = events["ts"].max() if not events.empty else now_u
    data_upto_sf = (pd.to_datetime(data_upto) + pd.Timedelta(hours=cfg.tz_offset_sf)).strftime("%Y-%m-%d")
    return {"rows_total": len(events), "rows_added": len(new_df), "data_upto_sf": data_upto_sf}

# ───────────────────────────── Özellik üretim yeri ─────────────────────────────
def build_features_placeholder(out_dir: Path) -> dict:
    ensure_dirs(out_dir)
    (out_dir / "_features_ok.txt").write_text(f"features built at {datetime.utcnow().isoformat()}Z\n", encoding="utf-8")
    return {"features_ok": True}

# ─────────────────────────────── Eğitim & metrikler ────────────────────────────
def train_and_evaluate_generic(df: pd.DataFrame, topk: int, rng_seed: int) -> Tuple[float, float, float]:
    """
    Tercihen gerçek veride:
      - Hedef: 'Y_label' (0/1)
      - Basit özellik: saat sin/cos + kategorik(örn. crime_category varsa)
    Aksi halde sentetik 'type' alanı üzerinde proxy hedef: burglary/robbery → 1
    """
    if df is None or df.empty:
        return 0.0, 0.0, 0.0

    # Zaman kolonu adları çeşitlenebilir:
    ts_col = None
    for cand in ("ts", "timestamp", "datetime", "occurred_at", "date"):
        if cand in df.columns:
            ts_col = cand
            break

    # Hedef belirleme
    if "Y_label" in df.columns:
        y = df["Y_label"].astype(int).to_numpy()
    elif "type" in df.columns:
        y = df["type"].astype(str).isin(["burglary", "robbery"]).astype(int).to_numpy()
    else:
        # hedef yoksa öğrenilecek bir şey de yok; rastgele skor
        y = np.zeros(len(df), dtype=int)

    # Basit özellikler
    if ts_col:
        hrs = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.hour.fillna(0).astype(int).to_numpy()
    else:
        hrs = np.zeros(len(df), dtype=int)
    hsin = np.sin(2 * np.pi * hrs / 24.0)
    hcos = np.cos(2 * np.pi * hrs / 24.0)

    cat_cols = []
    for cand in ("type", "crime_category", "offense", "category"):
        if cand in df.columns:
            cat_cols.append(cand)
            break
    if cat_cols:
        types = pd.get_dummies(df[cat_cols[0]].astype(str), prefix="c", drop_first=False)
        X = np.c_[hsin, hcos, types.to_numpy()]
    else:
        X = np.c_[hsin, hcos]

    n = len(y)
    if n < 50 or len(np.unique(y)) < 2:
        # frekans tabanlı yedek (etiket dağılımı zayıfsa)
        y_proba = np.clip((y.mean() if n else 0.0) * np.ones(n), 0, 1)
        auc = _auc_manual(y, y_proba)
        brier = _brier_manual(y, y_proba)
        hit = hit_rate_topk(y, y_proba, k=topk)
        return auc, hit, brier

    idx = np.arange(n)
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(idx)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, brier_score_loss

        clf = LogisticRegression(max_iter=500, solver="liblinear")
        clf.fit(X_tr, y_tr)
        y_proba_te = clf.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, y_proba_te)) if len(np.unique(y_te)) > 1 else 0.5
        brier = float(brier_score_loss(y_te, y_proba_te))
        hit = float(hit_rate_topk(y_te, y_proba_te, k=topk))
        return auc, hit, brier
    except Exception:  # noqa: BLE001
        # sklearn yoksa: rank-tabanlı approx
        # kategori frekansı → olasılık
        if cat_cols:
            types_df = pd.get_dummies(df.iloc[te][cat_cols[0]].astype(str), prefix="c", drop_first=False)
            # grup olasılıkları
            tmp = pd.get_dummies(df.iloc[tr][cat_cols[0]].astype(str), prefix="c", drop_first=False)
            tmp["__y__"] = y_tr
            probs_map = tmp.groupby(list(tmp.columns[:-1]))["__y__"].mean()
            global_p = float(y_tr.mean()) if len(y_tr) else 0.0
            def row_p(row):
                key = tuple(int(v) for v in row.tolist())
                return float(probs_map.get(key, global_p))
            y_proba = types_df.apply(row_p, axis=1).to_numpy()
        else:
            # yalnız saat → global p
            y_proba = np.full_like(y_te, fill_value=float(y_tr.mean()) if len(y_tr) else 0.0, dtype=float)
        auc = _auc_manual(y_te, y_proba)
        brier = _brier_manual(y_te, y_proba)
        hit = hit_rate_topk(y_te, y_proba, k=topk)
        return auc, hit, brier

# ─────────────────────────────── Sonuç yazıcıları ──────────────────────────────
def write_latest_csv(df: pd.DataFrame, out_dir: Path, name: str = "sf_crime_latest.csv") -> Path:
    ensure_dirs(out_dir)
    out_csv = out_dir / name
    df.to_csv(out_csv, index=False)
    return out_csv

def write_metadata(df: pd.DataFrame, src: str, out_dir: Path, cfg: Config) -> Path:
    meta_path = out_dir / "metadata.json"
    if callable(svc_build_basic_metadata):
        meta = svc_build_basic_metadata(df, src, meta_path)
    else:
        # minimal meta
        date_cols = [c for c in df.columns if "date" in c.lower() or c in ("ts", "timestamp", "datetime")]
        if date_cols:
            s = pd.to_datetime(df[date_cols[0]], errors="coerce")
            dmin = str(pd.to_datetime(s.min()).date()) if s.notna().any() else None
            dmax = str(pd.to_datetime(s.max()).date()) if s.notna().any() else None
        else:
            dmin = dmax = None
        meta = {
            "rows": int(len(df)),
            "date_min": dmin,
            "date_max": dmax,
            "src": src,
        }
        save_json(meta_path, meta)
    LOG.info("[OK] src=%s rows=%s date=%s→%s", src, meta.get("rows"), meta.get("date_min"), meta.get("date_max"))
    return meta_path

def save_metrics(auc: float, hit: float, brier: float) -> None:
    if callable(svc_save_latest_metrics):
        svc_save_latest_metrics(auc, hit, brier)
    else:
        save_json(Path("data/latest_metrics.json"), {"auc": auc, "hit_rate": hit, "brier": brier})

def try_update_from_csv() -> None:
    if callable(svc_update_from_csv):
        try:
            svc_update_from_csv(csv_path=None, hit_col=None, prefer_group="stacking")
        except Exception as e:  # noqa: BLE001
            LOG.warning("metrics otomatik güncellenemedi: %s", e)

def write_meta_compact(cfg: Config, data_upto_sf: Optional[str]) -> None:
    meta = {
        "data_upto": data_upto_sf,
        "model_version": cfg.model_version,
        "last_trained_at": datetime.utcnow().strftime("%Y-%m-%d"),
        "updated_at_sf": now_sf_str(cfg.tz_offset_sf),
    }
    save_json(cfg.meta_file, meta)

# ─────────────────────────────────── CLI ───────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full crime pipeline (load → features → train → metrics → meta)")
    p.add_argument("--since-days", type=int, default=30, help="Sentetik olay üretim ufku (gerekirse)")
    p.add_argument("--dry-run", action="store_true", help="Dosya yazmadan çalıştır")
    p.add_argument("--topk", type=int, default=None, help="HitRate@K için K")
    p.add_argument("--out-dir", type=Path, default=None, help="Çıktı klasörü override")
    p.add_argument("--force-synthetic", action="store_true", help="Loader yoksa sentetik olay üret")
    return p.parse_args()

# ────────────────────────────────── Main ───────────────────────────────────────
def main() -> int:
    cfg = try_import_settings(Config())
    args = parse_args()
    if args.topk is not None:
        cfg = Config(
            tz_offset_sf=cfg.tz_offset_sf,
            out_dir=cfg.out_dir,
            raw_dir=cfg.raw_dir,
            events_csv=cfg.events_csv,
            meta_file=cfg.meta_file,
            model_version=cfg.model_version,
            rng_seed=cfg.rng_seed,
            topk=int(args.topk),
        )

    # Çıktı klasörü önceliği: dataio.loaders.RESULTS_DIR → CLI --out-dir → cfg.out_dir
    results_dir_hint = LOADER_RESULTS_DIR if LOADER_RESULTS_DIR is not None else (args.out_dir or cfg.out_dir)

    LOG.info("▶ Pipeline başlıyor…")
    t0 = time.time()

    # 0) (Opsiyonel) Sentetik güncelleme
    data_upto_sf = None
    if args.force_synthetic or load_sf_crime_latest is None:
        stats = update_raw_data_synthetic(cfg, since_days=int(args.since_days))
        data_upto_sf = stats["data_upto_sf"]
        LOG.info("  ✓ events.csv güncellendi (toplam:%s, eklenen:%s)", stats["rows_total"], stats["rows_added"])

    # 1) Veri çek
    df, src, out_dir = fetch_latest_df(results_dir_hint)
    if df.empty:
        LOG.warning("Veri bulunamadı. Yalnızca meta/çıktı iskeleti yazılacak.")
    validate_schema_if_possible(df)

    # 2) Çıktı CSV
    if not args.dry_run and not df.empty:
        out_csv = write_latest_csv(df, out_dir, name="sf_crime_latest.csv")
        LOG.info("  ✓ yazıldı: %s", out_csv)

    # 3) Özellik inşası (placeholder)
    _ = build_features_placeholder(out_dir)
    LOG.info("  ✓ features üretildi")

    # 4) Eğitim & metrikler
    auc, hit_rate, brier = train_and_evaluate_generic(df, topk=cfg.topk, rng_seed=cfg.rng_seed)
    LOG.info("  ✓ metrics: AUC=%.3f  HitRate@%d=%.3f  Brier=%.3f", auc, cfg.topk, hit_rate, brier)

    if not args.dry_run:
        save_metrics(auc, hit_rate, brier)
        LOG.info("  ✓ latest_metrics.json güncellendi")
        try_update_from_csv()

    # 5) Metadata
    if not args.dry_run:
        _ = write_metadata(df, src, out_dir, cfg)
        write_meta_compact(cfg, data_upto_sf)
        LOG.info("  ✓ meta yazıldı → %s", cfg.meta_file)

    LOG.info("⏱ tamamlandı: %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
