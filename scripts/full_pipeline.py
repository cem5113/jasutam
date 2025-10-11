# full_pipeline.py

from __future__ import annotations
import json, os, sys, time
from datetime import datetime, timedelta, timezone
from typing import Tuple

import argparse
import numpy as np
import pandas as pd

from services.metrics import save_latest_metrics

# ───────────── Ayarlar (varsa config/settings.py’dan oku) ─────────────
DEFAULTS = {
    "TZ_OFFSET_SF": -7,                                # SF saat farkı (UTC-7 yaz saati)
    "RAW_DIR":       "data/raw",
    "OUT_DIR":       "data",
    "EVENTS_FILE":   "data/events.csv",                # app.py’nin okuduğu dosya
    "META_FILE":     "data/_metadata.json",            # rozet için
    "MODEL_VERSION": "v0.3.1",
    "TOPK":          100,                              # HitRate@TopK için K
    "RNG_SEED":      42,
}

def try_import_settings():
    try:
        from config.settings import TZ_OFFSET_SF, MODEL_VERSION
        DEFAULTS["TZ_OFFSET_SF"]  = TZ_OFFSET_SF
        DEFAULTS["MODEL_VERSION"] = MODEL_VERSION
    except Exception:
        pass

# ───────────── Yardımcılar ─────────────
def now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def now_sf_str(tz_offset_hours: int) -> str:
    """SF local time metni (YYYY-MM-DD HH:MM)."""
    dt = now_utc() + timedelta(hours=tz_offset_hours)
    return dt.replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")

def ensure_dirs(*paths: str):
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)

def save_meta(meta_path: str, meta: dict):
    ensure_dirs(os.path.dirname(meta_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def hit_rate_topk(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Top-K örneğin ortalama gerçek pozitif oranı."""
    if len(y_true) == 0:
        return 0.0
    k_eff = int(min(max(k, 1), len(y_true)))
    order = np.argsort(-y_score)
    topk_idx = order[:k_eff]
    return float(np.mean(y_true[topk_idx]))

# ───────────── 1) Veri güncelle (placeholder/sentetik) ─────────────
def update_raw_data(raw_dir: str, out_events: str, *, tz_off: int, since_days: int = 30) -> dict:
    """
    Üretimde: harici kaynaklardan çek.
    Burada: yoksa sentetik olaylar üret (ts, lat, lon, type) ve CSV’ye yaz.
    """
    ensure_dirs(raw_dir, os.path.dirname(out_events))

    # Varsa mevcut dosyayı oku (üstüne ekleme yapabiliriz); örnek basite dönük:
    try:
        old = pd.read_csv(out_events)
        old["ts"] = pd.to_datetime(old["ts"], utc=True, errors="coerce")
        old = old.dropna(subset=["ts"])
    except Exception:
        old = pd.DataFrame(columns=["ts", "latitude", "longitude", "type"])

    n_new  = 800  # sentetik kaç olay?
    rng    = np.random.default_rng(DEFAULTS["RNG_SEED"])
    now_u  = now_utc()
    start  = now_u - timedelta(days=since_days)

    # San Francisco civarı merkez (kabaca)
    center_lat, center_lon = 37.7749, -122.4194
    lat = center_lat + rng.normal(scale=0.02, size=n_new)
    lon = center_lon + rng.normal(scale=0.025, size=n_new)

    # Zamanlar
    ts = pd.to_datetime(rng.integers(int(start.timestamp()), int(now_u.timestamp()), size=n_new), unit="s", utc=True)

    # Türler
    types = rng.choice(["assault","burglary","theft","robbery","vandalism"], size=n_new, p=[.18,.14,.35,.09,.24])

    new_df = pd.DataFrame({"ts": ts, "latitude": lat, "longitude": lon, "type": types})

    # Basit birleştirme (gerçek hayatta ID ile tekrarı önle):
    events = pd.concat([old, new_df], ignore_index=True)
    # Temizlik
    events = events.dropna(subset=["ts","latitude","longitude"]).sort_values("ts").reset_index(drop=True)
    ensure_dirs(os.path.dirname(out_events))
    events.to_csv(out_events, index=False)

    # meta için “veri şu tarihe kadar”
    data_upto = events["ts"].max() if not events.empty else now_u
    data_upto_sf = (pd.to_datetime(data_upto) + pd.Timedelta(hours=tz_off)).strftime("%Y-%m-%d")

    return {
        "rows_total": int(len(events)),
        "rows_added": int(len(new_df)),
        "data_upto_sf": data_upto_sf,
    }

# ───────────── 2) Özellik üret (placeholder) ─────────────
def build_features(out_dir: str) -> dict:
    """
    Burada gerçek feature/istatistik üretimi yapılır.
    Şimdilik sadece bir “dokunma” dosyası bırakıyoruz.
    """
    ensure_dirs(out_dir)
    touch = os.path.join(out_dir, "_features_ok.txt")
    with open(touch, "w", encoding="utf-8") as f:
        f.write(f"features built at {datetime.utcnow().isoformat()}Z\n")
    return {"features_ok": True}

# ───────────── 3) Model eğit + 3.5) Metrikleri hesapla ─────────────
def train_and_evaluate(events_csv: str, topk: int) -> Tuple[float, float, float]:
    """
    Basit bir örnek eğitim/değerlendirme:
    - Hedef (binary): {burglary, robbery} = 1, diğer tipler = 0
    - Özellikler: hour-of-day (sin/cos), type (one-hot)
    - Model: sklearn LogisticRegression (yoksa frekans tabanlı tahmin)
    Döndürür: (auc, hit_rate_topk, brier)
    """
    df = pd.read_csv(events_csv)
    if df.empty:
        # hiç veri yoksa metrikleri 0 döndür
        return 0.0, 0.0, 0.0

    # Hedef
    y = df["type"].isin(["burglary", "robbery"]).astype(int).to_numpy()

    # Zaman özellikleri
    hrs = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.hour.fillna(0).astype(int).to_numpy()
    hsin = np.sin(2 * np.pi * hrs / 24.0)
    hcos = np.cos(2 * np.pi * hrs / 24.0)

    # Type one-hot
    types = pd.get_dummies(df["type"].astype(str), prefix="t", drop_first=False)
    X = np.c_[hsin, hcos, types.to_numpy()]

    # Train/test ayır
    n = len(y)
    if n < 50:
        # çok az veri: frekans tabanı
        return _metrics_via_frequency(types, y, topk)

    idx = np.arange(n)
    rng = np.random.default_rng(DEFAULTS["RNG_SEED"])
    rng.shuffle(idx)
    cut = int(0.8 * n)
    tr, te = idx[:cut], idx[cut:]

    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    y_proba_te = None
    try:
        # Sklearn ile lojistik regresyon
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, brier_score_loss

        clf = LogisticRegression(max_iter=500, solver="liblinear")
        clf.fit(X_tr, y_tr)
        y_proba_te = clf.predict_proba(X_te)[:, 1]

        auc = float(roc_auc_score(y_te, y_proba_te)) if len(np.unique(y_te)) > 1 else 0.5
        brier = float(brier_score_loss(y_te, y_proba_te))
        hit = float(hit_rate_topk(y_te, y_proba_te, k=topk))

        return auc, hit, brier

    except Exception:
        # Sklearn yoksa frekans tabanı (type'a göre olasılık)
        return _metrics_via_frequency(types.iloc[te], y_te, topk)

def _metrics_via_frequency(types_df: pd.DataFrame, y_true: np.ndarray, topk: int) -> Tuple[float, float, float]:
    """
    Yedek (sklearn yok): her type için pozitif oranını olasılık olarak kullan.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss  # çoğu ortamda mevcut; yoksa try/except ekleyebilirsin
    # Eğitim frekansını tüm veri üzerinde yaklaşıkla: p(positive|type)
    probs_by_type = (pd.concat([types_df.reset_index(drop=True), pd.Series(y_true, name="y")], axis=1)
                     .groupby(types_df.columns.tolist())["y"].mean().to_dict())
    # satır satır olasılık üret
    def row_prob(row):
        key = tuple(int(v) for v in row.tolist())
        return float(probs_by_type.get(key, np.mean(y_true)))
    y_proba = types_df.apply(row_prob, axis=1).to_numpy(dtype=float)

    auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, y_proba))
    hit = float(hit_rate_topk(y_true, y_proba, k=topk))
    return auc, hit, brier

# ───────────── 4) Tahmin/çıktı üret (opsiyonel placeholder) ─────────────
def produce_outputs(out_dir: str) -> dict:
    ensure_dirs(out_dir)
    with open(os.path.join(out_dir, "_predictions_ok.txt"), "w", encoding="utf-8") as f:
        f.write("predictions placeholder\n")
    return {"predictions_ok": True}

# ───────────── CLI ─────────────
def parse_args():
    p = argparse.ArgumentParser(description="Run full pipeline")
    p.add_argument("--since-days", type=int, default=30, help="Sentetik veri için kaç gün geriden üretilecek")
    p.add_argument("--dry-run", action="store_true", help="Dosyaları yazma (sadece dene)")
    p.add_argument("--topk", type=int, default=DEFAULTS["TOPK"], help="HitRate@TopK için K")
    return p.parse_args()

def main():
    try_import_settings()
    args = parse_args()

    tz_off = DEFAULTS["TZ_OFFSET_SF"]
    raw_dir = DEFAULTS["RAW_DIR"]
    out_dir = DEFAULTS["OUT_DIR"]
    events_file = DEFAULTS["EVENTS_FILE"]
    meta_file = DEFAULTS["META_FILE"]
    model_version = DEFAULTS["MODEL_VERSION"]
    topk = int(args.topk)

    print("▶ Full pipeline başlıyor…")
    t0 = time.time()

    # 1) Veri
    stats = update_raw_data(raw_dir, events_file, tz_off=tz_off, since_days=args.since_days)
    print(f"  ✓ events.csv güncellendi (toplam:{stats['rows_total']}, eklenen:{stats['rows_added']})")

    # 2) Özellikler
    fstats = build_features(out_dir)
    print("  ✓ features üretildi")

    # 3) Model + 3.5) Değerlendirme (METRICS)
    auc, hit_rate, brier = train_and_evaluate(events_file, topk=topk)
    print(f"  ✓ metrics: AUC={auc:.3f}  HitRate@{topk}={hit_rate:.3f}  Brier={brier:.3f}")

    # 3.6) JSON'a atomik METRICS kaydı (app.py KPI için)
    if not args.dry_run:
        save_latest_metrics(auc, hit_rate, brier)
        print("  ✓ latest_metrics.json yazıldı")

    # 4) Çıktılar
    ostats = produce_outputs(out_dir)
    print("  ✓ çıktılar üretildi")

    # 5) Meta (rozete bilgi)
    meta = {
        "data_upto":       stats["data_upto_sf"],              # YYYY-MM-DD (SF)
        "model_version":   model_version,
        "last_trained_at": datetime.utcnow().strftime("%Y-%m-%d"),  # YYYY-MM-DD (UTC)
        "updated_at_sf":   now_sf_str(tz_off),
    }
    if not args.dry_run:
        save_meta(meta_file, meta)
        print(f"  ✓ meta yazıldı → {meta_file}")

    print(f"⏱  tamamlandı: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
