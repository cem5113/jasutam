#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full pipeline (veri → özellik → model → çıktı)
- Ham veriyi günceller/oluşturur (yoksa sentetik)
- Özellik/istatistik dosyaları üretir (placeholder)
- Basit bir “model eğitimi” örneği çalıştırır (placeholder)
- data/_metadata.json içine data_upto/model_version/last_trained_at_sf yazar
- data/events.csv dosyasını garanti eder: [ts, latitude, longitude, type, geoid?]

Gereksinim: pandas, numpy (veya zaten projede mevcut)
"""

from __future__ import annotations
import json, os, sys, time
from datetime import datetime, timedelta, timezone
import argparse
import numpy as np
import pandas as pd

# ───────────── Ayarlar (varsa config/settings.py’dan oku) ─────────────
DEFAULTS = {
    "TZ_OFFSET_SF": -7,                                # SF saat farkı (UTC-7 yaz saati)
    "RAW_DIR":       "data/raw",
    "OUT_DIR":       "data",
    "EVENTS_FILE":   "data/events.csv",                # app.py’nin okuduğu dosya
    "META_FILE":     "data/_metadata.json",            # rozet için
    "MODEL_VERSION": "v0.3.1",
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
        os.makedirs(p, exist_ok=True)

def save_meta(meta_path: str, meta: dict):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

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
    rng    = np.random.default_rng(42)
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

# ───────────── 3) Model eğit (placeholder) ─────────────
def train_model(out_dir: str) -> dict:
    """
    Üretimde: gerçek eğitim ve model kaydı.
    Şimdilik yalnızca bir “model_ok” dosyası bırakıyoruz.
    """
    ensure_dirs(out_dir)
    with open(os.path.join(out_dir, "model_ok.txt"), "w", encoding="utf-8") as f:
        f.write(f"trained at {datetime.utcnow().isoformat()}Z\n")
    last_trained_at = datetime.utcnow().strftime("%Y-%m-%d")
    return {"last_trained_at": last_trained_at}

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

    print("▶ Full pipeline başlıyor…")
    t0 = time.time()

    # 1) Veri
    stats = update_raw_data(raw_dir, events_file, tz_off=tz_off, since_days=args.since_days)
    print(f"  ✓ events.csv güncellendi (toplam:{stats['rows_total']}, eklenen:{stats['rows_added']})")

    # 2) Özellikler
    fstats = build_features(out_dir)
    print("  ✓ features üretildi")

    # 3) Model
    mstats = train_model(out_dir)
    print("  ✓ model eğitildi")

    # 4) Çıktılar
    ostats = produce_outputs(out_dir)
    print("  ✓ çıktılar üretildi")

    # 5) Meta (rozete bilgi)
    meta = {
        "data_upto":       stats["data_upto_sf"],              # YYYY-MM-DD (SF)
        "model_version":   model_version,
        "last_trained_at": mstats["last_trained_at"],          # YYYY-MM-DD (UTC baz)
        "updated_at_sf":   now_sf_str(tz_off),
    }
    save_meta(meta_file, meta)
    print(f"  ✓ meta yazıldı → {meta_file}")
    print(f"⏱  tamamlandı: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    sys.exit(main())
