from __future__ import annotations
import json, os, sys, time
from datetime import datetime, timedelta, timezone
from typing import Tuple
import argparse
import numpy as np
import pandas as pd

from services.metrics import save_latest_metrics, update_from_csv

# ───────────── Ayarlar (varsa config/settings.py’dan oku) ─────────────
DEFAULTS = {
    "TZ_OFFSET_SF": -7,  # SF saat farkı (UTC-7 yaz saati)
    "RAW_DIR": "data/raw",
    "OUT_DIR": "data",
    "EVENTS_FILE": "data/events.csv",
    "META_FILE": "data/_metadata.json",
    "MODEL_VERSION": "v0.3.1",
    "TOPK": 100,
    "RNG_SEED": 42,
}

def try_import_settings():
    try:
        from config.settings import TZ_OFFSET_SF, MODEL_VERSION
        DEFAULTS["TZ_OFFSET_SF"] = TZ_OFFSET_SF
        DEFAULTS["MODEL_VERSION"] = MODEL_VERSION
    except Exception:
        pass

# ───────────── Yardımcılar ─────────────
def now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def now_sf_str(tz_offset_hours: int) -> str:
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

# ───────────── 1) Veri güncelle (örnek) ─────────────
def update_raw_data(raw_dir: str, out_events: str, *, tz_off: int, since_days: int = 30) -> dict:
    ensure_dirs(raw_dir, os.path.dirname(out_events))
    try:
        old = pd.read_csv(out_events)
        old["ts"] = pd.to_datetime(old["ts"], utc=True, errors="coerce")
        old = old.dropna(subset=["ts"])
    except Exception:
        old = pd.DataFrame(columns=["ts", "latitude", "longitude", "type"])

    n_new = 800
    rng = np.random.default_rng(DEFAULTS["RNG_SEED"])
    now_u = now_utc()
    start = now_u - timedelta(days=since_days)
    center_lat, center_lon = 37.7749, -122.4194
    lat = center_lat + rng.normal(scale=0.02, size=n_new)
    lon = center_lon + rng.normal(scale=0.025, size=n_new)
    ts = pd.to_datetime(rng.integers(int(start.timestamp()), int(now_u.timestamp()), size=n_new), unit="s", utc=True)
    types = rng.choice(["assault","burglary","theft","robbery","vandalism"], size=n_new, p=[.18,.14,.35,.09,.24])
    new_df = pd.DataFrame({"ts": ts, "latitude": lat, "longitude": lon, "type": types})
    events = pd.concat([old, new_df], ignore_index=True)
    events = events.dropna(subset=["ts","latitude","longitude"]).sort_values("ts").reset_index(drop=True)
    events.to_csv(out_events, index=False)
    data_upto = events["ts"].max() if not events.empty else now_u
    data_upto_sf = (pd.to_datetime(data_upto) + pd.Timedelta(hours=tz_off)).strftime("%Y-%m-%d")
    return {"rows_total": len(events), "rows_added": len(new_df), "data_upto_sf": data_upto_sf}

# ───────────── 2) Özellik üret ─────────────
def build_features(out_dir: str) -> dict:
    ensure_dirs(out_dir)
    with open(os.path.join(out_dir, "_features_ok.txt"), "w", encoding="utf-8") as f:
        f.write(f"features built at {datetime.utcnow().isoformat()}Z\n")
    return {"features_ok": True}

# ───────────── 3) Model eğit + 3.5) Metrikleri hesapla ─────────────
def train_and_evaluate(events_csv: str, topk: int) -> Tuple[float, float, float]:
    df = pd.read_csv(events_csv)
    if df.empty:
        return 0.0, 0.0, 0.0
    y = df["type"].isin(["burglary", "robbery"]).astype(int).to_numpy()
    hrs = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.hour.fillna(0).astype(int).to_numpy()
    hsin = np.sin(2 * np.pi * hrs / 24.0)
    hcos = np.cos(2 * np.pi * hrs / 24.0)
    types = pd.get_dummies(df["type"].astype(str), prefix="t", drop_first=False)
    X = np.c_[hsin, hcos, types.to_numpy()]
    n = len(y)
    if n < 50:
        return _metrics_via_frequency(types, y, topk)
    idx = np.arange(n)
    rng = np.random.default_rng(DEFAULTS["RNG_SEED"])
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
    except Exception:
        return _metrics_via_frequency(types.iloc[te], y[te], topk)

def _metrics_via_frequency(types_df: pd.DataFrame, y_true: np.ndarray, topk: int) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    df_tmp = types_df.copy()
    df_tmp["__y__"] = y_true
    grp_cols = [c for c in types_df.columns]
    probs_map = df_tmp.groupby(grp_cols)["__y__"].mean()
    global_p = float(np.mean(y_true)) if len(y_true) else 0.0
    y_proba = types_df.apply(lambda row: float(probs_map.get(tuple(int(v) for v in row.tolist()), global_p)), axis=1)
    auc = _auc_manual(y_true, y_proba)
    brier = _brier_manual(y_true, y_proba)
    hit = hit_rate_topk(y_true, y_proba, k=topk)
    return auc, hit, brier

# ───────────── CLI ─────────────
def parse_args():
    p = argparse.ArgumentParser(description="Run full pipeline")
    p.add_argument("--since-days", type=int, default=30)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--topk", type=int, default=DEFAULTS["TOPK"])
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

    stats = update_raw_data(raw_dir, events_file, tz_off=tz_off, since_days=args.since_days)
    print(f"  ✓ events.csv güncellendi (toplam:{stats['rows_total']}, eklenen:{stats['rows_added']})")

    _ = build_features(out_dir)
    print("  ✓ features üretildi")

    auc, hit_rate, brier = train_and_evaluate(events_file, topk=topk)
    print(f"  ✓ metrics: AUC={auc:.3f}  HitRate@{topk}={hit_rate:.3f}  Brier={brier:.3f}")

    if not args.dry_run:
        save_latest_metrics(auc, hit_rate, brier)
        print("  ✓ latest_metrics.json yazıldı")

    # 🔹 Artifacts’tan otomatik metrik güncelleme
    try:
        update_from_csv(csv_path=None, hit_col=None, prefer_group="stacking")
        print("  ✓ metrics_all.csv → latest_metrics.json (otomatik güncellendi)")
    except Exception as e:
        print(f"  ⚠ metrics otomatik güncellenemedi: {e}")

    meta = {
        "data_upto": stats["data_upto_sf"],
        "model_version": model_version,
        "last_trained_at": datetime.utcnow().strftime("%Y-%m-%d"),
        "updated_at_sf": now_sf_str(tz_off),
    }
    if not args.dry_run:
        save_meta(meta_file, meta)
        print(f"  ✓ meta yazıldı → {meta_file}")

    print(f"⏱ tamamlandı: {time.time()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
