# services/metrics.py
from __future__ import annotations
"""
SUTAM - Dinamik Model Performans / KPI Servisi
-----------------------------------------------
- latest_metrics.json dosyasını okur/yazar (atomik).
- İsteğe bağlı: crime_predict_data/metrics_all.csv (artifact) içinden
  en iyi (tercihen stacking) sonucu seçip JSON'a aktarır.
- UI/Streamlit kodu içermez.

Ortam değişkenleri:
- SUTAM_METRICS_FILE: latest_metrics.json için mutlak/bağıl yol override.
"""

import json
import os
import tempfile
import argparse
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TypedDict

# Pandas (CSV için gerekli); yüklü değilse CSV fonksiyonları kullanılamaz
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


class Metrics(TypedDict, total=False):
    auc: float
    hit_rate_topk: float
    brier: float
    timestamp: str


# ── Yol çözümleyici: proje kökü ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # services/.. = proje kökü
DEFAULT_METRICS_PATH = os.path.join(BASE_DIR, "data", "latest_metrics.json")

# Ortam değişkeni ile override (örn. CI/CD, Docker)
METRICS_FILE = os.environ.get("SUTAM_METRICS_FILE", DEFAULT_METRICS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# İç doğrulamalar
# ─────────────────────────────────────────────────────────────────────────────
def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _validate_metrics(d: Dict[str, Any]) -> Optional[Metrics]:
    """JSON içeriğini doğrula ve tip/aralık kontrollerini yap."""
    try:
        auc = _to_float(d.get("auc"))
        hit = _to_float(d.get("hit_rate_topk"))
        brier = _to_float(d.get("brier"))
        ts = d.get("timestamp")

        # Zorunlu alan: zaman damgası
        if ts is None:
            return None
        # Aralık kontrolleri (varsa)
        if auc is not None and not (0.0 <= auc <= 1.0):
            return None
        if hit is not None and not (0.0 <= hit <= 1.0):
            return None
        if brier is not None and not (0.0 <= brier <= 1.0):
            return None

        out: Metrics = Metrics(timestamp=str(ts))
        if auc is not None:
            out["auc"] = auc
        if hit is not None:
            out["hit_rate_topk"] = hit
        if brier is not None:
            out["brier"] = brier
        return out
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def get_latest_metrics() -> Optional[Metrics]:
    """
    En güncel metrikleri döndürür.
    Dosya yoksa veya doğrulama başarısızsa None döner (dummy yok).
    """
    try:
        if not os.path.exists(METRICS_FILE):
            print(f"[metrics] ⚠️ No metrics file found at: {METRICS_FILE}")
            return None
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        validated = _validate_metrics(data or {})
        if validated is None:
            print(f"[metrics] ❌ Invalid metrics content in: {METRICS_FILE}")
        return validated
    except Exception as e:
        print(f"[metrics] ❌ Error loading metrics from {METRICS_FILE}: {e}")
        return None


def save_latest_metrics(
    auc: Optional[float] = None,
    hit_rate_topk: Optional[float] = None,
    brier: Optional[float] = None,
) -> None:
    """
    Yeni metrikleri JSON dosyasına ATOMİK olarak kaydeder.
    (Önce temp dosyaya yazar, sonra os.replace ile hedefe taşır.)
    Tüm alanlar opsiyoneldir; yalnızca verilenler yazılır. Dummy üretmez.
    """
    try:
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

        payload: Metrics = Metrics(timestamp=datetime.now(timezone.utc).isoformat())
        if auc is not None:
            payload["auc"] = float(auc)
        if hit_rate_topk is not None:
            payload["hit_rate_topk"] = float(hit_rate_topk)
        if brier is not None:
            payload["brier"] = float(brier)

        if _validate_metrics(payload) is None:
            raise ValueError("Provided metrics failed validation.")

        dir_name = os.path.dirname(METRICS_FILE)
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=4)
            tmp_path = tmp.name

        os.replace(tmp_path, METRICS_FILE)  # aynı FS içinde atomic move
        print(f"[metrics] ✅ Metrics saved at {payload['timestamp']} → {METRICS_FILE}")

    except Exception as e:
        print(f"[metrics] ❌ Error saving metrics to {METRICS_FILE}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Artifact → JSON yardımcıları (opsiyonel)
# ─────────────────────────────────────────────────────────────────────────────
def find_latest_artifact_csv(base_dir: Optional[str] = None) -> Optional[str]:
    """
    crime_predict_data klasöründe metrics_all.csv dosyasını otomatik bulur.
    Bulamazsa None döner.
    """
    base_dir = base_dir or BASE_DIR
    candidate_dirs = [
        os.path.join(base_dir, "crime_predict_data"),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "artifacts"),
    ]
    for d in candidate_dirs:
        path = os.path.join(d, "metrics_all.csv")
        if os.path.exists(path):
            print(f"[metrics] 📊 Found artifact metrics file at: {path}")
            return path
    print("[metrics] ⚠️ No metrics_all.csv found under crime_predict_data / data / artifacts.")
    return None


def _pick_best_row(df: "pd.DataFrame", prefer_group: str = "stacking") -> "pd.Series":
    """
    Önce group==prefer_group filtrelenir; sonra en yüksek roc_auc'lu satır seçilir.
    roc_auc yoksa f1'e düşer. Hiçbiri yoksa ValueError.
    """
    cand = df
    if "group" in df.columns:
        mask = df["group"].astype(str).str.lower() == prefer_group.lower()
        if mask.any():
            cand = df[mask]

    metric = "roc_auc" if "roc_auc" in cand.columns else ("f1" if "f1" in cand.columns else None)
    if metric is None or cand.empty:
        raise ValueError("Uygun metrik sütunu (roc_auc / f1) yok ya da tablo boş.")

    return cand.sort_values(metric, ascending=False).iloc[0]


def update_from_csv(
    csv_path: Optional[str] = None,
    *,
    hit_col: Optional[str] = None,
    prefer_group: str = "stacking",
) -> Metrics:
    """
    metrics_all.csv içinden en iyi (tercihen stacking) satırı alır ve JSON'a yazar.
    - AUC ← roc_auc (yoksa f1)
    - Brier ← brier (varsa)
    - HitRate@TopK ← hit_col belirtilirse (varsa)
    Geriye yazılan payload'ı (timestamp hariç) döndürür.
    """
    if pd is None:
        raise RuntimeError("pandas yüklü değil; CSV'den metrik almak için pandas gerekli.")

    # Otomatik bul
    if csv_path is None:
        csv_path = find_latest_artifact_csv()
        if csv_path is None:
            raise FileNotFoundError("metrics_all.csv bulunamadı (crime_predict_data içinde).")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV boş.")

    row = _pick_best_row(df, prefer_group=prefer_group)

    # AUC
    auc: Optional[float] = None
    if "roc_auc" in row and pd.notna(row["roc_auc"]):
        auc = float(row["roc_auc"])
    elif "f1" in row and pd.notna(row["f1"]):
        # AUC yoksa f1’i AUC yerine göstermek ideal değil; isterseniz etiketi UI tarafında değiştirin.
        auc = float(row["f1"])

    # Brier
    brier: Optional[float] = None
    if "brier" in row and pd.notna(row["brier"]):
        brier = float(row["brier"])

    # HitRate@TopK (opsiyonel kolon adı)
    hit_rate_topk: Optional[float] = None
    if hit_col and (hit_col in row) and pd.notna(row[hit_col]):
        hit_rate_topk = float(row[hit_col])

    # Atomik yaz
    save_latest_metrics(auc=auc, hit_rate_topk=hit_rate_topk, brier=brier)

    # Geriye payload (timestamp hariç)
    out: Metrics = Metrics()
    if auc is not None:
        out["auc"] = auc
    if hit_rate_topk is not None:
        out["hit_rate_topk"] = hit_rate_topk
    if brier is not None:
        out["brier"] = brier
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI (opsiyonel)
#   python -m services.metrics                 → mevcut JSON'u yazdırır
#   python -m services.metrics --from-csv      → artifact'tan otomatik çeker ve JSON'a yazar
#   python -m services.metrics --from-csv PATH → belirtilen CSV'den JSON'a yazar
#   python -m services.metrics --from-csv --hit-col "hit_rate@100"
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(description="SUTAM metrics helper")
    ap.add_argument(
        "--from-csv",
        dest="from_csv",
        nargs="?",
        const="__AUTO__",   # argüman vermezsen otomatik ara
        default=None,
        help="metrics_all.csv yolu; verilmezse otomatik arar (crime_predict_data).",
    )
    ap.add_argument("--hit-col", dest="hit_col", default=None, help="HitRate@TopK kolonu adı (artifact'ta varsa)")
    ap.add_argument("--prefer-group", dest="prefer_group", default="stacking", help="Önce bu group'tan seç (varsayılan: stacking)")
    return ap.parse_args()


def _main() -> int:
    args = _parse_args()
    if args.from_csv is not None:
        path = None if args.from_csv == "__AUTO__" else args.from_csv
        m = update_from_csv(csv_path=path, hit_col=args.hit_col, prefer_group=args.prefer_group)
        print("[metrics] updated from CSV:", m)
        return 0
    else:
        m = get_latest_metrics()
        print("[metrics] current:", m)
        return 0


__all__ = [
    "Metrics",
    "METRICS_FILE",
    "get_latest_metrics",
    "save_latest_metrics",
    "find_latest_artifact_csv",
    "update_from_csv",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
