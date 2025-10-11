# services/metrics.py
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TypedDict

class Metrics(TypedDict, total=False):
    auc: float
    hit_rate_topk: float
    brier: float
    timestamp: str

# ── Yol çözümleyici: proje köküne göre mutlak yol ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # services/.. = proje kökü
DEFAULT_METRICS_PATH = os.path.join(BASE_DIR, "data", "latest_metrics.json")

# Ortam değişkeni ile override edilebilir (örn. CI/CD, Docker)
METRICS_FILE = os.environ.get("SUTAM_METRICS_FILE", DEFAULT_METRICS_PATH)

def _to_float(x):
    return float(x) if x is not None else None

def _validate_metrics(d: Dict[str, Any]) -> Optional[Metrics]:
    """JSON içeriğini doğrula ve tip/arama aralığı kontrolleri yap."""
    try:
        auc = _to_float(d.get("auc"))
        hit = _to_float(d.get("hit_rate_topk"))
        brier = _to_float(d.get("brier"))
        ts = d.get("timestamp")

        if ts is None:
            return None
        if auc is not None and not (0.0 <= auc <= 1.0):
            return None
        if hit is not None and not (0.0 <= hit <= 1.0):
            return None
        if brier is not None and not (0.0 <= brier <= 1.0):
            return None

        return Metrics(auc=auc, hit_rate_topk=hit, brier=brier, timestamp=str(ts))
    except Exception:
        return None

def get_latest_metrics() -> Optional[Metrics]:
    """
    Modelin en güncel metriklerini döndürür.
    Eğer dosya yoksa veya doğrulama başarısızsa None döner (dummy yok).
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

def save_latest_metrics(auc: float, hit_rate_topk: float, brier: float) -> None:
    """
    Yeni metrikleri JSON dosyasına ATOMİK olarak kaydeder.
    (Önce temp dosyaya yazar, sonra rename ile hedefe taşır.)
    Bu fonksiyon model eğitimi tamamlandığında çağrılmalıdır.
    """
    try:
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

        payload: Metrics = Metrics(
            auc=float(auc),
            hit_rate_topk=float(hit_rate_topk),
            brier=float(brier),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if _validate_metrics(payload) is None:
            raise ValueError("Provided metrics failed validation.")

        dir_name = os.path.dirname(METRICS_FILE)
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=4)
            tmp_path = tmp.name

        os.replace(tmp_path, METRICS_FILE)  # atomic move (aynı dosya sistemi içinde)
        print(f"[metrics] ✅ Metrics saved at {payload['timestamp']} → {METRICS_FILE}")

    except Exception as e:
        print(f"[metrics] ❌ Error saving metrics to {METRICS_FILE}: {e}")

__all__ = [
    "Metrics",
    "METRICS_FILE",
    "get_latest_metrics",
    "save_latest_metrics",
]
