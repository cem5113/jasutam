# services/metrics.py
"""
SUTAM - Dinamik Model Performans / KPI Servisi
-----------------------------------------------
Bu modül, modelin en son performans metriklerini (AUC, HitRate@TopK, Brier)
dosyadan okur veya eğitim sonrası otomatik olarak kaydeder.
Hiçbir durumda sabit (dummy) değer dönmez.
"""

import json
import os
from datetime import datetime, timezone

# Metriklerin kaydedileceği JSON dosyası
METRICS_FILE = "data/latest_metrics.json"


def get_latest_metrics():
    """
    Modelin en güncel metriklerini döndürür.
    Eğer metrics.json dosyası yoksa None döner (dummy yok!).
    """
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r") as f:
                data = json.load(f)
                return {
                    "auc": data.get("auc"),
                    "hit_rate_topk": data.get("hit_rate_topk"),
                    "brier": data.get("brier"),
                    "timestamp": data.get("timestamp"),
                }
        else:
            # Dosya yoksa hiç değer dönmez (UI kutuları gizlenir)
            print("[metrics] ⚠️ No metrics file found.")
            return None

    except Exception as e:
        print(f"[metrics] ❌ Error loading metrics: {e}")
        return None


def save_latest_metrics(auc, hit_rate_topk, brier):
    """
    Yeni metrikleri JSON dosyasına kaydeder.
    Bu fonksiyon model eğitimi tamamlandığında çağrılır.
    """
    try:
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        data = {
            "auc": float(auc),
            "hit_rate_topk": float(hit_rate_topk),
            "brier": float(brier),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(METRICS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[metrics] ✅ Metrics saved at {data['timestamp']}")
    except Exception as e:
        print(f"[metrics] ❌ Error saving metrics: {e}")
