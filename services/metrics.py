# services/metrics.py
from __future__ import annotations
"""
SUTAM - Dinamik Model Performans / KPI Servisi
-----------------------------------------------
- latest_metrics.json dosyasını okur/yazar (atomik).
- Artifact (metrics_all.csv veya metrics_all*.zip) içinden en iyi modeli
  (PR-AUC > ROC-AUC > F1 önceliği, tercihen group='stacking') seçip JSON'a yazar.
- UI/Streamlit kodu içermez.

Ortam değişkenleri:
- SUTAM_METRICS_FILE   : latest_metrics.json için mutlak/bağıl yol override
- SUTAM_METRICS_CSV    : metrics_all.csv veya .zip için doğrudan yol
- SUTAM_METRICS_GROUP  : tercih edilen grup adı (varsayılan: 'stacking')
- SUTAM_HIT_COL        : HitRate@TopK için kolon adı (örn. 'hit_rate@100')
"""

import argparse
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TypedDict, List

# pandas CSV için gerekli
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


class Metrics(TypedDict, total=False):
    auc: float
    hit_rate_topk: float
    brier: float
    timestamp: str
    # Ek alanlar (UI'da göstermek için):
    model_name: str
    model_group: str
    source_artifact: str


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
    """
    JSON içeriğini doğrula ve tip/aralık kontrollerini yap.
    - timestamp zorunlu
    - (varsa) auc, hit, brier ∈ [0,1]
    - model_name, model_group, source_artifact serbest metin
    """
    try:
        ts = d.get("timestamp")
        if ts is None:
            return None

        auc = _to_float(d.get("auc"))
        hit = _to_float(d.get("hit_rate_topk"))
        brier = _to_float(d.get("brier"))

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

        # serbest metinler:
        for k in ("model_name", "model_group", "source_artifact"):
            v = d.get(k)
            if v is not None:
                out[k] = str(v)
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
    *,
    model_name: Optional[str] = None,
    model_group: Optional[str] = None,
    source_artifact: Optional[str] = None,
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
        if model_name:
            payload["model_name"] = str(model_name)
        if model_group:
            payload["model_group"] = str(model_group)
        if source_artifact:
            payload["source_artifact"] = str(source_artifact)

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
# Artifact → JSON yardımcıları
# ─────────────────────────────────────────────────────────────────────────────
def _candidate_artifact_paths(base_dir: str) -> List[str]:
    """Olası klasörlerde metrics_all.csv / metrics_all*.zip arar."""
    folders = [
        os.environ.get("SUTAM_METRICS_CSV", ""),  # doğrudan yol da olabilir
        os.path.join(base_dir, "crime_predict_data"),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "artifacts"),
    ]
    paths: List[str] = []
    for d in folders:
        if not d:
            continue
        if os.path.isfile(d):
            # doğrudan dosya verilmiş
            paths.append(d)
            continue
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.lower().startswith("metrics_all") and (name.lower().endswith(".csv") or name.lower().endswith(".zip")):
                paths.append(os.path.join(d, name))
    return paths


def find_latest_artifact(base_dir: Optional[str] = None) -> Optional[str]:
    """
    metrics_all.csv veya metrics_all*.zip içinden EN YENİ dosyanın yolunu döndürür.
    Bulamazsa None.
    """
    base_dir = base_dir or BASE_DIR
    cands = _candidate_artifact_paths(base_dir)
    if not cands:
        print("[metrics] ⚠️ No metrics_all.csv or metrics_all*.zip found under crime_predict_data / data / artifacts.")
        return None
    # En yeni (mtime) dosyayı seç
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    print(f"[metrics] 📊 Found artifact: {cands[0]}")
    return cands[0]


def _read_metrics_df(path: str) -> "pd.DataFrame":
    """CSV’yi doğrudan; ZIP ise içindeki metrics_all*.csv dosyasını okuyup DataFrame döndürür."""
    if pd is None:
        raise RuntimeError("pandas yüklü değil; artifact okumak için pandas gerekli.")

    low = path.lower()
    if low.endswith(".csv"):
        return pd.read_csv(path)

    if low.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            # ZIP içinde metrics_all*.csv ara, ilk eşleşeni oku
            names = [n for n in zf.namelist() if n.lower().startswith("metrics_all") and n.lower().endswith(".csv")]
            if not names:
                # CSV adı farklıysa: tüm .csv’ler arasından en “makul” olanı seç
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                names.sort()
            if not names:
                raise FileNotFoundError("ZIP içinde CSV bulunamadı.")
            with zf.open(names[0], "r") as f:
                content = f.read()
                return pd.read_csv(io.BytesIO(content))

    raise ValueError(f"Bilinmeyen artifact uzantısı: {path}")


def _best_metric_column(df: "pd.DataFrame") -> Optional[str]:
    """
    Öncelik sırası:
      1) PR-AUC benzerleri
      2) ROC-AUC
      3) F1
    """
    pr_candidates = ["pr_auc", "precision_recall_auc", "average_precision", "ap", "ap_auc", "pr_auc_val"]
    roc_candidates = ["roc_auc", "auc", "rocauc"]
    f1_candidates = ["f1", "f1_score"]

    for cols in (pr_candidates, roc_candidates, f1_candidates):
        for c in cols:
            if c in df.columns:
                return c
    return None


def _pick_best_row(df: "pd.DataFrame", prefer_group: str = "stacking") -> "pd.Series":
    """
    Önce group==prefer_group filtrelenir; sonra PR-AUC/ROC-AUC/F1 önceliğine göre en yüksek satır seçilir.
    """
    cand = df
    if "group" in df.columns:
        mask = df["group"].astype(str).str.lower() == str(prefer_group).lower()
        if mask.any():
            cand = df[mask]

    metric_col = _best_metric_column(cand)
    if metric_col is None or cand.empty:
        raise ValueError("Uygun metrik sütunu (PR/ROC/F1) yok ya da tablo boş.")

    return cand.sort_values(metric_col, ascending=False).iloc[0]


def update_from_csv(
    csv_path: Optional[str] = None,
    *,
    hit_col: Optional[str] = None,
    prefer_group: Optional[str] = None,
) -> Metrics:
    """
    Artifact içinden en iyi (tercihen stacking) satırı alır ve JSON'a yazar.
    - AUC  ← PR-AUC varsa onu, yoksa ROC-AUC, yoksa F1
    - Brier ← 'brier' sütunu varsa
    - HitRate@TopK ← hit_col belirtilirse (varsa)
    Ek olarak:
    - model_name  ← ['model','algo','algorithm','estimator','clf','name'] sütunlarından ilk bulunan
    - model_group ← 'group' sütunu (varsa)
    - source_artifact ← kullanılan artifact dosyasının yolu
    """
    if pd is None:
        raise RuntimeError("pandas yüklü değil; CSV/ZIP'ten metrik almak için pandas gerekli.")

    prefer_group = prefer_group or os.environ.get("SUTAM_METRICS_GROUP", "stacking")
    if csv_path is None:
        csv_path = os.environ.get("SUTAM_METRICS_CSV") or find_latest_artifact()
        if csv_path is None:
            raise FileNotFoundError("metrics_all.csv veya .zip bulunamadı (crime_predict_data / data / artifacts).")

    df = _read_metrics_df(csv_path)
    if df is None or df.empty:
        raise ValueError("Artifact metrik tablosu boş.")

    row = _pick_best_row(df, prefer_group=prefer_group)

    # AUC değeri (seçim)
    metric_col = _best_metric_column(df)
    auc: Optional[float] = None
    if metric_col and pd.notna(row.get(metric_col)):
        auc = float(row[metric_col])

    # Brier
    brier: Optional[float] = None
    if "brier" in row and pd.notna(row["brier"]):
        brier = float(row["brier"])

    # HitRate@TopK (kolon adı env/param ile gelebilir; yoksa makul adaylar)
    hit_col = hit_col or os.environ.get("SUTAM_HIT_COL")
    hit_candidates = [hit_col] if hit_col else ["hit_rate@100", "hit_rate_topk", "hit@100", "hitrate_topk"]
    hit_rate_topk: Optional[float] = None
    for c in hit_candidates:
        if c and (c in row) and pd.notna(row[c]):
            hit_rate_topk = float(row[c])
            break

    # Model adı (algoritma)
    name_candidates = ["model", "algo", "algorithm", "estimator", "clf", "name"]
    model_name = next((str(row[c]) for c in name_candidates if c in row and pd.notna(row[c])), None)
    model_group = str(row["group"]) if "group" in row and pd.notna(row["group"]) else None

    # Atomik yaz
    save_latest_metrics(
        auc=auc,
        hit_rate_topk=hit_rate_topk,
        brier=brier,
        model_name=model_name,
        model_group=model_group,
        source_artifact=os.path.relpath(csv_path, BASE_DIR) if os.path.isabs(csv_path) else csv_path,
    )

    # Geriye payload (timestamp hariç)
    out: Metrics = Metrics()
    if auc is not None:
        out["auc"] = auc
    if hit_rate_topk is not None:
        out["hit_rate_topk"] = hit_rate_topk
    if brier is not None:
        out["brier"] = brier
    if model_name:
        out["model_name"] = model_name
    if model_group:
        out["model_group"] = model_group
    out["source_artifact"] = os.path.relpath(csv_path, BASE_DIR) if os.path.isabs(csv_path) else csv_path
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI (opsiyonel)
#   python -m services.metrics                 → mevcut JSON'u yazdırır
#   python -m services.metrics --from-artifact → artifact'tan otomatik çeker ve JSON'a yazar
#   python -m services.metrics --from-artifact PATH[.csv|.zip]
#   python -m services.metrics --from-artifact --hit-col "hit_rate@100" --prefer-group "stacking"
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(description="SUTAM metrics helper")
    ap.add_argument(
        "--from-artifact",
        dest="from_artifact",
        nargs="?",
        const="__AUTO__",   # argüman verilmezse otomatik ara
        default=None,
        help="metrics_all.csv veya metrics_all*.zip yolu; verilmezse otomatik arar.",
    )
    ap.add_argument("--hit-col", dest="hit_col", default=None, help="HitRate@TopK kolonu adı (artifact'ta varsa)")
    ap.add_argument("--prefer-group", dest="prefer_group", default=None, help="Önce bu group'tan seç (örn. stacking)")
    return ap.parse_args()


def _main() -> int:
    args = _parse_args()
    if args.from_artifact is not None:
        path = None if args.from_artifact == "__AUTO__" else args.from_artifact
        m = update_from_csv(csv_path=path, hit_col=args.hit_col, prefer_group=args.prefer_group)
        print("[metrics] updated from artifact:", m)
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
    "find_latest_artifact",
    "update_from_csv",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
