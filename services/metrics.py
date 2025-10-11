# services/metrics.py
from __future__ import annotations
"""
SUTAM - Dinamik Model Performans / KPI Servisi
- latest_metrics.json dosyasını okur/yazar (atomik).
- Artifact olarak metrics_all.csv'yi DİZİN İÇLERİNDE veya ZIP içinde otomatik bulur.
- En iyi satırı (pr_auc > roc_auc > f1) seçerek JSON'a yazar.
"""

import argparse
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TypedDict, List, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


class Metrics(TypedDict, total=False):
    auc: float                 # ROC AUC (yoksa f1 ile doldurulabilir)
    pr_auc: float              # varsa Precision-Recall AUC
    hit_rate_topk: float       # opsiyonel
    brier: float               # opsiyonel
    model_name: str            # seçilen modelin etiketi/ismi
    selection_metric: str      # pr_auc / roc_auc / f1
    selection_value: float     # seçime esas değer
    source_path: str           # bulunduğu csv/zip yolu
    timestamp: str


# ── Yol çözümleyici ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # services/.. = proje kökü
DEFAULT_METRICS_PATH = os.path.join(BASE_DIR, "data", "latest_metrics.json")
METRICS_FILE = os.environ.get("SUTAM_METRICS_FILE", DEFAULT_METRICS_PATH)
ARTIFACT_ROOT_ENV = os.environ.get("SUTAM_ARTIFACT_DIR", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# İç doğrulamalar
# ─────────────────────────────────────────────────────────────────────────────
def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _validate_metrics(d: Dict[str, Any]) -> Optional[Metrics]:
    try:
        out: Metrics = Metrics()
        out["timestamp"] = str(d.get("timestamp") or datetime.now(timezone.utc).isoformat())

        # Aralık kontrollü metrikler
        for key in ("auc", "pr_auc", "hit_rate_topk", "brier"):
            if key in d and d[key] is not None:
                val = _to_float(d[key])
                if val is None:
                    continue
                if key in ("auc", "pr_auc", "hit_rate_topk", "brier") and not (0.0 <= val <= 1.0):
                    return None
                out[key] = val

        # Serbest alanlar
        for k in ("model_name", "selection_metric", "selection_value", "source_path"):
            if k in d and d[k] is not None:
                out[k] = d[k] if k != "selection_value" else _to_float(d[k])

        return out
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# JSON API
# ─────────────────────────────────────────────────────────────────────────────
def get_latest_metrics() -> Optional[Metrics]:
    try:
        if not os.path.exists(METRICS_FILE):
            print(f"[metrics] ⚠️ No metrics file found at: {METRICS_FILE}")
            return None
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        val = _validate_metrics(data or {})
        if val is None:
            print(f"[metrics] ❌ Invalid metrics content in: {METRICS_FILE}")
        return val
    except Exception as e:
        print(f"[metrics] ❌ Error loading metrics from {METRICS_FILE}: {e}")
        return None


def save_latest_metrics(**kwargs: Any) -> None:
    try:
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        payload: Dict[str, Any] = dict(kwargs)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        if _validate_metrics(payload) is None:
            raise ValueError("Provided metrics failed validation.")

        dir_name = os.path.dirname(METRICS_FILE)
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=4)
            tmp_path = tmp.name
        os.replace(tmp_path, METRICS_FILE)
        print(f"[metrics] ✅ Metrics saved → {METRICS_FILE}")
    except Exception as e:
        print(f"[metrics] ❌ Error saving metrics to {METRICS_FILE}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Artifact tarama ve seçim
# ─────────────────────────────────────────────────────────────────────────────
def _candidate_roots(base_dir: str) -> List[str]:
    roots = []
    if ARTIFACT_ROOT_ENV:
        roots.append(os.path.abspath(ARTIFACT_ROOT_ENV))
    roots.extend([
        os.path.join(base_dir, "crime_predict_data"),
        os.path.join(base_dir, "artifacts"),
        os.path.join(base_dir, "data"),
    ])
    # tekilleştir
    uniq: List[str] = []
    for r in roots:
        r = os.path.abspath(r)
        if r not in uniq and os.path.isdir(r):
            uniq.append(r)
    return uniq


def _find_csv_recursively(roots: List[str]) -> Optional[str]:
    targets = {"metrics_all.csv", "METRICS_ALL.CSV"}
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower() == "metrics_all.csv":
                    path = os.path.join(dirpath, fn)
                    print(f"[metrics] 📊 Found metrics_all.csv → {path}")
                    return path
    return None


def _find_zip_recursively(roots: List[str]) -> Optional[str]:
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".zip"):
                    path = os.path.join(dirpath, fn)
                    # zip içinde metrics_all.csv var mı?
                    try:
                        with zipfile.ZipFile(path) as zf:
                            hits = [n for n in zf.namelist() if n.lower().endswith("metrics_all.csv")]
                            if hits:
                                print(f"[metrics] 📦 Found metrics_all.csv in zip → {path} :: {hits[0]}")
                                return path
                    except zipfile.BadZipFile:
                        continue
    return None


def _read_metrics_df(csv_path: Optional[str], roots: List[str]) -> Tuple[pd.DataFrame, str]:
    if pd is None:
        raise RuntimeError("pandas gerekli (metrics_all.csv okumak için).")

    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path), csv_path

    csv_on_disk = _find_csv_recursively(roots)
    if csv_on_disk:
        return pd.read_csv(csv_on_disk), csv_on_disk

    zip_path = _find_zip_recursively(roots)
    if zip_path:
        with zipfile.ZipFile(zip_path) as zf:
            inner = [n for n in zf.namelist() if n.lower().endswith("metrics_all.csv")][0]
            with zf.open(inner) as fh:
                data = fh.read()
            df = pd.read_csv(io.BytesIO(data))
            return df, f"{zip_path}!{inner}"

    raise FileNotFoundError("metrics_all.csv düz dosya veya zip içinde bulunamadı.")


def _pick_best_row(df: "pd.DataFrame") -> Tuple[pd.Series, str, float]:
    # Seçim sırası: pr_auc > roc_auc > f1
    for metric in ("pr_auc", "roc_auc", "f1"):
        if metric in df.columns and df[metric].notna().any():
            row = df.sort_values(metric, ascending=False).iloc[0]
            return row, metric, float(row[metric])
    raise ValueError("Seçim için pr_auc / roc_auc / f1 sütunlarından hiçbiri bulunamadı.")


def _guess_model_name(row: "pd.Series") -> str:
    for key in ("model", "model_name", "estimator", "algo", "algorithm"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    # group + model birleştir
    grp = str(row.get("group", "") or "").strip()
    mdl = str(row.get("model", "") or "").strip()
    if grp and mdl:
        return f"{grp}/{mdl}"
    return grp or mdl or "unknown"


def _guess_hit_col(df: "pd.DataFrame") -> Optional[str]:
    # en uygun görünen ilk kolon: hit_rate@K, hit@K, hit_topk, etc.
    candidates = [c for c in df.columns if str(c).lower().startswith(("hit_rate@", "hit@", "hit_top", "hitrate@"))]
    return candidates[0] if candidates else None


def update_from_csv(
    csv_path: Optional[str] = None,
    *,
    prefer_group: Optional[str] = None,   # artık zorunlu değil; tüm tablo üzerinden seçiyoruz
    hit_col: Optional[str] = None,
) -> Metrics:
    roots = _candidate_roots(BASE_DIR)
    df, src = _read_metrics_df(csv_path, roots)

    if prefer_group and "group" in df.columns:
        mask = df["group"].astype(str).str.lower() == prefer_group.lower()
        if mask.any():
            df = df[mask]

    if df.empty:
        raise ValueError("metrics_all.csv boş (filtre sonrası).")

    best, sel_metric, sel_value = _pick_best_row(df)

    # Çıktı alanları
    pr_auc = float(best["pr_auc"]) if "pr_auc" in best and pd.notna(best["pr_auc"]) else None
    auc    = float(best["roc_auc"]) if "roc_auc" in best and pd.notna(best["roc_auc"]) else (
             float(best["f1"]) if "f1" in best and pd.notna(best["f1"]) else None)
    brier  = float(best["brier"])  if "brier"  in best and pd.notna(best["brier"])  else None

    # HitRate@TopK
    if hit_col is None:
        hit_col = _guess_hit_col(df)
    hit_rate_topk = float(best[hit_col]) if hit_col and (hit_col in best) and pd.notna(best[hit_col]) else None

    model_name = _guess_model_name(best)

    payload: Metrics = Metrics(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
        selection_metric=sel_metric,
        selection_value=sel_value,
        source_path=src,
    )
    if pr_auc is not None:
        payload["pr_auc"] = pr_auc
    if auc is not None:
        payload["auc"] = auc
    if brier is not None:
        payload["brier"] = brier
    if hit_rate_topk is not None:
        payload["hit_rate_topk"] = hit_rate_topk

    save_latest_metrics(**payload)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(description="SUTAM metrics helper")
    ap.add_argument("--from-csv", nargs="?", const="__AUTO__", default=None,
                    help="metrics_all.csv yolu; verilmezse dizinler ve zip içinde otomatik arar.")
    ap.add_argument("--prefer-group", default=None, help="Önce bu group'tan seç (opsiyonel)")
    ap.add_argument("--hit-col", default=None, help="HitRate@TopK kolonu adı (otomatik tespit varsayılan)")
    return ap.parse_args()


def _main() -> int:
    args = _parse_args()
    if args.from_csv is not None:
        path = None if args.from_csv == "__AUTO__" else args.from_csv
        m = update_from_csv(csv_path=path, prefer_group=args.prefer_group, hit_col=args.hit_col)
        print("[metrics] updated from artifact:", m)
    else:
        print("[metrics] current:", get_latest_metrics())
    return 0


__all__ = [
    "Metrics",
    "METRICS_FILE",
    "get_latest_metrics",
    "save_latest_metrics",
    "update_from_csv",
]

if __name__ == "__main__":
    raise SystemExit(_main())
