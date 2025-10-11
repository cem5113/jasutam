# services/metrics.py
from __future__ import annotations
"""
SUTAM - Dinamik Model Performans / KPI Servisi (artifact-only)
- GitHub Actions artifact ZIP içindeki metrics_all.csv dosyasını bulup okur.
- En iyi satırı (pr_auc > roc_auc > f1) seçer ve data/latest_metrics.json'a yazar.
- Yerel klasör/ZIP fallback'leri KALDIRILDI: artifact bulunamazsa açık hata verir.
"""

import argparse
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TypedDict, List, Tuple

# ---- pandas gerekli ----
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

# ----------------- Ortam / Yol ayarları -----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # services/.. = proje kökü
DEFAULT_METRICS_PATH = os.path.join(BASE_DIR, "data", "latest_metrics.json")
METRICS_FILE = os.environ.get("SUTAM_METRICS_FILE", DEFAULT_METRICS_PATH)

# GitHub artifact erişimi
GITHUB_REPO          = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")   # owner/repo
GITHUB_ARTIFACT_NAME = os.getenv("GITHUB_ARTIFACT_NAME", "sutam-results")          # workflow'daki artifact adı
GH_TOKEN             = os.getenv("GH_TOKEN", "")

# --------------------------------------------------------
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


# ─────────────────────────────────────────────────────────────
# Yardımcılar (doğrulama / IO)
# ─────────────────────────────────────────────────────────────
def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _validate_metrics(d: Dict[str, Any]) -> Optional[Metrics]:
    try:
        out: Metrics = Metrics()
        out["timestamp"] = str(d.get("timestamp") or datetime.now(timezone.utc).isoformat())

        # [0,1] aralıklı metrikler
        for key in ("auc", "pr_auc", "hit_rate_topk", "brier"):
            if key in d and d[key] is not None:
                val = _to_float(d[key])
                if val is None:
                    continue
                if key in ("auc", "pr_auc", "hit_rate_topk", "brier") and not (0.0 <= val <= 1.0):
                    return None
                out[key] = val

        # serbest alanlar
        for k in ("model_name", "selection_metric", "selection_value", "source_path"):
            if k in d and d[k] is not None:
                out[k] = d[k] if k != "selection_value" else _to_float(d[k])

        return out
    except Exception:
        return None


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


# ─────────────────────────────────────────────────────────────
# GitHub Artifact erişimi (yalnızca artifact)
# ─────────────────────────────────────────────────────────────
def _gh_headers() -> Dict[str, str]:
    if not GH_TOKEN:
        raise RuntimeError("GH_TOKEN yok (env). Artifact erişimi için gereklidir.")
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _http_get_json(url: str, timeout: int = 30, retries: int = 3) -> Dict[str, Any]:
    import time as _t
    import requests
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=_gh_headers(), timeout=timeout)
            if r.status_code == 401:
                raise RuntimeError("GitHub API 401 Unauthorized: GH_TOKEN geçersiz veya yetkisiz.")
            if r.status_code == 403:
                raise RuntimeError(f"GitHub API 403 Forbidden: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            _t.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET {url} başarısız: {last}")


def _http_get_bytes(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    import time as _t
    import requests
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=_gh_headers(), timeout=timeout)
            if r.status_code == 401:
                raise RuntimeError("GitHub API 401 Unauthorized: GH_TOKEN geçersiz veya yetkisiz.")
            if r.status_code == 403:
                raise RuntimeError(f"GitHub API 403 Forbidden: {r.text[:200]}")
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            _t.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET(bytes) {url} başarısız: {last}")


def _artifact_bytes(picks: List[str], artifact_name: Optional[str] = None) -> bytes:
    """
    Son başarılı run’ların artifact’larından 'picks' içindeki ilk dosyayı döndürür (bytes).
    - 'artifact_name' verilirse önce onunla eşleşeni arar; yoksa herhangi NON-expired artifact'ı dener.
    - 'picks' hem tam ad hem de zip içindeki alt klasör varyantlarını dener; bulunamazsa sonek eşleşmesi yapar.
    """
    name = artifact_name or GITHUB_ARTIFACT_NAME
    runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=50"
    runs = _http_get_json(runs_url).get("workflow_runs", [])
    run_ids = [r["id"] for r in runs if r.get("conclusion") == "success"]

    for rid in run_ids:
        arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts?per_page=50"
        arts = _http_get_json(arts_url).get("artifacts", [])
        ordered = ([a for a in arts if a.get("name") == name and not a.get("expired", False)] or
                   [a for a in arts if not a.get("expired", False)])

        for a in ordered:
            zip_bytes = _http_get_bytes(a["archive_download_url"])
            try:
                zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
            except zipfile.BadZipFile:
                continue

            names = zf.namelist()
            # 1) Tam ad denemesi (alt klasör varyantlarıyla)
            for p in picks:
                for cand in (
                    p,
                    f"results/{p}",
                    f"out/{p}",
                    f"crime_prediction_data/{p}",
                    f"artifact/{p}",
                    f"metrics/{p}",
                ):
                    if cand in names:
                        return zf.read(cand)

            # 2) Sonek eşleşmesi (en yaygın)
            for n in names:
                if any(n.lower().endswith(p.lower()) for p in picks):
                    return zf.read(n)

    raise FileNotFoundError(
        f"Artifact içinde istenen dosya bulunamadı. repo={GITHUB_REPO}, artifact={name}, picks={picks}"
    )


def _read_metrics_df_from_artifact() -> Tuple["pd.DataFrame", str]:
    """Artifact içinden metrics_all.csv okumayı dener; bulunamazsa istisna fırlatır."""
    if pd is None:
        raise RuntimeError("pandas kurulu değil; metrics_all.csv okunamıyor. `pip install pandas`")

    blob = _artifact_bytes(picks=["metrics_all.csv", "METRICS_ALL.CSV"], artifact_name=GITHUB_ARTIFACT_NAME)

    try:
        df = pd.read_csv(io.BytesIO(blob))
    except Exception as e:
        raise RuntimeError(f"metrics_all.csv okuma hatası: {e}")

    cols = {c.lower() for c in df.columns}
    if not ({"pr_auc", "roc_auc", "f1"} & cols):
        raise ValueError(
            "metrics_all.csv içinde seçim için gerekli sütunlardan en az biri yok: pr_auc / roc_auc / f1"
        )

    return df, f"artifact:{GITHUB_ARTIFACT_NAME}"


# ─────────────────────────────────────────────────────────────
# Seçim ve isim tahmini
# ─────────────────────────────────────────────────────────────
def _pick_best_row(df: "pd.DataFrame") -> Tuple["pd.Series", str, float]:
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
    grp = str(row.get("group", "") or "").strip()
    mdl = str(row.get("model", "") or "").strip()
    if grp and mdl:
        return f"{grp}/{mdl}"
    return grp or mdl or "unknown"


def _guess_hit_col(df: "pd.DataFrame") -> Optional[str]:
    # En uygun görünen ilk kolon: hit_rate@K, hit@K, hit_topk, hitrate@K...
    candidates = [c for c in df.columns if str(c).lower().startswith(("hit_rate@", "hit@", "hit_top", "hitrate@"))]
    return candidates[0] if candidates else None


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def update_from_csv(
    csv_path: Optional[str] = None,  # kullanılmıyor (artifact-only)
    *,
    prefer_group: Optional[str] = None,
    hit_col: Optional[str] = None,
) -> Metrics:
    """Sadece artifact'tan okur; bulunamazsa istisna atar."""
    df, src = _read_metrics_df_from_artifact()

    # Opsiyonel grup filtresi
    if prefer_group and "group" in df.columns:
        mask = df["group"].astype(str).str.lower() == prefer_group.lower()
        if mask.any():
            df = df[mask]
        else:
            print(f"[metrics] ⚠️ prefer_group='{prefer_group}' için satır bulunamadı; tüm tablo üzerinden seçim yapılacak.")

    if df.empty:
        raise ValueError("metrics_all.csv boş (filtre sonrası).")

    best, sel_metric, sel_value = _pick_best_row(df)

    pr_auc = float(best["pr_auc"]) if "pr_auc" in best and pd.notna(best["pr_auc"]) else None
    auc    = float(best["roc_auc"]) if "roc_auc" in best and pd.notna(best["roc_auc"]) else (
             float(best["f1"]) if "f1" in best and pd.notna(best["f1"]) else None)
    brier  = float(best["brier"])  if "brier"  in best and pd.notna(best["brier"])  else None

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


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(description="SUTAM metrics helper (artifact-only)")
    ap.add_argument("--prefer-group", default=None, help="Önce bu group'tan seç (opsiyonel)")
    ap.add_argument("--hit-col", default=None, help="HitRate@TopK kolonu adı (otomatik tespit varsayılan)")
    return ap.parse_args()


def _main() -> int:
    args = _parse_args()
    m = update_from_csv(prefer_group=args.prefer_group, hit_col=args.hit_col)
    print("[metrics] updated:", m)
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
