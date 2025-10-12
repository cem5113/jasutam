# sutam/dataio/bootstrap.py
from __future__ import annotations
import os, io, json, zipfile, gzip
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import pandas as pd

# Basit HTTP; token'lı istekler için kullanılabilir
try:
    import requests  # Streamlit Cloud’da mevcut olur
except Exception:
    requests = None  # offline senaryoda local fallback zaten önde

# Proje kökü ve data yolu
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]          # .../ (sutam/ dataio/ ../..)
_DATA_DIR = _PROJECT_ROOT / "data"

# Okuma adayları (sizin repo düzeninize göre geniş tuttum)
CANDIDATES = [
    "sf_crime_52.parquet", "sf_crime_52.csv", "sf_crime_52.csv.gz", "sf_crime_52.zip",
    "sf_crime_50.parquet", "sf_crime_50.csv", "sf_crime_50.csv.gz", "sf_crime_50.zip",
    "sf_crime.parquet",    "sf_crime.csv",    "sf_crime.csv.gz",    "sf_crime.zip",
    "sf_crime_49.csv",     "sf_crime_49.csv.gz", "sf_crime_49.zip",
    "sf_crime_grid_full_labeled.csv", "sf_crime_grid_full_labeled.parquet",
    "sf_crime_01.csv", "sf_crime_01.parquet",
]

# RAW’dan okunacak varsayılan branch ve kök
DEFAULT_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
RAW_ROOT = f"https://raw.githubusercontent.com/{{repo}}/{DEFAULT_BRANCH}/data/{{filename}}"

def _read_any_bytes_to_df(name: str, content: bytes) -> pd.DataFrame:
    name_l = name.lower()
    bio = io.BytesIO(content)

    if name_l.endswith(".parquet"):
        return pd.read_parquet(bio)

    if name_l.endswith(".csv"):
        return pd.read_csv(bio)

    if name_l.endswith(".csv.gz") or name_l.endswith(".gz"):
        with gzip.GzipFile(fileobj=bio, mode="rb") as f:
            return pd.read_csv(io.BytesIO(f.read()))

    if name_l.endswith(".zip"):
        with zipfile.ZipFile(bio) as zf:
            # içindeki ilk csv/parquet’i aç
            names = zf.namelist()
            prefer = [n for n in names if n.lower().endswith((".parquet", ".csv"))]
            target = prefer[0] if prefer else names[0]
            with zf.open(target) as f:
                data = f.read()
                return _read_any_bytes_to_df(target, data)

    # son çare: düz metin csv gibi dene
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        return pd.DataFrame()

def _read_local(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as f:
                return pd.read_csv(io.BytesIO(f.read()))
        if path.suffix == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                prefer = [n for n in names if n.lower().endswith((".parquet", ".csv"))]
                target = prefer[0] if prefer else names[0]
                with zf.open(target) as f:
                    data = f.read()
                    return _read_any_bytes_to_df(target, data)
        # bilinmiyorsa csv dene
        return pd.read_csv(path)
    except Exception:
        return None

def _try_local_candidates() -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    for fname in CANDIDATES:
        df = _read_local(_DATA_DIR / fname)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return (f"local:data/{fname}", df)
    return (None, None)

def _http_get(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[bytes]:
    if requests is None:
        return None
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.content
        return None
    except Exception:
        return None

def _try_github_raw(repo: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    for fname in CANDIDATES:
        url = RAW_ROOT.format(repo=repo, filename=fname)
        content = _http_get(url)
        if content:
            df = _read_any_bytes_to_df(fname, content)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return (f"github:raw:{repo}/{DEFAULT_BRANCH}/data/{fname}", df)
    return (None, None)

# (İsteğe bağlı) Release ve Artifact denemeleri—kısa tutuyorum; yoksa local/raw zaten çoğu zaman yeter
def _try_github_release_asset(repo: str, token: Optional[str]) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    if requests is None or not token:
        return (None, None)
    try:
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        # Son release’i getir
        url = f"https://api.github.com/repos/{repo}/releases/latest"
        meta = _http_get(url, headers=headers)
        if not meta:
            return (None, None)
        j = json.loads(meta.decode("utf-8"))
        assets = j.get("assets") or []
        # adı adaylarla eşleşen ilk asset
        for a in assets:
            name = a.get("name", "")
            if not any(name.endswith(x) for x in [".csv", ".parquet", ".csv.gz", ".zip"]):
                continue
            dl_url = a.get("browser_download_url")
            if not dl_url:
                continue
            content = _http_get(dl_url, headers=headers) or _http_get(dl_url)  # bazıları public’tir
            if content:
                df = _read_any_bytes_to_df(name, content)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return (f"github:release:{repo}:{name}", df)
        return (None, None)
    except Exception:
        return (None, None)

def get_bootstrap() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    DÖNÜŞ:
      meta: {"source": "...", "app_name": "...", "error": "...?"}
      df  : pd.DataFrame (boş değilse başarı)
    Sıra:
      1) Yerel data/ adayları
      2) GitHub RAW (public)
      3) GitHub Releases (token gereksiz olabilir)
      4) (opsiyonel) Artifacts — burada atlıyoruz; gerekirse eklenir
    """
    app_name = os.environ.get("APP_NAME", "SUTAM")
    repo = os.environ.get("GITHUB_REPO", "cem5113/crime_prediction_data")
    token = os.environ.get("GH_TOKEN")  # opsiyonel

    # 1) Local
    src, df = _try_local_candidates()
    if df is not None and not df.empty:
        return ({"source": src, "app_name": app_name}, df)

    # 2) RAW
    src, df = _try_github_raw(repo)
    if df is not None and not df.empty:
        return ({"source": src, "app_name": app_name}, df)

    # 3) Release asset
    src, df = _try_github_release_asset(repo, token)
    if df is not None and not df.empty:
        return ({"source": src, "app_name": app_name}, df)

    # Hiçbiri yoksa:
    return (
        {
            "source": None,
            "app_name": app_name,
            "error": "Hiçbir kaynaktan veri bulunamadı. 'data/' içine sf_crime_52.csv (veya adaylardan biri) koyun "
                     "ya da GITHUB_REPO/RAW üzerinde data/<dosya> mevcut olsun.",
        },
        pd.DataFrame(),
    )
