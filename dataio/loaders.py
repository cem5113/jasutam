# dataio/loaders.py
from __future__ import annotations
import os, io, zipfile, json, requests
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd

# Ayarlar
from config.settings import DATA_DIR as _DATA_DIR, RESULTS_DIR as _RESULTS_DIR

DATA_DIR = Path(_DATA_DIR); DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(_RESULTS_DIR)

# Ortam değişkenleri
GITHUB_REPO          = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")   # owner/repo
GITHUB_WORKFLOW      = os.getenv("GITHUB_WORKFLOW", "full_pipeline.yml")
GITHUB_ARTIFACT_NAME = os.getenv("GITHUB_ARTIFACT_NAME", "sutam-results")          # workflow'daki artifact name
GH_TOKEN             = os.getenv("GH_TOKEN", "")

# Release fallback (opsiyonel)
CRIME_CSV_URL = os.getenv(
    "CRIME_CSV_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_crime.csv",
)

# GEOID uzunluğu (grid ID padleme)
GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# ----------------- yardımcılar -----------------
def _headers():
    if not GH_TOKEN:
        # GH_TOKEN yoksa artifact erişimini atla; caller try/except ile yakalıyor
        raise RuntimeError("GH_TOKEN yok (env). Artifact erişimi için gereklidir.")
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def _artifact_bytes(picks: List[str], artifact_name: Optional[str] = None) -> Optional[bytes]:
    """
    Son başarılı run’ın artifact’ından 'picks' içindeki ilk dosyayı döndürür (bytes).
    - 'artifact_name' verilirse önce onunla eşleşeni arar; yoksa herhangi NON-expired artifact'ı dener.
    - 'picks' hem tam ad hem de zip içindeki alt klasör (results/, out/, crime_prediction_data/) varyantlarını dener;
      bulunamazsa sonek (endswith) eşleşmesi yapar.
    """
    artifact_name = artifact_name or GITHUB_ARTIFACT_NAME
    runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"
    runs = requests.get(runs_url, headers=_headers(), timeout=30).json()
    run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]

    for rid in run_ids:
        arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
        arts = requests.get(arts_url, headers=_headers(), timeout=30).json().get("artifacts", [])
        # Önce isim eşleşen, yoksa herhangi NON-expired artifact
        ordered = ([a for a in arts if a.get("name") == artifact_name and not a.get("expired", False)] or
                   [a for a in arts if not a.get("expired", False)])

        for a in ordered:
            z = requests.get(a["archive_download_url"], headers=_headers(), timeout=60).content
            zf = zipfile.ZipFile(io.BytesIO(z))
            names = zf.namelist()

            # 1) Tam ad denemesi (alt klasör varyantlarıyla)
            for p in picks:
                for cand in (p, f"results/{p}", f"out/{p}", f"crime_prediction_data/{p}"):
                    if cand in names:
                        return zf.read(cand)

            # 2) Sonek eşleşmesi (en yaygın)
            for n in names:
                if any(n.endswith(p) for p in picks):
                    return zf.read(n)
    return None

def _normalize_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).str[:L].str.zfill(L)

def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    date → pandas datetime (saat bilgisi varsa kaybetme), event_hour türet.
    """
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
    else:
        out["date"] = pd.NaT
    if "event_hour" not in out.columns:
        out["event_hour"] = pd.to_datetime(out["date"], errors="coerce").dt.hour.fillna(0).astype(int)
    return out

def _ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    # Şimdilik dokunma; UI tarafı lat/lon yoksa uyarı gösteriyor
    return df.copy()

def _parse_and_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_latlon(_ensure_time_cols(df))
    if "GEOID" in df.columns:
        try:
            df["GEOID"] = _normalize_geoid(df["GEOID"])
        except Exception:
            df["GEOID"] = df["GEOID"].astype(str)
    return df

def _cache_latest(df: pd.DataFrame) -> None:
    """Her başarılı yüklemede yerel 'latest' alias'ını yaz."""
    try:
        (DATA_DIR / "sf_crime_latest.csv").write_text(
            df.to_csv(index=False), encoding="utf-8"
        )
    except Exception:
        pass

# ----------------- public API -----------------
def load_sf_crime_latest() -> Tuple[pd.DataFrame, str]:
    """
    Kaynak sırası:
      1) GitHub Actions artifact (ENV: GITHUB_ARTIFACT_NAME)
            aranan dosyalar: results/sf_crime_latest.parquet|csv, sf_crime_latest.parquet|csv, sf_crime.csv, sf_crime_09.csv, metrics_all.csv
      2) Release (latest): sf_crime.csv
      3) RESULTS_DIR: sf_crime_latest.parquet|csv
      4) Yerel cache (data/)
    Dönüş: (df, src_tag) — src_tag ∈ {"artifact","release","results","local:<ad>","empty"}
    """
    # 1) Artifact
    try:
        picks = [
            "sf_crime_latest.parquet", "sf_crime_latest.csv",
            "sf_crime.csv", "sf_crime_09.csv", "metrics_all.csv",
        ]
        blob = _artifact_bytes(picks=picks, artifact_name=GITHUB_ARTIFACT_NAME)
        if blob:
            # Önce CSV dene, olmazsa Parquet dene; her ikisi de başarısızsa tmp dosyadan dene
            try:
                df = pd.read_csv(io.BytesIO(blob), low_memory=False)
            except Exception:
                try:
                    df = pd.read_parquet(io.BytesIO(blob))  # pyarrow gerektirir
                except Exception:
                    tmp = DATA_DIR / "_artifact_tmp"
                    tmp.write_bytes(blob)
                    try:
                        # İçeriğe göre her iki okuma da denenir
                        try:
                            df = pd.read_parquet(tmp)
                        except Exception:
                            df = pd.read_csv(tmp, low_memory=False)
                    finally:
                        try: tmp.unlink()
                        except Exception: pass
            df = _parse_and_cleanup(df)
            _cache_latest(df)
            return df, "artifact"
    except Exception as e:
        print("artifact erişimi başarısız:", e)

    # 2) Release latest
    try:
        r = requests.get(CRIME_CSV_URL, timeout=60); r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
        df = _parse_and_cleanup(df)
        (DATA_DIR / "sf_crime_release.csv").write_bytes(r.content)
        _cache_latest(df)
        return df, "release"
    except Exception as e:
        print("release fallback başarısız:", e)

    # 3) RESULTS_DIR (repo içinde üretilmiş alias dosyaları)
    for cand, tag in [
        (RESULTS_DIR / "sf_crime_latest.parquet", "results"),
        (RESULTS_DIR / "sf_crime_latest.csv",     "results"),
    ]:
        if cand.exists():
            try:
                if str(cand.suffix).lower().endswith("parquet"):
                    df = pd.read_parquet(cand)
                else:
                    df = pd.read_csv(cand, low_memory=False)
                df = _parse_and_cleanup(df)
                _cache_latest(df)
                return df, tag
            except Exception as e:
                print("RESULTS okumada hata:", e)
                continue

    # 4) Yerel DATA_DIR (cache ve bilinen adlar)
    for name in ["sf_crime_latest.csv", "sf_crime_artifact_cache.csv", "sf_crime_09.csv", "metrics_all.csv", "sf_crime.csv"]:
        p = DATA_DIR / name
        if p.exists():
            try:
                df = pd.read_csv(p, low_memory=False)
                df = _parse_and_cleanup(df)
                _cache_latest(df)
                return df, f"local:{name}"
            except Exception:
                continue

    # 5) Hiçbiri yoksa — UI düşmesin
    df = pd.DataFrame({
        "GEOID": [], "date": [], "event_hour": [], "crime_count": [],
        "lat": [], "lon": []
    })
    return df, "empty"

def load_metadata() -> dict:
    """
    results/metadata.json → yoksa artifact’tan dene → yoksa {}
    """
    p = RESULTS_DIR / "metadata.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        blob = _artifact_bytes(
            picks=["metadata.json", "results/metadata.json", "out/metadata.json"],
            artifact_name=GITHUB_ARTIFACT_NAME,
        )
        if blob:
            return json.loads(blob.decode("utf-8"))
    except Exception:
        pass
    return {}
