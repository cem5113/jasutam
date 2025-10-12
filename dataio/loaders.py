# dataio/loaders.py
from __future__ import annotations

# --- imports ---
import os, io, zipfile, json, requests
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd

# --- paths ---
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]

# --- config ---
from config.settings import DATA_DIR as SETTINGS_DATA_DIR, RESULTS_DIR as SETTINGS_RESULTS_DIR
DATA_DIR = Path(SETTINGS_DATA_DIR); DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(SETTINGS_RESULTS_DIR); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- env ---
GITHUB_REPO          = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")
GITHUB_WORKFLOW      = os.getenv("GITHUB_WORKFLOW", "full_pipeline.yml")
GITHUB_ARTIFACT_NAME = os.getenv("GITHUB_ARTIFACT_NAME", "sutam-results")
GH_TOKEN             = os.getenv("GH_TOKEN", "")
CRIME_CSV_URL        = os.getenv("CRIME_CSV_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_crime.csv")
GEOID_LEN            = int(os.getenv("GEOID_LEN", "11"))

# --- schema ---
REQUIRED_COLS = ["GEOID", "date", "event_hour"]

# ===================== helpers =====================

def _headers(require_auth: bool = False) -> Optional[Dict[str, str]]:
    """
    GitHub API headerları. require_auth=True ise ve GH_TOKEN yoksa None döner.
    """
    base = {"Accept": "application/vnd.github+json"}
    if GH_TOKEN:
        base["Authorization"] = f"Bearer {GH_TOKEN}"
    if require_auth and not GH_TOKEN:
        return None
    return base

def _artifact_bytes(picks: List[str], artifact_name: Optional[str] = None) -> Optional[bytes]:
    """
    GitHub Actions artifactlarından ilk eşleşen dosyayı bytes olarak döndürür.
    GH_TOKEN yoksa None döner (sessiz atla).
    """
    headers = _headers(require_auth=True)
    if headers is None:
        return None

    artifact_name = artifact_name or GITHUB_ARTIFACT_NAME
    runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"
    try:
        runs = requests.get(runs_url, headers=headers, timeout=30).json()
    except Exception:
        return None

    run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]

    for rid in run_ids:
        try:
            arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
            arts = requests.get(arts_url, headers=headers, timeout=30).json().get("artifacts", [])
        except Exception:
            continue

        ordered = ([a for a in arts if a.get("name") == artifact_name and not a.get("expired", False)]
                   or [a for a in arts if not a.get("expired", False)])

        for a in ordered:
            try:
                z_content = requests.get(a["archive_download_url"], headers=headers, timeout=60).content
                zf = zipfile.ZipFile(io.BytesIO(z_content))
                names = zf.namelist()

                # doğrudan verilen yollar
                for p in picks:
                    for cand in (p, f"results/{p}", f"out/{p}", f"crime_prediction_data/{p}"):
                        if cand in names:
                            return zf.read(cand)

                # sonda eşleşme
                for n in names:
                    if any(n.endswith(p) for p in picks):
                        return zf.read(n)
            except Exception:
                continue
    return None

def _normalize_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).str[:L].str.zfill(L)

def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    date/datetime -> pandas datetime; UTC ise naive'a çevir (tz bilgisi kaldırılır).
    event_hour yoksa saat çıkarılır.
    """
    out = df.copy()
    # date/datetime normalize
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None)
    elif "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        out["date"] = pd.NaT

    # event_hour
    if "event_hour" not in out.columns:
        hours = pd.to_datetime(out["date"], errors="coerce").dt.hour
        out["event_hour"] = hours.where(hours.notna(), None).fillna(0).astype(int)
    return out

def _ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mevcut kolonlardan lat/lon türetir (lat/latitude/y/lat_dd, lon/longitude/x/lng/long/lon_dd).
    """
    out = df.copy()
    cand_lat = [c for c in out.columns if c.lower() in ["lat","latitude","y","lat_dd"]]
    cand_lon = [c for c in out.columns if c.lower() in ["lon","longitude","x","lng","long","lon_dd"]]
    if cand_lat and "lat" not in out.columns:
        out["lat"] = pd.to_numeric(out[cand_lat[0]], errors="coerce")
    if cand_lon and "lon" not in out.columns:
        out["lon"] = pd.to_numeric(out[cand_lon[0]], errors="coerce")
    return out

def _parse_and_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_latlon(_ensure_time_cols(df))
    if "GEOID" in df.columns:
        try:
            df["GEOID"] = _normalize_geoid(df["GEOID"])
        except Exception:
            df["GEOID"] = df["GEOID"].astype(str)
    return df

def _validate_schema(df: pd.DataFrame) -> tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return len(missing) == 0, missing

def _cache_latest(df: pd.DataFrame) -> None:
    try:
        (DATA_DIR / "sf_crime_latest.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    except Exception:
        pass

# ===================== unified metadata loader =====================

def load_metadata() -> Dict[str, Any]:
    """
    Öncelik: results/metadata.json → artifact → {}
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

def load_metadata_or_default() -> Dict[str, Any]:
    m = load_metadata()
    if m:
        return m
    return {
        "source": "unknown",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "rows": 0,
        "cols": 0,
        "columns": [],
        "date_min": None,
        "date_max": None,
        "has_latlon": False,
    }

# ===================== public API =====================

def load_sf_crime_latest() -> Tuple[pd.DataFrame, str]:
    """
    Kaynak sırası:
      1) GitHub Actions artifact (ENV: GITHUB_ARTIFACT_NAME)
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
            # Parquet/CSV olarak dene (sırayla)
            try:
                df = pd.read_csv(io.BytesIO(blob), low_memory=False)
            except Exception:
                try:
                    df = pd.read_parquet(io.BytesIO(blob))
                except Exception:
                    # İçerik tipi belirsizse diske yazıp tekrar dene
                    tmp = DATA_DIR / "_artifact_tmp"
                    tmp.write_bytes(blob)
                    try:
                        try:
                            df = pd.read_parquet(tmp)
                        except Exception:
                            df = pd.read_csv(tmp, low_memory=False)
                    finally:
                        try:
                            tmp.unlink()
                        except Exception:
                            pass
            df = _parse_and_cleanup(df)
            _cache_latest(df)
            return df, "artifact"
    except Exception as e:
        print("artifact erişimi başarısız:", e)

    # 2) Release (latest)
    try:
        r = requests.get(CRIME_CSV_URL, timeout=60); r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
        df = _parse_and_cleanup(df)
        (DATA_DIR / "sf_crime_release.csv").write_bytes(r.content)
        _cache_latest(df)
        return df, "release"
    except Exception as e:
        print("release fallback başarısız:", e)

    # 3) RESULTS_DIR
    for cand, tag in [
        (RESULTS_DIR / "sf_crime_latest.parquet", "results"),
        (RESULTS_DIR / "sf_crime_latest.csv",     "results"),
    ]:
        if cand.exists():
            try:
                if cand.suffix.lower() == ".parquet":
                    df = pd.read_parquet(cand)
                else:
                    df = pd.read_csv(cand, low_memory=False)
                df = _parse_and_cleanup(df)
                _cache_latest(df)
                return df, tag
            except Exception as e:
                print("RESULTS okumada hata:", e)
                continue

    # 4) Yerel DATA_DIR cache
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

    # 5) boş
    df = pd.DataFrame({"GEOID": [], "date": [], "event_hour": [], "crime_count": [], "lat": [], "lon": []})
    return df, "empty"
