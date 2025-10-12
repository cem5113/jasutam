from __future__ import annotations
import os
from pathlib import Path

# ── Uygulama adı ───────────────────────────────────────────────────────────────
APP_NAME: str = os.getenv("APP_NAME", "SUTAM – Suç Tahmin Modeli")

# ── Zaman / biçimler ──────────────────────────────────────────────────────────
TZ_SF: str = os.getenv("TZ_SF", "America/Los_Angeles")
TZ_TR: str = os.getenv("TZ_TR", "Europe/Istanbul")
DATE_FMT: str = os.getenv("DATE_FMT", "%Y-%m-%d")
DATETIME_FMT: str = os.getenv("DATETIME_FMT", "%Y-%m-%d %H:%M")

# ── Dizinler ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(os.getenv("BASE_DIR", ".")).resolve()
DATA_DIR    = Path(os.getenv("DATA_DIR", BASE_DIR / "data")).resolve()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results")).resolve()
OUT_DIR     = Path(os.getenv("OUT_DIR", BASE_DIR / "out")).resolve()
LOG_DIR     = Path(os.getenv("LOG_DIR", BASE_DIR / "logs")).resolve()

# Uygulama çalışsın diye en azından out/logs var olsun
for p in (OUT_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)
    
# ── Yardımcılar ───────────────────────────────────────────────────────────────
def _git_rev_short(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return None

def _read_meta() -> Dict[str, Any]:
    p = RESULTS_DIR / "metadata.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _fmt_sf(iso_or_dt: Any) -> str:
    """ISO string / datetime → SF saatine formatlanmış string; yoksa '—'."""
    if not iso_or_dt:
        return "—"
    try:
        if isinstance(iso_or_dt, str):
            # 'Z' son ekini destekle
            iso_or_dt = iso_or_dt.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso_or_dt)
        elif isinstance(iso_or_dt, datetime):
            dt = iso_or_dt
        else:
            return "—"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ZoneInfo(TZ_SF)).strftime(DATETIME_FMT)
    except Exception:
        return "—"

def _json_env(name: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """ENV’den JSON oku; yoksa/default döndür. Bozuksa default’a düş."""
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return {**default, **val}
    except Exception:
        pass
    return default

def _bool_env(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")

# ── Meta / sürüm / zaman damgaları ────────────────────────────────────────────
META = _read_meta()
_git = _git_rev_short(BASE_DIR)

# Öncelik: ENV → META → git → 'unknown'
MODEL_VERSION: str = (
    os.getenv("MODEL_VERSION")
    or str(META.get("model_version") or "")
    or (f"git-{_git}" if _git else "unknown")
)

# ENV ile override edilebilir (ISO), aksi halde META’dan okunur
_last_trained_raw = os.getenv("LAST_TRAINED_AT") or META.get("last_trained_at")
_last_data_raw    = os.getenv("LAST_DATA_REFRESH_AT") or META.get("last_data_refresh_at")
LAST_TRAINED_AT: str       = _fmt_sf(_last_trained_raw)
LAST_DATA_REFRESH_AT: str  = _fmt_sf(_last_data_raw)

# ── Risk eşikleri / ağırlıklar ────────────────────────────────────────────────
# ENV örneği: RISK_THRESHOLDS='{"low":0.25,"mid":0.50,"high":0.75}'
_RISK_DEFAULT = {"low": 0.25, "mid": 0.50, "high": 0.75}
RISK_THRESHOLDS: Dict[str, float] = _json_env("RISK_THRESHOLDS", _RISK_DEFAULT)

# Devriye planlama için basit skor ağırlıkları (gerekirse ENV ile ayrı ayrı ezebilirsin)
RISK_WEIGHTS = {
    "very_high": int(os.getenv("WEIGHT_VERY_HIGH", "10")),
    "high":      int(os.getenv("WEIGHT_HIGH", "8")),
    "mid":       int(os.getenv("WEIGHT_MID", "6")),
    "low":       int(os.getenv("WEIGHT_LOW", "1")),
}

# ── Cache ve harita varsayılları ──────────────────────────────────────────────
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "600"))  # 10 dk
MAP_CENTER = {
    "lat": float(os.getenv("MAP_LAT", "37.7749")),       # SF
    "lon": float(os.getenv("MAP_LON", "-122.4194")),
    "zoom": float(os.getenv("MAP_ZOOM", "11")),
}

# ── Veri yayın/aksi değerlere dair ipuçları ───────────────────────────────────
DATA_RELEASE_TAG: str = os.getenv("DATA_RELEASE_TAG", "latest")
ARTIFACT_PRIORITY: bool = _bool_env("ARTIFACT_PRIORITY", True)  # loaders.py zaten artifact-öncelikli
