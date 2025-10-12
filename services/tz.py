# services/tz.py
from __future__ import annotations
from datetime import datetime, date, timezone
from typing import Optional

# Ayarları settings'ten al
from config.settings import TZ_SF, TZ_TR, DATE_FMT, DATETIME_FMT

# zoneinfo yoksa pytz'e düş
try:
    from zoneinfo import ZoneInfo as _Zone
    _HAS_PYTZ = False

    def _get_tz(name: str):
        return _Zone(name)
except Exception:  # pragma: no cover
    from pytz import timezone as _pytz_tz  # type: ignore
    _HAS_PYTZ = True

    def _get_tz(name: str):
        return _pytz_tz(name)

# ── Temel "şimdi" yardımcıları ────────────────────────────────────────────────
def now_in(tz: str = TZ_SF) -> datetime:
    """Belirtilen saat diliminde 'aware' datetime döndürür."""
    return datetime.now(_get_tz(tz))

def now_sf() -> datetime:
    return now_in(TZ_SF)

def now_tr() -> datetime:
    return now_in(TZ_TR)

def today_sf() -> date:
    return now_sf().date()

def today_tr() -> date:
    return now_tr().date()

# ── Dönüşüm & biçimlendirme ───────────────────────────────────────────────────
def to_tz(dt: datetime, tz: str) -> datetime:
    """
    Naive ise verilen tz'yi **atanmış** kabul eder (localize/replace); 
    aware ise hedef saat dilimine dönüştürür.
    """
    tzobj = _get_tz(tz)
    if dt.tzinfo is None:
        # pytz için localize; zoneinfo için tzinfo atama
        if _HAS_PYTZ:  # pragma: no cover
            return tzobj.localize(dt)  # type: ignore[attr-defined]
        return dt.replace(tzinfo=tzobj)
    # zaten aware → hedefe çevir
    return dt.astimezone(tzobj)

def as_sf(dt: datetime) -> datetime:
    return to_tz(dt, TZ_SF)

def as_tr(dt: datetime) -> datetime:
    return to_tz(dt, TZ_TR)

def fmt(dt: datetime, fmt_str: str = DATETIME_FMT) -> str:
    """Datetime'i verilen biçime çevirir (tz fark etmeksizin)."""
    return dt.strftime(fmt_str)

def now_sf_str(fmt_str: str = DATETIME_FMT) -> str:
    return fmt(now_sf(), fmt_str)

def now_tr_str(fmt_str: str = DATETIME_FMT) -> str:
    return fmt(now_tr(), fmt_str)

# ── ISO yardımcıları ──────────────────────────────────────────────────────────
def parse_iso(s: str, assume_utc: bool = True) -> Optional[datetime]:
    """
    ISO-8601 string → aware datetime. 'Z' soneki desteklenir.
    Zaman dilimi yoksa (assume_utc=True) UTC varsayar.
    Hatalı ise None döndürür.
    """
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc) if assume_utc else dt
        return dt
    except Exception:
        return None

def iso_to_sf_str(s: str, fmt_str: str = DATETIME_FMT) -> str:
    """
    ISO string'i SF saatine çevirip formatlar; hata halinde '—' döndürür.
    """
    dt = parse_iso(s)
    if dt is None:
        return "—"
    return fmt(as_sf(dt), fmt_str)

# ── Tarih aralığı yardımcıları (gün başı/sonu) ────────────────────────────────
def day_bounds_sf(d: date) -> tuple[datetime, datetime]:
    """SF gününün başlangıç/bitiş (aware) değerleri."""
    start = to_tz(datetime(d.year, d.month, d.day, 0, 0, 0), TZ_SF)
    end   = to_tz(datetime(d.year, d.month, d.day, 23, 59, 59), TZ_SF)
    return start, end
