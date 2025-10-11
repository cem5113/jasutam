# utils/tz.py
from __future__ import annotations
from datetime import datetime, timedelta
from utils.constants import SF_TZ_OFFSET  # örn: -7

def now_sf() -> datetime:
    """San Francisco (PST/PDT) için basit offset tabanlı 'now' (naive)."""
    return datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)

def fmt_sf(dt: datetime, with_time: bool = True) -> str:
    if not isinstance(dt, datetime):
        return str(dt)
    return dt.strftime("%Y-%m-%d %H:%M") if with_time else dt.strftime("%Y-%m-%d")

def today_sf_str() -> str:
    return fmt_sf(now_sf(), with_time=False)
