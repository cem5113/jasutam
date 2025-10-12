# patrol/approvals.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import os, json, hashlib

from services.tz import now_sf  # SF zaman damgası

# JSONL log yolu: ENV ile ezilebilir
LOG_PATH = Path(os.environ.get("APPROVAL_LOG_PATH", "logs/approvals.jsonl"))

# Zorunlu alanlar (UI formuna uygun)
REQUIRED_FIELDS = ("alt_id", "assignment", "teams", "start", "end", "approver")


# ----------------- yardımcılar -----------------
def _content_hash(payload: dict) -> str:
    """İçerik karması (deterministik)."""
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _normalize_payload(payload: dict) -> dict:
    """
    Basit şema doğrulama/normalize:
    - Tüm zorunlu alanlar var mı?
    - teams -> liste (str)
    - start/end -> stringe çevrilir (ISO önerilir)
    """
    missing = [k for k in REQUIRED_FIELDS if k not in payload]
    if missing:
        raise ValueError(f"Eksik alan(lar): {', '.join(missing)}")

    p = dict(payload or {})  # kopya
    # teams: listeleştir
    t = p.get("teams", [])
    if isinstance(t, (str, bytes)):
        t = [str(t)]
    elif isinstance(t, (set, tuple)):
        t = list(t)
    p["teams"] = [str(x) for x in t]

    # metin alanlarını stringle
    for k in ("alt_id", "assignment", "start", "end", "approver"):
        if k in p and p[k] is not None:
            p[k] = str(p[k])

    return p


def _append_jsonl(path: Path, record: dict) -> None:
    """Tek satırlık JSONL yaz; dizini oluştur."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------- halka açık API -----------------
def save_approval(payload: dict) -> str:
    """
    Onayı kalıcı olarak kaydeder, benzersiz bir event_id döndürür.
    JSONL kaydı: {event_id, hash, ts_sf, ...payload}
    """
    p = _normalize_payload(payload)

    content_hash = _content_hash(p)
    ts_sf = now_sf().isoformat(timespec="seconds")
    # Aynı içerik tekrar onaylanırsa bile event_id farklı olsun (zaman damgası dahil)
    event_id = hashlib.sha256(f"{content_hash}|{ts_sf}".encode("utf-8")).hexdigest()[:12]

    rec = {
        "event_id": event_id,
        "hash": content_hash,
        "ts_sf": ts_sf,
        **p,
    }
    _append_jsonl(LOG_PATH, rec)
    return event_id


def list_approvals(limit: int = 500) -> List[dict]:
    """Son 'limit' kaydı döndürür (yeni → eski)."""
    if not LOG_PATH.exists():
        return []
    rows: List[Dict] = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows[-int(limit):]


def get_approval(event_id: str) -> Optional[dict]:
    """event_id ile tek bir kaydı bul (varsa)."""
    if not LOG_PATH.exists():
        return None
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("event_id") == event_id:
                    return rec
            except Exception:
                continue
    return None


def verify_integrity(record: dict) -> bool:
    """
    Bir JSONL kaydının bütünlüğünü doğrula:
    - İçerikten hash’i yeniden hesapla, 'hash' ile eşleşiyor mu?
    """
    if not isinstance(record, dict) or "hash" not in record:
        return False
    payload_keys = set(REQUIRED_FIELDS)
    payload = {k: record.get(k) for k in payload_keys}
    try:
        return _content_hash(_normalize_payload(payload)) == record["hash"]
    except Exception:
        return False
