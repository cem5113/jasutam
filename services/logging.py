# services/logging.py
from __future__ import annotations
from pathlib import Path
import os, json, uuid, hashlib
from typing import Any, Dict, Optional

from services.tz import now_sf

# Ortam değişkenleriyle özelleştirilebilir
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
AUDIT_PATH = LOG_DIR / os.environ.get("AUDIT_FILE", "audit.jsonl")

def _session_id() -> str:
    """Her Streamlit oturumu için kısa bir UID üret/hatırla."""
    try:
        import streamlit as st
        return st.session_state.setdefault("_sid", uuid.uuid4().hex[:12])
    except Exception:
        # UI dışında çağrılırsa tek seferlik üret
        return uuid.uuid4().hex[:12]

def _event_id(event: str, payload: Dict[str, Any], ts: str) -> str:
    """Kayıt için kısa, stabil bir kimlik üret (12 hex)."""
    s = json.dumps({"e": event, "p": payload, "t": ts},
                   sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def audit(event: str,
          actor: Optional[str] = None,
          payload: Optional[Dict[str, Any]] = None) -> str:
    """
    Append-only JSONL denetim kaydı.
    Dönüş: event_id (hata durumunda boş string).
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = now_sf().isoformat(timespec="seconds")
        payload = payload or {}
        rec = {
            "event_id": _event_id(event, payload, ts),
            "ts_sf": ts,
            "session": _session_id(),
            "event": event,
            "actor": actor or "-",
            "payload": payload,
        }
        with AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec["event_id"]
    except Exception:
        # Log yazılamazsa uygulamayı durdurma
        return ""

def tail(limit: int = 100) -> list[dict]:
    """Son N denetim kaydını döndür."""
    if not AUDIT_PATH.exists():
        return []
    rows: list[dict] = []
    with AUDIT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows[-limit:]

# Geriye dönük uyumluluk
def tail_audit(limit: int = 100) -> list[dict]:
    return tail(limit)
