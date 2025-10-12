# services/ids.py
from __future__ import annotations
import hashlib, uuid
from typing import Any, Iterable
import hashlib, json
from typing import Any, Dict

def _canonical(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def content_hash(obj: Dict | Any, algo: str = "sha1") -> str:
    data = _canonical(obj).encode("utf-8")
    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()


def short_id(n: int = 8) -> str:
    """Kısa rasgele id (hex)."""
    return uuid.uuid4().hex[:max(4, n)]

def stable_hash(payload: Any, n: int = 12) -> str:
    """Deterministik içerik karması (json-sıralı önerilir)."""
    s = str(payload).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:max(6, n)]

def normalize_geoid(g: Any) -> str:
    """GEOID'i stringe çevirip kırpma/trim."""
    return str(g).strip()
