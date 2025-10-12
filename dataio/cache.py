# dataio/cache.py
from __future__ import annotations
from functools import lru_cache

# basit kullanım:
# @disk_or_mem_cache(...) -> burada sadece bellek içi örnek veriyoruz
def memcache(maxsize: int = 8):
    return lru_cache(maxsize=maxsize)
