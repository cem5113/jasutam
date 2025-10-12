# sutam/dataio/bootstrap.py
from __future__ import annotations
import os
import streamlit as st
from sutam.dataio.loaders import load_metadata, load_sf_crime_latest

# TTL: dakikada bir otomatik tazelensin istiyorsan ortamdan ayarla (dakika)
_TTL_MIN = int(os.getenv("CACHE_TTL_MIN", "0"))  # 0 = sınırsız
_ttl = _TTL_MIN * 60 if _TTL_MIN > 0 else None

@st.cache_data(show_spinner=False, ttl=_ttl)
def get_bootstrap():
    """
    Uygulama açılışında tek seferde veri + meta yükler (cache).
    """
    meta = load_metadata() or {}
    df = load_sf_crime_latest()
    return meta, df
