# components/last_update.py
from __future__ import annotations
from datetime import datetime, date
from typing import Optional, Union
import html

import streamlit as st

Dateish = Union[str, datetime, date, None]

def _fmt_date(
    d: Dateish,
    show_time: bool = False,
    date_fmt: str = "%Y-%m-%d",
    time_fmt: str = "%H:%M"
) -> str:
    """Dateish'i tek biçimde yaz. str gelirse aynen bırak (kullanıcı özel biçim istemiş olabilir)."""
    if d is None:
        return ""
    if isinstance(d, str):
        # Kullanıcı özel bir biçim vermiş olabilir → dokunma
        return d
    if isinstance(d, datetime):
        return d.strftime(f"{date_fmt} {time_fmt}" if show_time else date_fmt)
    if isinstance(d, date):
        return d.strftime(date_fmt)
    return str(d)

def show_last_update_badge(
    data_upto: Dateish = None,
    model_version: Optional[str] = None,
    last_train: Dateish = None,
    *,
    show_times: bool = False,             # tarih yanında saat de göster
    tz_label: Optional[str] = None,       # "SF" gibi kısa etiket ekle
    date_fmt: str = "%Y-%m-%d",
    time_fmt: str = "%H:%M",
    auto_prefix_v: bool = True            # "v" prefiksi ekle (v1.2.3)
) -> None:
    """
    Başlığın altında tazelik rozeti gösterir.
    Ör: "Veri: 2025-10-05’e kadar • Model v0.3.1 • Son eğitim: 2025-10-04"
    """
    parts = []

    if data_upto:
        text = _fmt_date(data_upto, show_time=show_times, date_fmt=date_fmt, time_fmt=time_fmt)
        suffix = f" ({tz_label})" if tz_label and show_times else ""
        parts.append(f"Veri: {html.escape(text)}’e kadar{suffix}")

    if model_version:
        v = model_version.strip()
        if auto_prefix_v and not v.lower().startswith("v"):
            v = f"v{v}"
        parts.append(f"Model {html.escape(v)}")

    if last_train:
        text = _fmt_date(last_train, show_time=show_times, date_fmt=date_fmt, time_fmt=time_fmt)
        suffix = f" ({tz_label})" if tz_label and show_times else ""
        parts.append(f"Son eğitim: {html.escape(text)}{suffix}")

    if not parts:
        return

    # Tema-duyarlı renkler
    base = st.get_option("theme.base") or "light"
    bg = "#0e1117" if base == "dark" else "#f6f8fa"
    fg = "#c7d1d8" if base == "dark" else "#0b1220"
    border = "#2b313e" if base == "dark" else "#e1e4e8"

    st.markdown(
        f"""
        <div style="
            background:{bg};
            color:{fg};
            border:1px solid {border};
            padding:8px 12px;
            border-radius:10px;
            display:inline-block;
            font-size:0.95rem;
            line-height:1.4;
        ">{' • '.join(parts)}</div>
        """,
        unsafe_allow_html=True,
    )
