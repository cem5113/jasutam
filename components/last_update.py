# components/last_update.py
from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Optional, Union, Callable
import html
import streamlit as st

# ——— Opsiyonel SF zamanı yardımcıları (varsa kullan, yoksa UTC fallback) ———
def _now_sf_fallback() -> datetime:
    """
    SF saatini döndürür. Eğer utils.tz.now_sf varsa onu kullanır,
    yoksa constants.SF_TZ_OFFSET’e, o da yoksa UTC’ye düşer.
    """
    try:
        from utils.tz import now_sf  # tercih edilen
        return now_sf()
    except Exception:
        try:
            from utils.constants import SF_TZ_OFFSET
            return datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)
        except Exception:
            return datetime.utcnow()

Dateish = Union[str, datetime, date, None]

def _fmt_date(
    d: Dateish,
    show_time: bool = False,
    date_fmt: str = "%Y-%m-%d",
    time_fmt: str = "%H:%M",
) -> str:
    """Dateish'i tek biçimde yaz. str gelirse aynen bırak (kullanıcı özel biçim istemiş olabilir)."""
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, datetime):
        return d.strftime(f"{date_fmt} {time_fmt}" if show_time else date_fmt)
    if isinstance(d, date):
        return d.strftime(date_fmt)
    return str(d)

def show_last_update_badge(
    *,
    app_name: str = "SUTAM",
    data_upto: Dateish = None,                 # "YYYY-MM-DD" ya da datetime/str
    model_version: Optional[str] = None,       # "v0.3.1" ya da "0.3.1"
    last_train: Dateish = None,                # "YYYY-MM-DD" ya da datetime/str
    daily_update_hour_sf: Optional[int] = 19,  # her gün hedef saat (SF)
    # Görünüm seçenekleri
    show_times: bool = False,                  # tarih yanında saat göster
    tz_label: Optional[str] = "SF",            # "(SF)" etiketi
    date_fmt: str = "%Y-%m-%d",
    time_fmt: str = "%H:%M",
    auto_prefix_v: bool = True,                # model_version başına "v" koy
    # Aksiyon butonları
    show_actions: bool = True,
    on_pipeline_click: Optional[Callable[[], None]] = None,  # tıklandığında çalışacak fonksiyon (opsiyonel)
) -> None:
    """
    Başlığın altında tazelik rozeti + opsiyonel aksiyon butonları gösterir.

    Ör: "SUTAM • Veri: 2025-10-05’e kadar • Model v0.3.1 • Son eğitim: 2025-10-04 (SF)
         • Günlük güncellenir: ~19:00 (SF) • Şu an (SF): 2025-10-05 18:42"
    """
    # --- zaman/state hazırlığı ---
    now_sf = _now_sf_fallback()
    st.session_state.setdefault("last_reload_at_sf", _fmt_date(now_sf, True, date_fmt, time_fmt))

    if data_upto:
        st.session_state["data_upto_sf"] = _fmt_date(data_upto, show_time=False, date_fmt=date_fmt, time_fmt=time_fmt)
    else:
        st.session_state.setdefault("data_upto_sf", "-")

    parts = []

    # Sol baş: App adı
    parts.append(f"<strong style='font-size:.98rem'>{html.escape(app_name)}</strong>")

    # Veri tazeliği
    if data_upto or st.session_state.get("data_upto_sf"):
        du_txt = st.session_state.get("data_upto_sf", "-")
        suffix = f" ({tz_label})" if tz_label and show_times else ""
        parts.append(f"Veri: <b>{html.escape(du_txt)}</b>’e kadar{suffix}")

    # Model versiyonu
    if model_version:
        v = model_version.strip()
        if auto_prefix_v and not v.lower().startswith("v"):
            v = f"v{v}"
        parts.append(f"Model: <b>{html.escape(v)}</b>")

    # Son eğitim
    if last_train:
        lt_txt = _fmt_date(last_train, show_time=show_times, date_fmt=date_fmt, time_fmt=time_fmt)
        suffix = f" ({tz_label})" if tz_label and show_times else ""
        parts.append(f"Son eğitim: <b>{html.escape(lt_txt)}</b>{suffix}")

    # Günlük güncelleme saati (opsiyonel)
    if daily_update_hour_sf is not None:
        parts.append(f"Günlük güncellenir: ~{int(daily_update_hour_sf):02d}:00{f' ({tz_label})' if tz_label else ''}")

    # Sağ uç: şu anki SF zamanı
    now_txt = _fmt_date(now_sf, show_time=True, date_fmt=date_fmt, time_fmt=time_fmt)
    parts.append(f"<span style='color:#6b7280'>Şu an ({tz_label or ''}): {html.escape(now_txt)}</span>")

    # Tema-duyarlı renkler
    try:
        base = st.get_option("theme.base") or "light"
    except Exception:
        base = "light"
    bg = "#0e1117" if base == "dark" else "#f8fafc"
    fg = "#d1d5db" if base == "dark" else "#0b1220"
    border = "#2b313e" if base == "dark" else "#e5e7eb"

    # Rozet
    st.markdown(
        f"""
        <div style="
            background:{bg};
            color:{fg};
            border:1px solid {border};
            padding:.55rem .7rem;
            border-radius:.6rem;
            display:flex; gap:.6rem; flex-wrap:wrap;
            font-size:.88rem; line-height:1.35;">
            {' • '.join(parts)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Aksiyonlar (opsiyonel) ---
    if show_actions:
        c1, c2, c3 = st.columns([0.20, 0.22, 0.58])

        # Yeniden yükle
        with c1:
            if st.button("↻ Yeniden yükle", help="Cache temizle & veriyi yeniden oku"):
                # Burada varsa sizin cache temizleme fonksiyonlarınızı çağırın (opsiyonel)
                st.session_state["last_reload_at_sf"] = _fmt_date(_now_sf_fallback(), True, date_fmt, time_fmt)
                # Streamlit >= 1.30
                try:
                    st.rerun()
                except AttributeError:
                    # Eski sürüm desteği
                    st.experimental_rerun()

        # Pipeline tetikleyici
        with c2:
            if st.button("⚙️ Full pipeline", help="Tam ETL/ML hattını tetikle"):
                if callable(on_pipeline_click):
                    try:
                        on_pipeline_click()
                        st.success("Pipeline tetikleme isteği gönderildi.")
                    except Exception as e:
                        st.warning(f"Pipeline tetiklenemedi: {e}")
                else:
                    st.info("Pipeline tetik isteği (mock).")

        # Son reload bilgisi
        with c3:
            st.caption(f"Son yeniden yükleme (SF): {st.session_state.get('last_reload_at_sf', '-')}")
