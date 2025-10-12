# ui/home.py
from __future__ import annotations
import os
import streamlit as st

from config.settings import APP_NAME, MODEL_VERSION
from services.tz import now_sf_str
from dataio.loaders import load_sf_crime_latest, load_metadata
from services.metrics import compute_kpis


def _pick_meta(meta: dict) -> tuple[str, str, str]:
    """
    Meta şeması esnek olabilir:
    - model.version / model.last_trained_at / data.last_data_refresh_at
    - veya düz alanlar: model_version / last_trained_at / last_data_refresh_at|data_refresh_at
    """
    mv = (
        meta.get("model", {}).get("version")
        or meta.get("model_version")
        or MODEL_VERSION
    )
    ltr = (
        meta.get("model", {}).get("last_trained_at")
        or meta.get("last_trained_at")
        or "—"
    )
    ldr = (
        meta.get("data", {}).get("last_data_refresh_at")
        or meta.get("last_data_refresh_at")
        or meta.get("data_refresh_at")
        or "—"
    )
    return str(mv), str(ltr), str(ldr)


def _get_secret_or_env(key: str, default: str | None = None) -> str | None:
    """Streamlit secrets > ENV fallback."""
    try:
        if key in st.secrets:
            v = st.secrets.get(key, default)  # type: ignore[attr-defined]
            return str(v) if v is not None else default
    except Exception:
        pass
    v = os.getenv(key, default)
    return str(v) if v is not None else None

def _trigger_pipeline() -> tuple[int, str]:
    import os
    repo = os.getenv("GITHUB_REPO")
    wf   = os.getenv("GITHUB_WORKFLOW", "full_pipeline.yml")
    tok  = os.getenv("GH_TOKEN")
    ref  = os.getenv("GITHUB_REF_TO_DISPATCH", "main")  # default branch'iniz master ise burayı 'master' yapın

    if not (repo and wf and tok):
        return 400, "GITHUB_REPO / GITHUB_WORKFLOW / GH_TOKEN ayarlı değil."

    try:
        import requests
    except Exception:
        return 400, "requests modülü yok. requirements.txt içine 'requests' ekleyin."

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{wf}/dispatches"
    try:
        r = requests.post(
            url,
            json={"ref": ref},             # 👈 inputs YOK
            headers={
                "Authorization": f"Bearer {tok}",
                "Accept": "application/vnd.github+json",
            },
            timeout=30,
        )
        return r.status_code, (r.text or "OK")
    except Exception as e:
        return 500, f"İstek hatası: {e}"

def render():
    st.title(APP_NAME)

    # --- Meta ---
    try:
        meta = load_metadata() or {}
    except Exception:
        meta = {}
    mv, ltr, ldr = _pick_meta(meta)

    # Üst bilgi şeridi
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", mv or "—")
    c2.metric("Son eğitim (SF)", ltr or "—")
    c3.metric("Veri güncelleme (SF)", ldr or "—")
    c4.metric("Şu an (SF)", now_sf_str())

    with st.expander("ⓘ Model/Veri Meta", expanded=False):
        st.json(meta, expanded=False)

    # --- Güncelleme / Senkronizasyon ---
    st.markdown("#### 🔄 Güncelleme / Senkronizasyon")
    colA, colB = st.columns(2)

    if colA.button("⟳ Veriyi yeniden yükle", use_container_width=True):
        # Streamlit cache temizle → bir sonraki okuma taze veri
        try:
            st.cache_data.clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        # İsteğe bağlı: hemen okuma yapıp kullanıcıya bildirelim
        try:
            _df_chk, _ = load_sf_crime_latest()
            st.success("Veri ve özetler yeniden yüklendi.")
        except Exception as e:
            st.warning("Yeniden yükleme çağrısı atıldı; sekmelerde veri okunacaktır.")
            st.exception(e)

    if colB.button("▶️ Full Data Pipeline’ı çalıştır", use_container_width=True):
        code, msg = _trigger_pipeline()
        if 200 <= code < 300:
            st.success("✅ Workflow tetiklendi. GitHub Actions üzerinden ilerlemeyi izleyebilirsiniz.")
        else:
            st.error(f"❌ Tetikleme başarısız: {code}\n{msg}")

    st.caption(
        "Not: ‘Veriyi yeniden yükle’ cache’i boşaltır ve mevcut artefact/kaynakları tekrar okutur. "
        "‘Pipeline’ ise ham veriyi toplayıp özellik/tahmin/rapor artefact’larını yeniden üretir (opsiyonel)."
    )
    st.markdown("---")

    # --- Veri yükleme + KPI ---
    try:
        df, src = load_sf_crime_latest()
    except Exception as e:
        st.error("Veri yüklenemedi")
        st.exception(e)
        st.stop()
        return

    kpi = compute_kpis(df)
    s1, s2, s3 = st.columns(3)
    s1.metric("HitRate@Top10", "—" if kpi["hitrate_top10"] is None else f"{kpi['hitrate_top10']}%")
    s2.metric("Brier", "—" if kpi["brier"] is None else f"{kpi['brier']}")

    # Kaynak etiketi
    src_label = {"artifact": "Artifact (güncel)", "release": "Release (yedek)"}.get(str(src), str(src))
    s3.metric("Veri Kaynağı", src_label)

    # Mini Model Kartı
    with st.expander("ⓘ Model Kartı (mini)", expanded=False):
        st.markdown(
            f"- **Sürüm:** {mv}\n"
            f"- **Son eğitim:** {ltr}\n"
            f"- **Veri güncelleme:** {ldr}\n"
            "- **Notlar:** Düşük olaylı hücrelerde belirsizlik yükselebilir. "
            "Saha görünümünde tahmin seviyesi (Yüksek/Orta/Düşük) sadeleştirilmiştir."
        )

    # Küçük önizleme
    with st.expander("İlk 15 satır (önizleme)", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)

    # Hızlı kontroller (yer tutucu)
    st.markdown("#### Hızlı kontroller")
    col1, col2, col3 = st.columns(3)
    col1.toggle("Tahmin katmanı (risk)", value=True, help="Açılışta risk haritası görünür.")
    col2.toggle("Geçici hotspot", value=True, help="Son olaylara dayalı anomali noktaları.")
    col3.toggle("Kalıcı hotspot", value=True, help="Uzun dönem ısı haritası.")
    st.caption("🔌 Katmanlar ileride haritaya bağlanacak. KPI'lar güncel veriden otomatik hesaplanır.")
