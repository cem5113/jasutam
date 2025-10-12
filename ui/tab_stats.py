# sutam/ui/tab_stats.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np

# Matplotlib güvenli import
try:
    import matplotlib
    matplotlib.use("Agg")  # headless ortam için backend
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    _mpl_err = e

from sutam.dataio.loaders import load_sf_crime_latest
from sutam.features.stats_classic import (
    time_distributions,
    spatial_top_geoid,
    offense_breakdown,
)

# --------- yardımcılar ----------
def _category_options(df: pd.DataFrame):
    for c in ("category_grouped", "category", "subcategory_grouped", "subcategory"):
        if c in df.columns:
            return c, sorted(df[c].dropna().astype(str).unique())
    return None, []


def _latlon_cols(df: pd.DataFrame):
    for y, x in (("lat", "lon"), ("latitude", "longitude"),
                 ("LAT", "LON"), ("y", "x")):
        if y in df.columns and x in df.columns:
            return y, x
    return None, None


# --------- Ana render ---------
def render():
    if plt is None:
        st.error(
            "Grafikler için **matplotlib** gerekli. "
            "Lütfen `requirements.txt` içine `matplotlib` ekleyip yeniden deploy edin.\n\n"
            f"Teknik detay: {type(_mpl_err).__name__}: {_mpl_err}"
        )
        return

    df = load_sf_crime_latest()
    if df is None or df.empty:
        st.warning("Veri yüklenemedi.")
        return

    st.subheader("📊 Zaman Dağılımları")
    fig1 = time_distributions(df)
    st.pyplot(fig1)

    st.subheader("🗺️ Mekânsal Dağılım (Top GEOID)")
    fig2 = spatial_top_geoid(df)
    st.pyplot(fig2)

    st.subheader("⚖️ Suç Türleri Dağılımı")
    fig3 = offense_breakdown(df)
    st.pyplot(fig3)

def _safe_date_range(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    # df["date"] varsa min/max; yoksa bugünden son 30 gün varsay
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        dmin, dmax = d.min(), d.max()
        if pd.isna(dmin) or pd.isna(dmax):
            dmin = pd.Timestamp.today() - pd.Timedelta(days=30)
            dmax = pd.Timestamp.today()
    else:
        dmin = pd.Timestamp.today() - pd.Timedelta(days=30)
        dmax = pd.Timestamp.today()
    return dmin.normalize(), dmax.normalize()

# --------- ana render ----------
def render():
    st.subheader("📊 Suç İstatistikleri (Geçmiş)")

    # Veri
    try:
        df, src = load_sf_crime_latest()
    except Exception as e:
        st.error("Veri yüklenemedi.")
        st.exception(e)
        return

    left, mid, right = st.columns([1.1, 2.0, 1.2])

    # -------- Sol panel: filtreler --------
    with left:
        st.markdown("**Filtreler**")

        dmin, dmax = _safe_date_range(df)
        default_start = max(dmin, dmax - pd.Timedelta(days=30))

        d1, d2 = st.date_input(
            "Tarih aralığı",
            value=(default_start.date(), dmax.date()),
            min_value=dmin.date(),
            max_value=dmax.date(),
        )

        # gün sonunu dahil et (23:59:59)
        date_range = (
            pd.to_datetime(d1),
            pd.to_datetime(d2) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
        )

        ccol, opts = _category_options(df)
        pick_cats = st.multiselect("Suç türü", options=opts, default=[]) if opts else []

        mode = st.radio("Gösterim", ["Zamansal", "Mekânsal", "Suç Türleri"], index=0)
        st.caption("ℹ️ Grafiklerin üzerine gelerek değerleri görebilirsiniz.")

    # -------- Orta panel: grafik/harita --------
    with mid:
        if mode == "Zamansal":
            st.markdown("**Zamansal dağılımlar**")
            sums = time_distributions(df, date_range=date_range, categories=pick_cats)

            # Saatlik
            if "by_hour" in sums and not sums["by_hour"].empty:
                st.caption("Saatlik dağılım")
                st.line_chart(sums["by_hour"].set_index("event_hour"))
            else:
                st.info("Saatlik dağılım için yeterli veri yok.")

            # Günlere göre
            if "by_dow" in sums and not sums["by_dow"].empty:
                st.caption("Haftanın günlerine göre")
                st.bar_chart(sums["by_dow"].set_index("day_name"))
            else:
                st.info("Gün bazlı dağılım için yeterli veri yok.")

            # Gün × Saat ısı matrisi
            if "heat" in sums and not sums["heat"].empty:
                st.caption("Gün × Saat ısı matrisi")
                fig, ax = plt.subplots(figsize=(6, 3))
                hm = sums["heat"].fillna(0)
                im = ax.imshow(hm.to_numpy(), aspect="auto")
                ax.set_yticks(range(len(hm.index)))
                ax.set_yticklabels(hm.index)
                ax.set_xticks(range(len(hm.columns)))
                ax.set_xticklabels(hm.columns, rotation=0, fontsize=8)
                ax.set_xlabel("Saat")
                ax.set_ylabel("Gün")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Isı matrisi için yeterli veri yok.")

        elif mode == "Mekânsal":
            st.markdown("**GEOID — en yoğun bölgeler**")
            top = spatial_top_geoid(df, n=15, date_range=date_range, categories=pick_cats)
            st.dataframe(top, use_container_width=True, height=360)
            st.download_button(
                "⬇️ Top GEOID listesi (CSV)",
                top.to_csv(index=False).encode("utf-8"),
                file_name="top_geoid.csv",
                mime="text/csv",
            )

            # Opsiyonel harita (lat/lon varsa)
            lat, lon = _latlon_cols(df)
            if lat and lon and "GEOID" in df.columns:
                # seçili aralık + kategori filtresi için basit yoğunluk
                dmask = pd.to_datetime(df.get("date", pd.NaT), errors="coerce").between(date_range[0], date_range[1])
                dfx = df.loc[dmask].copy()
                if pick_cats and ccol:
                    dfx = dfx[dfx[ccol].astype(str).isin(pick_cats)]
                val_col = "crime_count" if "crime_count" in dfx.columns else None
                if val_col:
                    geo_agg = (
                        dfx.groupby("GEOID", as_index=False)
                           .agg({val_col: "sum", lat: "mean", lon: "mean"})
                    )
                    vmax = float(geo_agg[val_col].max() or 1.0)
                    geo_agg["level"] = (geo_agg[val_col] / vmax).clip(0, 1)

                    tooltip = {
                        "html": "<b>GEOID:</b> {GEOID}<br/><b>Toplam:</b> "
                                + "{"+val_col+"}" + "<br/><b>Seviye:</b> {level}",
                        "style": {"backgroundColor": "rgba(30,30,30,0.8)", "color": "white"},
                    }
                    st.pydeck_chart({
                        "initialViewState": {
                            "latitude": float(geo_agg[lat].mean()),
                            "longitude": float(geo_agg[lon].mean()),
                            "zoom": 11,
                        },
                        "layers": [{
                            "@@type": "ScatterplotLayer",
                            "data": geo_agg.to_dict("records"),
                            "get_position": f"[{lon}, {lat}]",
                            "get_radius": 90,
                            "pickable": True,
                            "opacity": 0.7,
                            "get_fill_color": "[255, (1-level)*200, (1-level)*100]",
                        }],
                        "tooltip": tooltip,
                        "mapProvider": "carto",
                    })
                    st.caption("Not: Grid poligonları hazır olduğunda PolygonLayer’a geçilecek.")
                else:
                    st.info("Harita yoğunluğu için 'crime_count' sütunu bulunamadı.")
            else:
                st.info("Harita için lat/lon sütunları yok (lat/lon veya latitude/longitude beklenir).")

        else:  # Suç Türleri
            st.markdown("**Suç türü dağılımları**")
            br = offense_breakdown(df, date_range=date_range)
            if not br.empty:
                st.bar_chart(br.set_index("offense").head(20))
                with st.expander("Tablo (ilk 100)"):
                    st.dataframe(br.head(100), use_container_width=True)
                st.download_button(
                    "⬇️ Tür dağılımı (CSV)",
                    br.to_csv(index=False).encode("utf-8"),
                    file_name="offense_breakdown.csv",
                    mime="text/csv",
                )
            else:
                st.info("Tür bilgisi bulunamadı.")

    # -------- Sağ panel: özet kutuları --------
    with right:
        st.markdown("**Özet**")
        # Filtrelenmiş alt küme
        dmask = pd.to_datetime(df.get("date", pd.NaT), errors="coerce").between(date_range[0], date_range[1])
        sub = df.loc[dmask].copy()
        if pick_cats and ccol:
            sub = sub[sub[ccol].astype(str).isin(pick_cats)]

        ycol = "crime_count" if "crime_count" in sub.columns else None
        total = int(sub[ycol].sum()) if ycol else len(sub)
        st.metric("Toplam olay", f"{total:,}")

        # yoğun saat/gün
        try:
            td = time_distributions(df, date_range=date_range, categories=pick_cats)
            if "by_hour" in td and not td["by_hour"].empty:
                top_hour = int(td["by_hour"].sort_values("value", ascending=False).iloc[0]["event_hour"])
                st.metric("En yoğun saat", f"{top_hour:02d}:00")
            if "by_dow" in td and not td["by_dow"].empty:
                top_day = str(td["by_dow"].sort_values("value", ascending=False).iloc[0]["day_name"])
                st.metric("En yoğun gün", top_day)
        except Exception:
            st.caption("Yoğunluk özeti hesaplanamadı.")

        st.markdown("---")
        st.caption("📥 İndir")
        # Zamansal modda CSV butonları
        try:
            if mode == "Zamansal":
                if "by_hour" in td and not td["by_hour"].empty:
                    st.download_button(
                        "Saatlik dağılım (CSV)",
                        td["by_hour"].to_csv(index=False).encode("utf-8"),
                        file_name="by_hour.csv",
                        mime="text/csv",
                    )
                if "heat" in td and not td["heat"].empty:
                    st.download_button(
                        "Gün × saat ısı (CSV)",
                        td["heat"].to_csv().encode("utf-8"),
                        file_name="heatmatrix.csv",
                        mime="text/csv",
                    )
        except Exception:
            pass
