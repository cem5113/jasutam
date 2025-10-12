# ui/tab_reports.py (revize)
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from dataio.loaders import load_sf_crime_latest
from patrol.approvals import list_approvals
from features.stats_classic import time_distributions, spatial_top_geoid, offense_breakdown
from reports.builder import approvals_to_df, frames_to_csv_bytes, pack_zip
# Hızlı ZIP sarmalayıcıları (varsa)
try:
    from reports.daily import export_daily_zip
    from reports.weekly import export_weekly_zip
    from reports.monthly import export_monthly_zip
except Exception:
    export_daily_zip = export_weekly_zip = export_monthly_zip = None

# (opsiyonel) geçici hotspot; yoksa sessiz atla
try:
    from features.near_repeat import compute_temp_hotspot
except Exception:
    compute_temp_hotspot = None

# ---------- helpers ----------
def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns and "datetime" in out.columns:
        out["date"] = pd.to_datetime(out["datetime"], errors="coerce").dt.date
    out["date"] = pd.to_datetime(out.get("date", pd.NaT), errors="coerce")
    return out.dropna(subset=["date"], errors="ignore")

def _category_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("category_grouped","category","subcategory_grouped","subcategory"):
        if c in df.columns: return c
    return None

def _period_to_range(period: str, ref: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    ref = pd.to_datetime(ref).normalize()
    if period.startswith("Gün"):   start, end = ref, ref
    elif period.startswith("Haft"): start, end = ref - pd.Timedelta(days=6), ref
    else:                           start, end = ref - pd.Timedelta(days=29), ref
    return start, end

def _slice_df(df: pd.DataFrame, d1: pd.Timestamp, d2: pd.Timestamp,
              cats: List[str], ccol: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out = out[(out["date"] >= d1) & (out["date"] <= d2)]
    if ccol and cats:
        out = out[out[ccol].astype(str).isin(cats)]
    return out

def _pick_score_col(df: pd.DataFrame) -> str:
    if "pred_p_occ" in df.columns:    return "pred_p_occ"     # olasılık → mean
    if "pred_expected" in df.columns: return "pred_expected"  # beklenen → sum
    if "crime_count" in df.columns:   return "crime_count"
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return num[0] if num else df.columns[0]

def _geo_aggregate(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    if len(df) == 0 or "GEOID" not in df.columns:
        return pd.DataFrame(columns=["GEOID", score_col, "risk_norm"])
    if score_col == "pred_p_occ":
        g = df.groupby("GEOID", as_index=False)[score_col].mean()
    else:
        g = df.groupby("GEOID", as_index=False)[score_col].sum()
    mx = float(g[score_col].max() or 1.0)
    g["risk_norm"] = (g[score_col] / (mx if mx > 0 else 1.0)).clip(0, 1)
    return g

def _planned_counts_by_geoid(start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, int]:
    rows = list_approvals(limit=1000)
    counts: Dict[str,int] = {}
    for r in rows:
        try:
            ts = pd.to_datetime(r.get("ts_sf"))
            if not (start <= ts <= end): continue
            for c in r.get("cells", []):
                key = str(c)
                counts[key] = counts.get(key, 0) + 1
        except Exception:
            continue
    return counts

def _temp_hotspot_map(df_win: pd.DataFrame) -> Dict[str, float]:
    if compute_temp_hotspot is None or len(df_win) == 0:
        return {}
    try:
        tdf = compute_temp_hotspot(df_win, window_days=2, baseline_days=30)  # GEOID,temp_score
        return dict(zip(tdf["GEOID"].astype(str), tdf["temp_score"]))
    except Exception:
        return {}

def _suggest_for_geoid(geoid: str, planned: int, risk_norm: float, temp_score: Optional[float]) -> str:
    """
    Basit öneri dili:
      hedef ~ risk_norm * 12  (0..12 kaba ölçek)  → ±1 tolerans
      temp_score >=1.5 ise artış cümlesi güçlenir
    """
    target = int(np.ceil(risk_norm * 12))
    lo, hi = max(target-1, 0), max(target+1, 1)

    why_bits = []
    if temp_score is not None and np.isfinite(temp_score):
        if temp_score >= 1.5:    why_bits.append("son 48 saatte belirgin artış")
        elif temp_score >= 1.1:  why_bits.append("yakın dönemde hafif artış")
    if   risk_norm >= 0.75:      why_bits.append("risk seviyesi yüksek")
    elif risk_norm <= 0.25:      why_bits.append("risk seviyesi düşük")
    why = "; ".join(why_bits) if why_bits else "son dönem risk özeti"

    if planned < lo:
        return f"GEOID {geoid}: {planned} planlı; {lo}–{hi} **düşünülebilir**. Neden: {why}."
    if planned > hi:
        return f"GEOID {geoid}: {planned} planlı; {lo}–{hi} düzeyine **azaltılabilir**. Neden: {why}."
    return f"GEOID {geoid}: {planned} planlı; mevcut seviye **korunabilir**. Neden: {why}."

# ---------- UI ----------
def render():
    st.subheader("🧾 Raporlar & Operasyonel Öneriler")

    # veri
    try:
        df_raw, _ = load_sf_crime_latest()
    except Exception as e:
        st.error("Veri yüklenemedi."); st.exception(e); return
    df_raw = _ensure_time(df_raw)

    left, mid, right = st.columns([1.12, 1.8, 1.25])

    # --- Sol: filtreler
    with left:
        st.markdown("**Rapor türü & filtreler**")
        period = st.radio("Rapor türü", ["Günlük", "Haftalık", "Aylık"], index=0)
        latest = pd.to_datetime(df_raw["date"], errors="coerce").dropna().max() if "date" in df_raw.columns else pd.Timestamp.today()
        ref_day = st.date_input("Referans gün", value=(latest.date() if pd.notnull(latest) else pd.Timestamp.today().date()))
        d1, d2 = _period_to_range(period, pd.to_datetime(ref_day))

        ccol = _category_col(df_raw)
        cats = sorted(df_raw[ccol].dropna().astype(str).unique()) if ccol else []
        pick_cats = st.multiselect("Suç türü (opsiyonel)", cats, default=[])

        mode = st.radio("Gösterim", ["Planlanan devriyeler", "Suç gerçekleşmeleri"], index=0,
                        help="İcra edilen devriyeler logu geldiğinde üçüncü seçenek eklenecek.")
        k = st.number_input("Top-K GEOID", min_value=5, max_value=50, value=10, step=5)

    # --- Veri kesiti
    df_win = _slice_df(df_raw, d1, d2, pick_cats, ccol)
    score_col = _pick_score_col(df_win)
    geo_scores = _geo_aggregate(df_win, score_col)
    planned_map = _planned_counts_by_geoid(d1, d2)
    temp_map = _temp_hotspot_map(df_win)

    # --- Orta panel
    with mid:
        if mode == "Planlanan devriyeler":
            st.markdown("**Planlanan devriyeler (onay kayıtları)**")
            appr_df = approvals_to_df(list_approvals(limit=1000))
            if len(appr_df) > 0:
                st.dataframe(appr_df.sort_values("ts_sf", ascending=False),
                             use_container_width=True, height=380)
            else:
                st.info("Henüz onay kaydı yok.")
        else:
            st.markdown("**Suç yoğunluğu özeti**")
            top_geo = spatial_top_geoid(df_win, n=int(k))
            st.dataframe(top_geo, use_container_width=True, height=220)
            sums = time_distributions(df_win)
            st.caption("Saatlik dağılım"); st.line_chart(sums["by_hour"].set_index("event_hour"))
            st.caption("Gün × saat ısı (son 7 gün)")
            st.dataframe(sums["heat"].fillna(0).iloc[-7:, :], use_container_width=True, height=180)

    # --- Sağ: öneriler + indirme
    with right:
        st.markdown("**Öneriler (olasılıklı dil)**")
        if len(geo_scores) == 0:
            st.caption("Liste boş.")
        else:
            topk = geo_scores.sort_values("risk_norm", ascending=False).head(int(k)).copy()
            topk["planned"] = topk["GEOID"].astype(str).map(lambda g: planned_map.get(str(g), 0))
            topk["temp_score"] = topk["GEOID"].astype(str).map(lambda g: temp_map.get(str(g), np.nan))

            for _, r in topk.iterrows():
                s = _suggest_for_geoid(
                    geoid=str(r["GEOID"]),
                    planned=int(r.get("planned", 0) or 0),
                    risk_norm=float(r.get("risk_norm", 0.0) or 0.0),
                    temp_score=(None if pd.isna(r.get("temp_score", np.nan)) else float(r["temp_score"]))
                )
                st.write("• " + s)

            st.markdown("---")
            st.markdown("**Rapor Çıkışı (CSV ZIP)**")

            files: Dict[str, pd.DataFrame] = {}
            # approvals
            files["planned_approvals.csv"] = approvals_to_df(list_approvals(limit=1000))
            # suç özetleri
            try:
                sums = time_distributions(df_win)
                br = offense_breakdown(df_win)
                files["by_hour.csv"] = sums["by_hour"]
                files["by_dow.csv"] = sums["by_dow"]
                files["heatmatrix.csv"] = sums["heat"].reset_index().rename(columns={"index":"day"})
                files["offense_breakdown.csv"] = br
                files[f"top{int(k)}_geoid.csv"] = spatial_top_geoid(df_win, n=int(k))
            except Exception:
                pass

            # Top-K + öneriler
            out_sugg = topk[["GEOID", score_col, "risk_norm", "planned", "temp_score"]].copy()
            out_sugg["suggestion"] = out_sugg.apply(
                lambda r: _suggest_for_geoid(
                    geoid=str(r["GEOID"]),
                    planned=int(r.get("planned", 0) or 0),
                    risk_norm=float(r.get("risk_norm", 0.0) or 0.0),
                    temp_score=(None if pd.isna(r.get("temp_score", np.nan)) else float(r["temp_score"]))
                ),
                axis=1
            )
            files["topk_with_suggestions.csv"] = out_sugg

            csv_blobs = frames_to_csv_bytes(files)
            zip_blob = pack_zip(csv_blobs)
            st.download_button(
                "⬇️ ZIP indir (CSV raporlar)",
                data=zip_blob,
                file_name=f"reports_{period.lower()}_{d1.date()}_{d2.date()}.zip",
                mime="application/zip",
                use_container_width=True
            )

            # Hazır sarmalayıcılar (varsa)
            if any([export_daily_zip, export_weekly_zip, export_monthly_zip]):
                st.markdown("---")
                st.markdown("**Hızlı ZIP Dışa Aktarım**")
                c1, c2, c3 = st.columns(3)
                if export_daily_zip and c1.button("📦 Günlük ZIP"):
                    path = export_daily_zip(df_raw, end=d2, cats=(pick_cats or None))
                    st.success(f"Kaydedildi: {path}")
                    try:
                        st.download_button("⬇️ Günlük ZIP indir", data=Path(path).read_bytes(),
                                           file_name=Path(path).name, mime="application/zip")
                    except Exception:
                        pass
                if export_weekly_zip and c2.button("📦 Haftalık ZIP"):
                    path = export_weekly_zip(df_raw, end=d2, cats=(pick_cats or None))
                    st.success(f"Kaydedildi: {path}")
                    try:
                        st.download_button("⬇️ Haftalık ZIP indir", data=Path(path).read_bytes(),
                                           file_name=Path(path).name, mime="application/zip")
                    except Exception:
                        pass
                if export_monthly_zip and c3.button("📦 Aylık ZIP"):
                    path = export_monthly_zip(df_raw, end=d2, cats=(pick_cats or None))
                    st.success(f"Kaydedildi: {path}")
                    try:
                        st.download_button("⬇️ Aylık ZIP indir", data=Path(path).read_bytes(),
                                           file_name=Path(path).name, mime="application/zip")
                    except Exception:
                        pass

    st.caption(
        f"Aralık (SF): {d1.strftime('%Y-%m-%d')} → {d2.strftime('%Y-%m-%d')} • "
        f"Skor: {score_col} • Plan kaynağı: approvals.jsonl"
    )
