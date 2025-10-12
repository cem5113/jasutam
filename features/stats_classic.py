# features/stats_classic.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List

# ---------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------
def _pick_category_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("category_grouped", "category", "subcategory_grouped", "subcategory", "crime_type"):
        if c in df.columns:
            return c
    return None

def _score_col(df: pd.DataFrame) -> Optional[str]:
    """
    Skor kolonunu otomatik seç:
      1) pred_expected  2) pred_p_occ (ortalama)  3) crime_count
    Yoksa None döndür → sayım temelli fallback (_y_col) çalışır.
    """
    if "pred_expected" in df.columns: return "pred_expected"
    if "pred_p_occ"   in df.columns:  return "pred_p_occ"
    if "crime_count"  in df.columns:  return "crime_count"
    return None

def _y_col(df: pd.DataFrame) -> str:
    """
    Sayım tabanlı özetler için güvenli hedef kolon.
    Yoksa her satırı 1 sayar.
    """
    for c in ("crime_count", "count", "y", "events"):
        if c in df.columns:
            return c
    df["_ones"] = 1
    return "_ones"

def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # date
    if "date" not in out.columns:
        if "datetime" in out.columns:
            out["date"] = pd.to_datetime(out["datetime"], errors="coerce")
        else:
            for cand in ("timestamp", "occurred_at", "reported_at"):
                if cand in out.columns:
                    out["date"] = pd.to_datetime(out[cand], errors="coerce")
                    break
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # event_hour
    if "event_hour" not in out.columns:
        if "datetime" in out.columns:
            out["event_hour"] = pd.to_datetime(out["datetime"], errors="coerce").dt.hour
        elif "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["event_hour"] = out["date"].dt.hour
        else:
            out["event_hour"] = 0

    # day_of_week / day_name
    if "day_of_week" not in out.columns:
        if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["day_of_week"] = out["date"].dt.dayofweek  # 0=Mon
        else:
            out["day_of_week"] = 0
    if "day_name" not in out.columns:
        try:
            out["day_name"] = out["date"].dt.day_name()
        except Exception:
            # Türkçe kısaltma fallback
            dow_map = {0: "Pzt", 1: "Sal", 2: "Çar", 3: "Per", 4: "Cum", 5: "Cmt", 6: "Paz"}
            out["day_name"] = out["day_of_week"].map(dow_map)

    # month
    if "month" not in out.columns:
        if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["month"] = out["date"].dt.month
        else:
            out["month"] = 1

    return out

def _apply_filters(
    df: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    geoids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = _ensure_time_cols(df)

    # tarih aralığı
    if date_range and "date" in out.columns:
        d1, d2 = date_range
        d1 = pd.to_datetime(d1) if d1 is not None else None
        d2 = pd.to_datetime(d2) if d2 is not None else None
        if d1 is not None:
            out = out[out["date"] >= d1]
        if d2 is not None:
            out = out[out["date"] <= d2]

    # GEOID filtresi
    if geoids and "GEOID" in out.columns:
        out["GEOID"] = out["GEOID"].astype(str)
        gset = set(map(str, geoids))
        out = out[out["GEOID"].isin(gset)]

    # kategori filtresi (var olan ilk uygun kolon)
    if categories:
        cats = set(map(str, categories))
        for c in ("category_grouped", "category", "subcategory_grouped", "subcategory", "crime_type"):
            if c in out.columns:
                out = out[out[c].astype(str).isin(cats)]
                break

    return out

# ---------------------------------------------------------------------
# Mekânsal özetler
# ---------------------------------------------------------------------
def spatial_top_geoid(
    df_raw: pd.DataFrame,
    n: int = 15,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    GEOID bazında en yüksek risk/yoğunluk listesi döner.
    Çıkış: ["GEOID", "value"]
      * pred_p_occ → mean
      * pred_expected / crime_count → sum
      * hiçbir skor yoksa satır sayımı (fallback)
    """
    d = _apply_filters(df_raw, date_range, None, categories)
    if "GEOID" not in d.columns:
        return pd.DataFrame(columns=["GEOID", "value"])

    sc = _score_col(d)
    if sc is None:
        y = _y_col(d)
        g = d.groupby("GEOID", as_index=False)[y].sum().rename(columns={y: "value"})
    else:
        if sc == "pred_p_occ":
            g = d.groupby("GEOID", as_index=False)[sc].mean().rename(columns={sc: "value"})
        else:
            g = d.groupby("GEOID", as_index=False)[sc].sum().rename(columns={sc: "value"})

    return g.sort_values("value", ascending=False).head(int(n)).reset_index(drop=True)

def spatial_summary(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    GEOID başına temel özet: toplam/ortalama skor ve son görülen tarih.
    Çıkış: GEOID, score, (opsiyonel) crime_count, last_date
    """
    if "GEOID" not in df_raw.columns:
        return pd.DataFrame()

    out = _apply_filters(df_raw, date_range, None, categories)
    sc = _score_col(out)
    agg: Dict[str, str] = {}

    if sc is None:
        y = _y_col(out); agg[y] = "sum"
    else:
        agg[sc] = "mean" if sc == "pred_p_occ" else "sum"
        if "crime_count" in out.columns and sc != "crime_count":
            agg["crime_count"] = "sum"

    res = out.groupby("GEOID", as_index=False).agg(agg)
    res = res.rename(columns={(sc or _y_col(out)): "score"}).sort_values("score", ascending=False)

    if "date" in out.columns:
        last_seen = out.groupby("GEOID")["date"].max().rename("last_date")
        res = res.merge(last_seen, on="GEOID", how="left")
    return res

# ---------------------------------------------------------------------
# Zamansal özetler
# ---------------------------------------------------------------------
def hourly_distribution(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    geoids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Saat 0–23 dağılımı (crime_count varsa sum; yoksa satır sayısı)."""
    d = _apply_filters(df_raw, date_range, geoids, categories)
    if "event_hour" not in d.columns:
        return pd.DataFrame({"event_hour": [], "value": []})
    y = "crime_count" if "crime_count" in d.columns else _y_col(d)
    g = d.groupby("event_hour", as_index=False)[y].sum().rename(columns={y: "value"})
    return g.sort_values("event_hour")

def dow_distribution(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    geoids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Haftanın günleri (0=Mon … 6=Sun) dağılımı + gün adı."""
    d = _apply_filters(df_raw, date_range, geoids, categories)
    if "day_of_week" not in d.columns:
        return pd.DataFrame({"day_of_week": [], "day_name": [], "value": []})
    y = "crime_count" if "crime_count" in d.columns else _y_col(d)
    g = d.groupby("day_of_week", as_index=False)[y].sum().rename(columns={y: "value"})
    dow_map = {0:"Pzt",1:"Sal",2:"Çar",3:"Per",4:"Cum",5:"Cmt",6:"Paz"}
    g["day_name"] = g["day_of_week"].map(dow_map)
    return g.sort_values("day_of_week")

def heatmap_day_hour(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    geoids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Gün(0–6) × saat(0–23) matrisi — haftanın günü bazlı.
    (UI’de gün-tarih bazlı ısı isteniyorsa `time_distributions` içindeki 'heat' kullanılır.)
    """
    d = _apply_filters(df_raw, date_range, geoids, categories)
    if "event_hour" not in d.columns or "day_of_week" not in d.columns:
        return pd.DataFrame()
    y = "crime_count" if "crime_count" in d.columns else _y_col(d)
    p = (d.groupby(["day_of_week", "event_hour"], as_index=False)[y].sum()
           .pivot(index="day_of_week", columns="event_hour", values=y)
           .fillna(0))
    return p.sort_index().sort_index(axis=1)

# ---------------------------------------------------------------------
# Tür (kategori) özetleri
# ---------------------------------------------------------------------
def type_distribution(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Suç türleri (category*) dağılımı — ilk 20."""
    d = _apply_filters(df_raw, date_range, None, categories)
    col = _pick_category_col(d)
    if not col:
        return pd.DataFrame(columns=["type", "value"])

    if "crime_count" in d.columns:
        g = d.groupby(col, as_index=False)["crime_count"].sum().rename(columns={col: "type", "crime_count": "value"})
    else:
        y = _y_col(d)
        g = d.groupby(col, as_index=False)[y].sum().rename(columns={col: "type", y: "value"})
    return g.sort_values("value", ascending=False).head(20).reset_index(drop=True)

# ---------------------------------------------------------------------
# Kapsayıcı özet (UI uyumlu)
# ---------------------------------------------------------------------
def time_distributions(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    geoids: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Saat / gün / (opsiyonel ay) dağılımları ve **tarih × saat** ısı matrisi döner.
    UI beklentileri:
      - by_hour:   ["event_hour","value"]
      - by_dow:    ["day_of_week","day_name","value"]
      - heat:      index=tarih (day), columns=hour(0..23)
    """
    df = _apply_filters(df_raw, date_range, geoids, categories)
    y = _y_col(df)

    # Saatlik
    by_hour = (df.groupby("event_hour", as_index=False)[y].sum()
                 .rename(columns={y: "value"})
                 .sort_values("event_hour"))

    # Gün (0=Mon) + isim
    by_dow = (df.groupby("day_of_week", as_index=False)[y].sum()
                .rename(columns={y: "value"}))
    dow_map = {0:"Pzt",1:"Sal",2:"Çar",3:"Per",4:"Cum",5:"Cmt",6:"Paz"}
    by_dow["day_name"] = by_dow["day_of_week"].map(dow_map)
    by_dow = by_dow.sort_values("day_of_week")

    # Ay (opsiyonel; bazı UI'lar kullanmıyor)
    by_month = (df.groupby("month", as_index=False)[y].sum()
                  .rename(columns={y: "value"})
                  .sort_values("month"))

    # Tarih × saat ısı matrisi (UI: son 7 günü .iloc[-7:,:] ile kullanıyor)
    heat = (df.assign(day=df["date"].dt.date)
              .groupby(["day", "event_hour"], as_index=False)[y].sum()
              .pivot(index="day", columns="event_hour", values=y)
              .sort_index())

    # tüm saat kolonlarını 0..23 tamamla
    for h in range(24):
        if h not in heat.columns:
            heat[h] = 0
    heat = heat.reindex(sorted(heat.columns), axis=1)

    return {"by_hour": by_hour, "by_dow": by_dow, "by_month": by_month, "heat": heat}

def offense_breakdown(
    df_raw: pd.DataFrame,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Tür bazında toplam (ilk uygun kategori kolonu).
    Çıkış: ["offense","value"]
    """
    df = _apply_filters(df_raw, date_range, None, None)
    y = _y_col(df)
    for c in ("category_grouped", "category", "subcategory_grouped", "subcategory", "crime_type"):
        if c in df.columns:
            bk = (df.groupby(c, as_index=False)[y].sum()
                    .rename(columns={y: "value"})
                    .sort_values("value", ascending=False))
            return bk.rename(columns={c: "offense"})
    return pd.DataFrame(columns=["offense", "value"])
