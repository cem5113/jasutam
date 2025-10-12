# reports/builder.py
from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from datetime import datetime
import json
import pandas as pd

from services.tz import now_sf
from patrol.approvals import list_approvals

# -------------------------------------------------------------------
# Genel yardımcılar
# -------------------------------------------------------------------
OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _pick_value_col(df: pd.DataFrame) -> str:
    for c in ("pred_expected", "pred_p_occ", "crime_count"):
        if c in df.columns:
            return c
    return df.columns[0]

def _cat_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("category_grouped", "category", "subcategory_grouped", "subcategory"):
        if c in df.columns:
            return c
    return None

def _date_mask(df: pd.DataFrame, d1: pd.Timestamp, d2: pd.Timestamp) -> pd.Series:
    if "date" not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    d = pd.to_datetime(df["date"], errors="coerce")
    return d.between(d1, d2)

def _latlon_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    for y, x in (("lat", "lon"), ("latitude", "longitude"), ("y", "x"), ("LAT", "LON")):
        if y in df.columns and x in df.columns:
            return y, x
    return None, None

def _mk_name(prefix: str, ext: str = "csv") -> str:
    ts = now_sf().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"

def _list_to_csv(val) -> str:
    """list -> 'a,b,c', None -> '', str -> str"""
    if isinstance(val, list):
        return ",".join(map(str, val))
    if val is None:
        return ""
    return str(val)

# -------------------------------------------------------------------
# ZIP / CSV paketleyiciler
# -------------------------------------------------------------------
def pack_zip(files: Dict[str, bytes]) -> bytes:
    """{dosya_adı: bytes} -> tek zip bytes"""
    bio = BytesIO()
    with ZipFile(bio, "w", compression=ZIP_DEFLATED) as z:
        for name, blob in files.items():
            z.writestr(name, blob)
    return bio.getvalue()

def frames_to_csv_bytes(frames: Dict[str, pd.DataFrame]) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    for name, df in frames.items():
        out[name] = df.to_csv(index=False).encode("utf-8")
    return out

# -------------------------------------------------------------------
# Onaylar (approvals) tablo dönüşümleri
# -------------------------------------------------------------------
def approvals_to_df(rows: List[dict]) -> pd.DataFrame:
    """
    JSONL onay kayıtlarını tabloya çevirir.
    Alan adları farklı gelebileceği için (route_geoids/cells) toleranslıdır.
    """
    if not rows:
        return pd.DataFrame(columns=[
            "ts_sf", "event_id", "alt_id", "assignment", "teams",
            "start", "end", "coverage", "diversity", "cells"
        ])

    recs = []
    for r in rows:
        teams = _list_to_csv(r.get("teams", ""))
        cells = r.get("cells", None)
        if cells is None:
            cells = r.get("route_geoids", None)  # UI tarafında route_geoids olarak da gelebilir
        cells_csv = _list_to_csv(cells)

        recs.append({
            "ts_sf": r.get("ts_sf", ""),
            "event_id": r.get("event_id", ""),
            "alt_id": r.get("alt_id", ""),
            "assignment": r.get("assignment", ""),
            "teams": teams,
            "start": r.get("start", ""),
            "end": r.get("end", ""),
            "coverage": r.get("coverage", ""),
            "diversity": r.get("diversity", ""),
            "cells": cells_csv,
        })
    return pd.DataFrame(recs)

def planned_patrols_table(limit: int = 200) -> pd.DataFrame:
    rows = list_approvals(limit=limit)
    return approvals_to_df(rows)

# -------------------------------------------------------------------
# Günlük/haftalık temel tablolar
# -------------------------------------------------------------------
def daily_tables(df: pd.DataFrame,
                 d1: pd.Timestamp, d2: pd.Timestamp,
                 cats: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Günlük/haftalık/aylık hepsi için temel tablolar (filtreli).
    Dönenler: top_geoid, offense, by_hour, subset
    """
    vcol = _pick_value_col(df)
    mask = _date_mask(df, d1, d2)
    sub = df.loc[mask].copy()
    ccol = _cat_col(df)
    if ccol and cats:
        sub = sub[sub[ccol].astype(str).isin(cats)]

    # Top-N GEOID
    if "GEOID" in sub.columns:
        top_geoid = (sub.groupby("GEOID", as_index=False)[vcol]
                        .sum().sort_values(vcol, ascending=False).head(20))
    else:
        top_geoid = pd.DataFrame(columns=["GEOID", vcol])

    # Tür dağılımı
    if ccol:
        off = (sub.groupby(ccol, as_index=False)[vcol]
                 .sum().sort_values(vcol, ascending=False)
                 .rename(columns={ccol: "offense"}))
    else:
        off = pd.DataFrame(columns=["offense", vcol])

    # Zaman dilimi (saat) özeti
    if "event_hour" in sub.columns:
        by_hour = sub.groupby("event_hour", as_index=False)[vcol].sum()
    else:
        by_hour = pd.DataFrame(columns=["event_hour", vcol])

    return {"top_geoid": top_geoid, "offense": off, "by_hour": by_hour, "subset": sub}

# -------------------------------------------------------------------
# Heuristik öneriler
# -------------------------------------------------------------------
def heuristic_suggestions(df_sub: pd.DataFrame,
                          top_geoid: pd.DataFrame,
                          planned_df: pd.DataFrame,
                          vcol: Optional[str] = None,
                          max_msgs: int = 6) -> List[Dict]:
    """
    Basit, olasılıklı dil kullanan öneriler:
    - Planlı devriye ile riskli GEOID’ler arasındaki boşluklara 'artırılabilir'
    - Fazla yoğun devriye varsa 'azaltılabilir'
    - Dengeli ise 'korunabilir'
    Gerekçe: son X gün risk özeti + saat dilimi vurgusu.
    """
    if vcol is None:
        vcol = _pick_value_col(df_sub) if not df_sub.empty else "score"

    # plan sayacı (cells / route_geoids kabul et)
    planned_counts: Dict[str, int] = {}
    for _, r in planned_df.iterrows():
        raw = r.get("cells", "")
        if not raw:
            raw = r.get("route_geoids", "")
        for g in str(raw).split(","):
            g = g.strip()
            if not g:
                continue
            planned_counts[g] = planned_counts.get(g, 0) + 1

    msgs = []
    # 1) riskli ilk 10 hücreyi gez
    for _, row in top_geoid.head(10).iterrows():
        geoid = str(row.get("GEOID", "-"))
        try:
            risk_val = float(row.get(vcol, 0.0))
        except Exception:
            risk_val = 0.0
        planned = planned_counts.get(geoid, 0)

        # basit eşikler
        median_val = top_geoid[vcol].median() if vcol in top_geoid.columns and len(top_geoid) > 0 else 0.0
        if risk_val > 0 and planned == 0:
            txt = f"GEOID {geoid} için devriye **artırılabilir**."
            why = "Son dönemde bu bölgede risk göstergeleri yüksek görünüyor."
        elif planned >= 2 and risk_val < median_val:
            txt = f"GEOID {geoid} için devriye **azaltılabilir**."
            why = "Planlanan devriye yoğunluğuna kıyasla risk sinyali sınırlı görünüyor."
        else:
            txt = f"GEOID {geoid} için devriye **korunabilir**."
            why = "Risk/plan dengesi görece uyumlu görünüyor."

        msgs.append({
            "text": txt,
            "why": why,
            "geoid": geoid,
            "risk_value": risk_val,
            "planned": planned,
        })
        if len(msgs) >= max_msgs:
            break
    return msgs

# -------------------------------------------------------------------
# Rapor dışa aktarımı + ZIP
# -------------------------------------------------------------------
def export_report(prefix: str,
                  tables: Dict[str, pd.DataFrame],
                  suggestions: List[Dict]) -> Dict[str, str]:
    """
    CSV (tablolar) + MD (öneriler + küçük özet) dosyaları yazar.
    Dönüş: {'key': 'out/filename.csv', 'summary_md': 'out/xxx.md', ...}
    """
    out: Dict[str, str] = {}

    # CSV’ler
    for key, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            name = _mk_name(f"{prefix}_{key}", "csv")
            path = OUT_DIR / name
            df.to_csv(path, index=False)
            out[key] = str(path)

    # MD rapor
    name_md = _mk_name(f"{prefix}_summary", "md")
    md = [
        f"# {prefix.replace('_', ' ').title()} Raporu",
        f"- Oluşturma (SF): {now_sf().isoformat(timespec='seconds')}",
    ]
    if "subset" in tables and isinstance(tables["subset"], pd.DataFrame):
        sub = tables["subset"]
        md.append(f"- Kayıt sayısı: {len(sub):,}")
    if "top_geoid" in tables and isinstance(tables["top_geoid"], pd.DataFrame) and not tables["top_geoid"].empty:
        best = tables["top_geoid"].iloc[0]
        vcol = _pick_value_col(tables["top_geoid"])
        md.append(f"- En yüksek risk: GEOID {best.get('GEOID', '-')} (skor={best.get(vcol, 0)})")

    if suggestions:
        md.append("\n## Öneriler (olasılıklı)")
        for s in suggestions:
            md.append(f"- {s['text']}  \n  _Neden:_ {s['why']}")

    (OUT_DIR / name_md).write_text("\n".join(md), encoding="utf-8")
    out["summary_md"] = str(OUT_DIR / name_md)
    return out

def export_report_zip(prefix: str,
                      tables: Dict[str, pd.DataFrame],
                      suggestions: List[Dict]) -> Path:
    """
    CSV’leri ve özet MD’yi tek ZIP’e toplar, diske yazar ve ZIP yolunu döndürür.
    """
    # önce normal export (diskte csv/md)
    paths = export_report(prefix, tables, suggestions)

    # dosyaları oku -> zip bytes
    files_bytes: Dict[str, bytes] = {}
    for key, p in paths.items():
        pth = Path(p)
        files_bytes[pth.name] = pth.read_bytes()

    zip_name = _mk_name(prefix, "zip")
    zip_path = OUT_DIR / zip_name
    zip_blob = pack_zip(files_bytes)
    zip_path.write_bytes(zip_blob)
    return zip_path
