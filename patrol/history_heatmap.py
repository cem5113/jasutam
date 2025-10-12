# patrol/history_heatmap.py
from __future__ import annotations
import pandas as pd
from typing import Tuple
from datetime import timedelta
from .approvals import list_approvals

def approvals_df(limit: int = 5000) -> pd.DataFrame:
    rows = list_approvals(limit=limit)
    if not rows:
        return pd.DataFrame(columns=["ts_sf","event_id","cells"])
    df = pd.DataFrame(rows)
    if "ts_sf" in df.columns:
        df["ts_sf"] = pd.to_datetime(df["ts_sf"], errors="coerce")
        df["day"] = df["ts_sf"].dt.date
    # cells list -> explode
    if "cells" in df.columns:
        df = df.explode("cells")
        df["GEOID"] = df["cells"].astype(str)
    return df

def patrol_intensity(d1: pd.Timestamp, d2: pd.Timestamp) -> pd.DataFrame:
    df = approvals_df()
    if df.empty or "day" not in df.columns or "GEOID" not in df.columns:
        return pd.DataFrame(columns=["GEOID","count"])
    m = (pd.to_datetime(df["day"]) >= pd.to_datetime(d1).normalize()) & \
        (pd.to_datetime(df["day"]) <= pd.to_datetime(d2).normalize())
    g = (df[m].groupby("GEOID", as_index=False)["event_id"]
            .count().rename(columns={"event_id":"count"}))
    return g.sort_values("count", ascending=False)

def patrol_heat_matrix(d1: pd.Timestamp, d2: pd.Timestamp) -> pd.DataFrame:
    """
    Gün x GEOID pivot (devriye yoğunluğu).
    """
    df = approvals_df()
    if df.empty or "day" not in df.columns or "GEOID" not in df.columns:
        return pd.DataFrame()
    m = (pd.to_datetime(df["day"]) >= pd.to_datetime(d1).normalize()) & \
        (pd.to_datetime(df["day"]) <= pd.to_datetime(d2).normalize())
    p = (df[m].groupby(["day","GEOID"], as_index=False)["event_id"].count()
           .pivot(index="day", columns="GEOID", values="event_id").fillna(0))
    return p
