# features/time_windows.py
from __future__ import annotations
import pandas as pd
from typing import Tuple

def daily(ref: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    r = pd.to_datetime(ref).normalize()
    return r, r

def last_n_days(ref: pd.Timestamp, n: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    r = pd.to_datetime(ref).normalize()
    return (r - pd.Timedelta(days=n-1)).normalize(), r
