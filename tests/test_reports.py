# tests/test_reports.py
import pandas as pd
from reports.daily import make_daily

def test_make_daily_empty():
    df = pd.DataFrame({"GEOID":[], "date":[], "crime_count":[]})
    tables, tips = make_daily(df)
    assert set(tables.keys()) >= {"top_geoid","offense","by_hour","subset"}
    assert isinstance(tips, list)
