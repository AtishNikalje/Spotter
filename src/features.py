from __future__ import annotations

import numpy as np
import pandas as pd


TARGET = "posted_rate"
ID_COLUMN = "load_id"


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result.pop("date"), errors="coerce")
    day_of_year = dates.dt.dayofyear.astype(float)
    day_of_week = dates.dt.dayofweek.astype(float)

    result["month"] = dates.dt.month.astype(float)
    result["day_of_week"] = day_of_week
    result["day_of_year"] = day_of_year
    result["days_since_start"] = (dates - pd.Timestamp("2025-01-01")).dt.days.astype(float)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["weekday_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    result["weight_missing"] = result["weight"].isna().astype(int)
    result["market_index_missing"] = result["market_index"].isna().astype(int)
    result["weight"] = result["weight"].fillna(result["weight"].median()).clip(lower=0)
    result["market_index"] = result["market_index"].fillna(result["market_index"].median())
    result["distance_log"] = np.log1p(result["distance"])
    result["route"] = result["pickup"].astype(str) + " -> " + result["delivery"].astype(str)
    return result.drop(columns=[ID_COLUMN, TARGET], errors="ignore")


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [column for column in ("pickup", "delivery", "equipment", "route") if column in frame]
    numeric = [column for column in frame.columns if column not in categorical]
    return numeric, categorical