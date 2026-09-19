"""
Freight Rate Prediction - End-to-End Training & Inference Pipeline
==================================================================
Best model: HistGradientBoostingRegressor (L1 / absolute_error loss)
            trained on rate-per-mile (posted_rate / distance) target
            with target-encoded lane features.

Validation strategy: time-based split (Jan-Aug train, Sep-Oct holdout).
Final model: retrained on all labeled data (Jan-Oct) before inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "train-test.csv"
VALID_FILE = DATA_DIR / "validation.csv"
DECEMBER_FILE = DATA_DIR / "december-chart-inputs.csv"
TEMPLATE_FILE = DATA_DIR / "validation-predictions-template.csv"
SUBMISSION_FILE = Path("validation_predictions.csv")

# December fixed inputs (no lat/lon in that file)
DECEMBER_PICKUP = "Lexington"
DECEMBER_DELIVERY = "Fort Wayne"
DECEMBER_PICKUP_LAT, DECEMBER_PICKUP_LON = 36.99152, -84.99876
DECEMBER_DELIVERY_LAT, DECEMBER_DELIVERY_LON = 41.31561, -85.36206

CAT_FEATURES = ["pickup", "delivery", "equipment"]

# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, city_coords: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Fill coordinates for december-chart-inputs (no lat/lon cols there)
    if city_coords is not None:
        if "pickup_lat" not in df.columns or df["pickup_lat"].isna().all():
            df["pickup_lat"] = df["pickup"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[0])
            df["pickup_lon"] = df["pickup"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[1])
        if "delivery_lat" not in df.columns or df["delivery_lat"].isna().all():
            df["delivery_lat"] = df["delivery"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[0])
            df["delivery_lon"] = df["delivery"].map(lambda x: city_coords.get(x, (np.nan, np.nan))[1])

    # Haversine distance & bearing
    lat1, lon1 = np.radians(df["pickup_lat"]), np.radians(df["pickup_lon"])
    lat2, lon2 = np.radians(df["delivery_lat"]), np.radians(df["delivery_lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["haversine_miles"] = 3956.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    df["bearing"] = np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )

    # Distance transforms
    dist = np.maximum(0, df["distance"])
    df["log_distance"] = np.log1p(dist)
    df["sqrt_distance"] = np.sqrt(dist)
    df["dist_diff"] = df["distance"] - df["haversine_miles"]
    df["circuitous_ratio"] = df["distance"] / (df["haversine_miles"] + 1.0)

    # Weight features (impute median)
    w = np.maximum(0, df["weight"].fillna(df["weight"].median()))
    df["log_weight"] = np.log1p(w)
    df["dist_weight"] = df["distance"] * w
    df["weight_per_mile"] = w / (df["distance"] + 1e-5)

    # Market & quote signal interactions (may be absent in december-chart-inputs)
    if "market_index" in df.columns and "quote_signal" in df.columns:
        df["market_x_quote"] = df["market_index"] * df["quote_signal"]
        df["market_x_dist"] = df["market_index"] * df["distance"]
        df["quote_x_dist"] = df["quote_signal"] * df["distance"]
    else:
        df["market_index"] = np.nan
        df["quote_signal"] = np.nan
        df["market_x_quote"] = np.nan
        df["market_x_dist"] = np.nan
        df["quote_x_dist"] = np.nan

    # Cyclical temporal encodings
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    for col in CAT_FEATURES:
        df[col] = df[col].astype("category")

    return df


def add_target_encodings(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame | None,
    target_col: str,
    group_cols: list[str],
    prior_weight: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Smoothed target encoding (training stats only, applied to inference)."""
    global_mean = train_df[target_col].mean()
    encodings: dict = {}
    for col in group_cols:
        stats = train_df.groupby(col)[target_col].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + prior_weight * global_mean) / (
            stats["count"] + prior_weight
        )
        enc_dict = smooth.to_dict()
        encodings[col] = (enc_dict, global_mean)
        feature_name = f"{col}_te_rpm"
        train_df[feature_name] = train_df[col].map(enc_dict).fillna(global_mean)
        if infer_df is not None:
            infer_df[feature_name] = infer_df[col].map(enc_dict).fillna(global_mean)
    return train_df, infer_df, encodings


def get_features(df: pd.DataFrame) -> list[str]:
    exclude = {"load_id", "date", "posted_rate", "rate_per_mile", "lane"}
    return [c for c in df.columns if c not in exclude]


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(**kwargs) -> HistGradientBoostingRegressor:
    params = dict(
        loss="absolute_error",
        max_iter=700,
        learning_rate=0.035,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.5,
        categorical_features=CAT_FEATURES,
        random_state=42,
        n_iter_no_change=50,
        validation_fraction=0.05,
    )
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Freight Rate Prediction — Training Pipeline")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    train_raw = pd.read_csv(TRAIN_FILE)
    valid_raw = pd.read_csv(VALID_FILE)
    dec_raw = pd.read_csv(DECEMBER_FILE)

    print(f"  Train: {train_raw.shape}, Valid: {valid_raw.shape}, December: {dec_raw.shape}")

    # City coordinate lookup (from training data — covers all training cities)
    city_coords: dict = {}
    for df_, lat_col, lon_col, city_col in [
        (train_raw, "pickup_lat", "pickup_lon", "pickup"),
        (train_raw, "delivery_lat", "delivery_lon", "delivery"),
        (valid_raw, "pickup_lat", "pickup_lon", "pickup"),
        (valid_raw, "delivery_lat", "delivery_lon", "delivery"),
    ]:
        if lat_col in df_.columns:
            sub = df_.dropna(subset=[lat_col, lon_col]).drop_duplicates(subset=[city_col])
            for _, row in sub.iterrows():
                city_coords[row[city_col]] = (row[lat_col], row[lon_col])
    # Hardcode December fixed cities in case they appear
    city_coords.setdefault(DECEMBER_PICKUP, (DECEMBER_PICKUP_LAT, DECEMBER_PICKUP_LON))
    city_coords.setdefault(DECEMBER_DELIVERY, (DECEMBER_DELIVERY_LAT, DECEMBER_DELIVERY_LON))

    # ── Feature engineering ───────────────────────────────────────────────────
    print("[2/6] Engineering features...")
    train_eng = engineer_features(train_raw, city_coords)
    valid_eng = engineer_features(valid_raw, city_coords)
    dec_eng = engineer_features(dec_raw, city_coords)

    train_eng["rate_per_mile"] = train_eng["posted_rate"] / train_eng["distance"]

    # ── Time-based validation (Jan-Aug train → Sep-Oct holdout) ──────────────
    print("[3/6] Running time-based validation (Jan-Aug → Sep-Oct)...")
    tr = train_eng[train_eng["date"] < "2025-09-01"].copy()
    ho = train_eng[train_eng["date"] >= "2025-09-01"].copy()

    tr, ho, _ = add_target_encodings(
        tr, ho, "rate_per_mile",
        group_cols=["pickup", "delivery", "equipment"],
    )
    feats = get_features(tr)

    val_model = build_model()
    val_model.fit(tr[feats], tr["rate_per_mile"])
    ho_pred = val_model.predict(ho[feats]) * ho["distance"]

    mae  = mean_absolute_error(ho["posted_rate"], ho_pred)
    rmse = np.sqrt(mean_squared_error(ho["posted_rate"], ho_pred))
    r2   = r2_score(ho["posted_rate"], ho_pred)
    print(f"  Holdout (Sep-Oct)  MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")

    # ── Final model (all labeled data) ────────────────────────────────────────
    print("[4/6] Training final model on full labeled data (Jan-Oct)...")
    full_train = train_eng.copy()
    full_train, valid_eng, te_enc = add_target_encodings(
        full_train, valid_eng, "rate_per_mile",
        group_cols=["pickup", "delivery", "equipment"],
    )
    full_train, dec_eng, _ = add_target_encodings(
        full_train, dec_eng, "rate_per_mile",
        group_cols=["pickup", "delivery", "equipment"],
    )

    # Rebuild feature list on full data
    feats_full = get_features(full_train)
    final_model = build_model()
    final_model.fit(full_train[feats_full], full_train["rate_per_mile"])

    # ── Generate predictions ──────────────────────────────────────────────────
    print("[5/6] Generating predictions...")
    valid_pred = final_model.predict(valid_eng[feats_full]) * valid_eng["distance"]
    dec_pred   = final_model.predict(dec_eng[feats_full])   * dec_eng["distance"]

    # Validation predictions
    submission = pd.read_csv(TEMPLATE_FILE)
    pred_map = dict(zip(valid_raw["load_id"], valid_pred))
    submission["predicted_rate"] = submission["load_id"].map(pred_map)
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"  Saved: {SUBMISSION_FILE}  ({len(submission)} rows)")

    # December chart predictions
    dec_raw["predicted_rate"] = dec_pred
    dec_raw.to_csv(DECEMBER_FILE, index=False)
    print(f"  Saved: {DECEMBER_FILE}  ({len(dec_raw)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("[6/6] Done.")
    print(f"\n  Validation holdout  MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.4f}")
    print(f"  Predicted rate range: ${valid_pred.min():.0f} – ${valid_pred.max():.0f}")
    print(f"\nNext: python score.py --predictions {SUBMISSION_FILE} "
          f"--december-predictions {DECEMBER_FILE}")


if __name__ == "__main__":
    main()
