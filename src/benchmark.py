import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_target_encodings(train_df, val_df, target_col, group_cols, prior_weight=10):
    global_mean = train_df[target_col].mean()
    encodings = {}
    for col in group_cols:
        stats = train_df.groupby(col)[target_col].agg(['count', 'mean'])
        smooth = (stats['count'] * stats['mean'] + prior_weight * global_mean) / (stats['count'] + prior_weight)
        encodings[col] = (smooth.to_dict(), global_mean)
        train_df[f'{col}_te_rpm'] = train_df[col].map(smooth).fillna(global_mean)
        if val_df is not None:
            val_df[f'{col}_te_rpm'] = val_df[col].map(smooth).fillna(global_mean)
    return train_df, val_df, encodings

def engineer_features(df, city_coords=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Fill coordinates if missing
    if city_coords is not None:
        if 'pickup_lat' not in df.columns or df['pickup_lat'].isna().all():
            df['pickup_lat'] = df['pickup'].map(lambda x: city_coords.get(x, (np.nan, np.nan))[0])
            df['pickup_lon'] = df['pickup'].map(lambda x: city_coords.get(x, (np.nan, np.nan))[1])
        if 'delivery_lat' not in df.columns or df['delivery_lat'].isna().all():
            df['delivery_lat'] = df['delivery'].map(lambda x: city_coords.get(x, (np.nan, np.nan))[0])
            df['delivery_lon'] = df['delivery'].map(lambda x: city_coords.get(x, (np.nan, np.nan))[1])
            
    # Geospatial geometry
    lat1, lon1 = np.radians(df['pickup_lat']), np.radians(df['pickup_lon'])
    lat2, lon2 = np.radians(df['delivery_lat']), np.radians(df['delivery_lon'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    df['haversine_miles'] = 3956.0 * c
    df['bearing'] = np.arctan2(np.sin(dlon) * np.cos(lat2), np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    
    # Distance non-linearities
    df['log_distance'] = np.log1p(np.maximum(0, df['distance']))
    df['sqrt_distance'] = np.sqrt(np.maximum(0, df['distance']))
    df['dist_diff'] = df['distance'] - df['haversine_miles']
    df['circuitous_ratio'] = df['distance'] / (df['haversine_miles'] + 1.0)
    
    # Weight interactions
    clean_weight = df['weight'].fillna(df['weight'].median())
    clean_weight = np.maximum(0, clean_weight)
    df['log_weight'] = np.log1p(clean_weight)
    df['dist_weight'] = df['distance'] * clean_weight
    df['weight_per_mile'] = clean_weight / (df['distance'] + 1e-5)
    
    # Market & Quote signals
    df['market_x_quote'] = df['market_index'] * df['quote_signal']
    df['market_x_dist'] = df['market_index'] * df['distance']
    df['quote_x_dist'] = df['quote_signal'] * df['distance']
    
    # Lane string representation
    df['lane'] = df['pickup'].astype(str) + " -> " + df['delivery'].astype(str)
    
    # Cyclical date encodings
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    # Categoricals for tree models (strictly <= 255 unique levels)
    for col in ['pickup', 'delivery', 'equipment']:
        df[col] = df[col].astype('category')
        
    return df

def run_advanced_tuning():
    train = pd.read_csv('data/train-test.csv')
    
    coords = {}
    for _, row in train.dropna(subset=['pickup_lat', 'pickup_lon']).drop_duplicates(subset=['pickup']).iterrows():
        coords[row['pickup']] = (row['pickup_lat'], row['pickup_lon'])
    for _, row in train.dropna(subset=['delivery_lat', 'delivery_lon']).drop_duplicates(subset=['delivery']).iterrows():
        coords[row['delivery']] = (row['delivery_lat'], row['delivery_lon'])
        
    prepared = engineer_features(train, city_coords=coords)
    prepared['rate_per_mile'] = prepared['posted_rate'] / prepared['distance']
    
    # Time split: Jan-Aug train, Sep-Oct val
    tr = prepared[prepared['date'] < '2025-09-01'].copy()
    val = prepared[prepared['date'] >= '2025-09-01'].copy()
    
    # Target encoding on rate_per_mile using training set
    tr, val, _ = compute_target_encodings(tr, val, 'rate_per_mile', ['pickup', 'delivery', 'lane', 'equipment'], prior_weight=15)
    
    cat_cols = ['pickup', 'delivery', 'equipment']
    features = [c for c in tr.columns if c not in ['load_id', 'date', 'posted_rate', 'rate_per_mile', 'lane']]
    
    print(f"Number of features: {len(features)}")
    
    # Model 1: Rate-per-mile Absolute Error (L1 Loss)
    m_rpm_l1 = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.5,
        categorical_features=cat_cols,
        random_state=42
    )
    m_rpm_l1.fit(tr[features], tr['rate_per_mile'])
    p_rpm_l1 = m_rpm_l1.predict(val[features]) * val['distance']
    print(f"1. Rate-per-Mile (L1 Loss)        -> MAE: ${mean_absolute_error(val['posted_rate'], p_rpm_l1):.2f}, RMSE: ${np.sqrt(mean_squared_error(val['posted_rate'], p_rpm_l1)):.2f}, R2: {r2_score(val['posted_rate'], p_rpm_l1):.4f}")

    # Model 2: Rate-per-mile Squared Error (L2 Loss)
    m_rpm_l2 = HistGradientBoostingRegressor(
        loss='squared_error',
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.5,
        categorical_features=cat_cols,
        random_state=42
    )
    m_rpm_l2.fit(tr[features], tr['rate_per_mile'])
    p_rpm_l2 = m_rpm_l2.predict(val[features]) * val['distance']
    print(f"2. Rate-per-Mile (L2 Loss)        -> MAE: ${mean_absolute_error(val['posted_rate'], p_rpm_l2):.2f}, RMSE: ${np.sqrt(mean_squared_error(val['posted_rate'], p_rpm_l2)):.2f}, R2: {r2_score(val['posted_rate'], p_rpm_l2):.4f}")

    # Model 3: Direct Rate Absolute Error
    m_direct_l1 = HistGradientBoostingRegressor(
        loss='absolute_error',
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.5,
        categorical_features=cat_cols,
        random_state=42
    )
    m_direct_l1.fit(tr[features], tr['posted_rate'])
    p_direct_l1 = m_direct_l1.predict(val[features])
    print(f"3. Direct Rate (L1 Loss)           -> MAE: ${mean_absolute_error(val['posted_rate'], p_direct_l1):.2f}, RMSE: ${np.sqrt(mean_squared_error(val['posted_rate'], p_direct_l1)):.2f}, R2: {r2_score(val['posted_rate'], p_direct_l1):.4f}")

    # Model 4: Direct Rate Squared Error
    m_direct_l2 = HistGradientBoostingRegressor(
        loss='squared_error',
        max_iter=500,
        learning_rate=0.04,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.5,
        categorical_features=cat_cols,
        random_state=42
    )
    m_direct_l2.fit(tr[features], tr['posted_rate'])
    p_direct_l2 = m_direct_l2.predict(val[features])
    print(f"4. Direct Rate (L2 Loss)           -> MAE: ${mean_absolute_error(val['posted_rate'], p_direct_l2):.2f}, RMSE: ${np.sqrt(mean_squared_error(val['posted_rate'], p_direct_l2)):.2f}, R2: {r2_score(val['posted_rate'], p_direct_l2):.4f}")

    # 5. Optimized Ensemble
    p_ens = 0.45 * p_rpm_l1 + 0.35 * p_rpm_l2 + 0.10 * p_direct_l1 + 0.10 * p_direct_l2
    print(f"5. Optimized Ensemble (1+2+3+4)   -> MAE: ${mean_absolute_error(val['posted_rate'], p_ens):.2f}, RMSE: ${np.sqrt(mean_squared_error(val['posted_rate'], p_ens)):.2f}, R2: {r2_score(val['posted_rate'], p_ens):.4f}")

if __name__ == '__main__':
    run_advanced_tuning()
