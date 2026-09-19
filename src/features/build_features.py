import pandas as pd
import numpy as np

def extract_date_features(df, date_col='date'):
    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.month
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def impute_missing(df, median_weight=None, median_market=None):
    if median_weight is None:
        median_weight = df['weight'].median()
    if median_market is None:
        median_market = df['market_index'].median()
        
    df['weight'] = df['weight'].fillna(median_weight)
    df['market_index'] = df['market_index'].fillna(median_market)
    
    return df, median_weight, median_market

def create_features(df, is_train=True, median_weight=None, median_market=None):
    df = df.copy()
    df = extract_date_features(df)
    
    if is_train:
        df, median_weight, median_market = impute_missing(df)
    else:
        df, _, _ = impute_missing(df, median_weight, median_market)
        
    # Interaction features
    df['distance_weight'] = df['distance'] * df['weight']
    df['distance_per_weight'] = df['distance'] / (df['weight'] + 1e-5)
    
    # Categorical features - frequency encoding for unseen robustness
    cat_cols = ['pickup', 'delivery', 'equipment']
    
    # In a real pipeline, we'd save these frequencies from train and apply to test.
    # We will do this in the model pipeline using scikit-learn TargetEncoder or similar,
    # or let LightGBM handle categories natively. LightGBM handles categories well if they are pandas 'category' dtype.
    for col in cat_cols:
        df[col] = df[col].astype('category')
        
    return df, median_weight, median_market
