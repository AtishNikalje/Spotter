import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from src.features.build_features import create_features

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def train_lightgbm(X_train, y_train, X_val, y_val, cat_features=None):
    if cat_features is None:
        cat_features = 'auto'
        
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=64,
        random_state=42,
        objective='regression'
    )
    
    # We use early stopping if validation set is provided
    if X_val is not None:
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_features,
            callbacks=callbacks
        )
    else:
        model.fit(
            X_train, y_train,
            categorical_feature=cat_features
        )
        
    return model

def main():
    print("Loading data...")
    train = pd.read_csv('data/train-test.csv')
    valid_pred = pd.read_csv('data/validation.csv')
    december = pd.read_csv('data/december_chart_inputs.csv')
    
    # Validation strategy: Train Jan-Sep, Validate Oct
    # Or train on all data for the final model
    print("Feature engineering...")
    
    # Create final model using all labeled data
    train_features, med_w, med_m = create_features(train, is_train=True)
    valid_features, _, _ = create_features(valid_pred, is_train=False, median_weight=med_w, median_market=med_m)
    dec_features, _, _ = create_features(december, is_train=False, median_weight=med_w, median_market=med_m)
    
    # Define features
    features = [c for c in train_features.columns if c not in ['load_id', 'date', 'posted_rate']]
    
    X_train = train_features[features]
    y_train = train_features['posted_rate']
    
    print(f"Training on {X_train.shape[0]} rows and {X_train.shape[1]} features...")
    
    # Train final model
    model = train_lightgbm(X_train, y_train, None, None, cat_features=['pickup', 'delivery', 'equipment'])
    
    print("Making predictions...")
    
    # Generate predictions
    valid_pred['predicted_rate'] = model.predict(valid_features[features])
    december['predicted_rate'] = model.predict(dec_features[features])
    
    # Save outputs
    print("Saving outputs...")
    valid_pred[['load_id', 'predicted_rate']].to_csv('validation_predictions.csv', index=False)
    december.to_csv('data/december_chart_inputs.csv', index=False)
    
    print("Training complete! You can now run `python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv`")

if __name__ == '__main__':
    main()
