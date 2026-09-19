import pandas as pd
import numpy as np

def run_eda():
    train = pd.read_csv('data/train-test.csv', parse_dates=['date'])
    valid = pd.read_csv('data/validation.csv', parse_dates=['date'])
    
    print("--- Train Data ---")
    print(train.info())
    print("\nMissing Values:")
    print(train.isnull().sum())
    print("\nDate Range:", train['date'].min(), "-", train['date'].max())
    
    print("\n--- Valid Data ---")
    print(valid.info())
    print("\nMissing Values:")
    print(valid.isnull().sum())
    print("\nDate Range:", valid['date'].min(), "-", valid['date'].max())
    
    # Check for unseen categories
    for col in ['pickup', 'delivery', 'equipment']:
        train_cats = set(train[col].unique())
        valid_cats = set(valid[col].unique())
        unseen = valid_cats - train_cats
        print(f"\nUnseen {col} in valid:", unseen)

if __name__ == '__main__':
    run_eda()
