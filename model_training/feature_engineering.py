import numpy as np
import pandas as pd

def feature_engineering(train_df, test_df):
    idx = features = train_df.columns.values[2:202]
    for df in [test_df, train_df]:
        df['sum'] = df[idx].sum(axis=1)  
        df['min'] = df[idx].min(axis=1)
        df['max'] = df[idx].max(axis=1)
        df['mean'] = df[idx].mean(axis=1)
        df['std'] = df[idx].std(axis=1)
        df['skew'] = df[idx].skew(axis=1)
        df['kurt'] = df[idx].kurtosis(axis=1)
        df['med'] = df[idx].median(axis=1)
    features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
    for feature in features:
        train_df['r2_'+feature] = np.round(train_df[feature], 2)
        test_df['r2_'+feature] = np.round(test_df[feature], 2)
        train_df['r1_'+feature] = np.round(train_df[feature], 1)
        test_df['r1_'+feature] = np.round(test_df[feature], 1)
    return df