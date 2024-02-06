import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.lightgbm

TRACKING_SERVER_HOST = "ec2-13-49-77-251.eu-north-1.compute.amazonaws.com"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000") 
print(f"Tracking Server URI: '{mlflow.get_tracking_uri()}'")

model_uri = 'runs:/8b1fdd6e4b8c4c08b475225f718aca23/model'
PATH = "../mlopscasestudy24/data/"
train_df = pd.read_csv(PATH+"train.csv")
test_df = pd.read_csv(PATH+"test.csv")
print(test_df.head())

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
print(features)
idx = train_df.columns.values[2:202]

print(train_df.shape)
print(test_df.shape)
def feature_engineering(df, idx, features):
    for df in [df]:
        df['sum'] = df[idx].sum(axis=1)  
        df['min'] = df[idx].min(axis=1)
        df['max'] = df[idx].max(axis=1)
        df['mean'] = df[idx].mean(axis=1)
        df['std'] = df[idx].std(axis=1)
        df['skew'] = df[idx].skew(axis=1)
        df['kurt'] = df[idx].kurtosis(axis=1)
        df['med'] = df[idx].median(axis=1)
    for feature in features:
        df['r2_'+feature] = np.round(df[feature], 2)
        df['r1_'+feature] = np.round(df[feature], 1)
    return df

eng_df = feature_engineering(test_df, idx, features)

# model prediction
lgb_model = mlflow.lightgbm.load_model(model_uri)
predictions = lgb_model.predict(eng_df[features], num_iteration=lgb_model.best_iteration, predict_disable_shape_check=True)

print(predictions)