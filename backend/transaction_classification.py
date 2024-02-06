import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.lightgbm


# Set the tracking URI to reference the remote workspace
TRACKING_SERVER_HOST = "ec2-16-171-52-118.eu-north-1.compute.amazonaws.com"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000") 
print(f"Tracking Server URI: '{mlflow.get_tracking_uri()}'")
model_run_id = os.environ.get('ML_MODEL_RUN_ID')
print(model_run_id)

if model_run_id=="defaultvalue": #for local testing
    model_uri = "runs:/ef04c6d7f2714b8683fea3dbd249239e/model"
else:
    model_uri =  "runs:/"+model_run_id+"/model"

print(model_run_id)

# def feature_engineering(df, idx, features):
#     for df in [df]:
#         df['sum'] = df[idx].sum(axis=1)  
#         df['min'] = df[idx].min(axis=1)
#         df['max'] = df[idx].max(axis=1)
#         df['mean'] = df[idx].mean(axis=1)
#         df['std'] = df[idx].std(axis=1)
#         df['skew'] = df[idx].skew(axis=1)
#         df['kurt'] = df[idx].kurtosis(axis=1)
#         df['med'] = df[idx].median(axis=1)
#     for feature in features:
#         df['r2_'+feature] = np.round(df[feature], 2)
#         df['r1_'+feature] = np.round(df[feature], 1)
#     return df

def get_transaction(df):
    features = [c for c in df.columns if c not in ['ID_code', 'target']]                            
    idx = df.columns.values[2:202]
    # engineered_data = feature_engineering(df, idx, features                                                                 )
    lgb_model = mlflow.lightgbm.load_model(model_uri)
    predictions = lgb_model.predict(df[features], num_iteration=lgb_model.best_iteration, predict_disable_shape_check=True)
    result_df = pd.concat([df, pd.DataFrame(predictions, columns=['prediction'])], axis=1)
    print(result_df.head())    
    return result_df