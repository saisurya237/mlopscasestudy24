import lightgbm as lgb
import pandas as pd
import os
import mlflow
import mlflow.lightgbm

# Set the tracking URI to reference the remote workspace
TRACKING_SERVER_HOST = "ec2-13-49-77-251.eu-north-1.compute.amazonaws.com"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000") 
print(f"Tracking Server URI: '{mlflow.get_tracking_uri()}'")
model_run_id = os.environ.get('MODEL_RUN_ID')
model_uri = "runs:/"+model_run_id+"/model" #complete the model uri
#other possible sol = model_uri = f"runs:/{mlflow.last_active_run().info.run_id}/model"

def get_transaction(df):
    lgb_model = mlflow.lightgbm.load_model(model_uri)
    predictions = lgb_model.predict(num_iteration=lgb_model.best_iteration)
    result_df = pd.concat([df, pd.DataFrame(predictions, columns=['prediction'])], axis=1)
    return result_df