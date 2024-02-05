import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from feature_engineering import feature_engineering
warnings.filterwarnings('ignore')


PATH = "../mlopscasestudy24/data/"
train_df = pd.read_csv(PATH+"train.csv")
test_df = pd.read_csv(PATH+"test.csv")

feature_engineering(train_df, test_df)

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']

X_train, X_test, y_train, y_test = train_test_split(train_df, test_df, test_size=0.2, random_state=42)

param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}

oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

# Set the tracking URI to reference the remote workspace
TRACKING_SERVER_HOST = "ec2-13-49-77-251.eu-north-1.compute.amazonaws.com"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000") 
print(f"Tracking Server URI: '{mlflow.get_tracking_uri()}'")

# Set the experiment
mlflow.set_experiment("mlflow_experiment/mlops24")
mlflow.lightgbm.autolog(log_models=True)
with mlflow.start_run():
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=44000)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
        print("Fold {}".format(fold_))
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

        num_round = 1000000
        print("started training")
        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], callbacks=[lgb.early_stopping(3000)])
        print("finished training")
        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
    # ROC_AUC = roc_auc_score(target, oof)
    # # Log model metrics
    # mlflow.log_metric("ROC_AUC_SCORE", ROC_AUC)
    # # Log model parameters
    # mlflow.log_param(**param)
    # Log the trained model as an artifact
    mlflow.lightgbm.log_model(clf, registered_model_name="model")
