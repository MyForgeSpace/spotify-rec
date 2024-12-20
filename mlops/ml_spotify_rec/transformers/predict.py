from xgboost import XGBClassifier
import requests
import pickle
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime

def from_pkl(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        obj = pickle.loads(response.content)
        print("Объект успешно загружен из удалённого файла!")
        return obj
    else:
        print(f"Ошибка при скачивании: {response.status_code}")

@transformer
def transform(X, *args, **kwargs):
    current_time = datetime.now().strftime("%H:%M:%S")
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("spotify-tracks-rec-inlive")
    # best model url
    model_xgb_url = "https://drive.google.com/uc?export=download&id=1GTuephEZVzwWC3STGrGbVTw64MgM8-19"
    # Download
    model_xgb = from_pkl(model_xgb_url)

    threshold = 0.3
    with mlflow.start_run():
        mlflow.set_tag("developer", "Amir")
        mlflow.log_param("logged_time", current_time)
        mlflow.log_param("threshold", threshold)
        mlflow.xgboost.autolog()
        y_pred_proba = model_xgb.predict_proba(X)
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
    print(y_pred)
    return y_pred
