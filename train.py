import os
import pickle
import click
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import mlflow

def load_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed Spotify data was saved"
)
def run_train(data_path: str):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))
    
    with mlflow.start_run():
        mlflow.set_tag("developer", "Amir")
        mlflow.xgboost.autolog()
        mlflow.log_param("train-data-path", os.path.join(data_path, "train.pkl"))
        mlflow.log_param("test-data-path", os.path.join(data_path, "test.pkl"))
    
        model_xgb = XGBClassifier(
            max_depth=2,
            n_estimators=210,
            random_state=42,
            reg_alpha=0.7729525381085978,  
            reg_lambda=0.08462470748949007, 
            subsample=0.8039386833104527,  
            colsample_bytree=0.834587666802032,  
            learning_rate = 0.06434698663069276
        )
        model_xgb.fit(X_train, y_train)
        
        threshold = 0.3
        mlflow.log_param("threshold", threshold)
        
        y_pred_train_proba = model_xgb.predict_proba(X_train)
        y_pred_train = (y_pred_train_proba[:, 1] >= threshold).astype(int)
        acc_score_train = accuracy_score(y_train, y_pred_train)
        mlflow.log_metric("accuracy_score_train", acc_score_train)

        
        y_pred_proba = model_xgb.predict_proba(X_test)
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
        
        acc_score_test = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy_score_test", acc_score_test)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000/") # or sqlite:///mlflow.db
    mlflow.set_experiment("spotify-tracks-rec")
    run_train()
