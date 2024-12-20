import os
import pickle
import click
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.xgboost

def load_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed Spotify data was saved"
)
def run_hyperopt(data_path: str):
    # Load data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    def objective(params):
        """Objective function for Hyperopt"""
        with mlflow.start_run(nested=True):
            mlflow.set_tag("developer", "Amir")

            # Log hyperparameters
            mlflow.log_params(params)

            # Train model
            model = XGBClassifier(
                max_depth=int(params['max_depth']),
                n_estimators=int(params['n_estimators']),
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                random_state=42
            )
            model.fit(X_train, y_train)

            # Evaluate model
            threshold = 0.3
            y_pred_proba = model.predict_proba(X_test)
            y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("accuracy_score_test", accuracy)

            return {'loss': -accuracy, 'status': STATUS_OK}

    # Define search space
    space = {
        'max_depth': hp.quniform('max_depth', 2, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': hp.loguniform('reg_alpha', -3, 2),
        'reg_lambda': hp.loguniform('reg_lambda', -3, 2)
    }

    # Initialize Hyperopt Trials object
    trials = Trials()

    # Run Hyperopt optimization
    with mlflow.start_run():
        mlflow.set_experiment("spotify-tracks-rec-opt")
        mlflow.log_param("train-data-path", os.path.join(data_path, "train.pkl"))
        mlflow.log_param("test-data-path", os.path.join(data_path, "test.pkl"))

        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials
        )

        # Log best hyperparameters
        mlflow.log_params(best_params)

        # Train the best model
        best_model = XGBClassifier(
            max_depth=int(best_params['max_depth']),
            n_estimators=int(best_params['n_estimators']),
            learning_rate=best_params['learning_rate'],
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            reg_alpha=best_params['reg_alpha'],
            reg_lambda=best_params['reg_lambda'],
            random_state=42
        )
        best_model.fit(X_train, y_train)
        

        # Evaluate the best model
        threshold = 0.3
        mlflow.log_param("threshold", threshold)
        y_pred_proba = best_model.predict_proba(X_test)
        y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)

        # Log final metrics
        mlflow.log_metric("final_accuracy_score_test", accuracy)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000/")
    run_hyperopt()
