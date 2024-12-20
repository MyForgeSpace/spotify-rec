import os
import pickle
import click
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# csv -> pkl
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

# read and return df    
def read_table(filename: str):
    df = pd.read_csv(filename)
    return df

# preprocessing
def preprocess(X_old: pd.DataFrame, dv: DictVectorizer, fit_df: bool, std_scaler: StandardScaler, fit_std_scaler: bool):
    if fit_std_scaler:
        X_std = std_scaler.fit_transform(X_old)
    else:
        X_std = std_scaler.transform(X_old)
    
    X_std_df = pd.DataFrame(X_std, columns=X_old.columns)
    dicts = X_std_df.to_dict(orient='records')
    
    if fit_df:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv, std_scaler

# run main function
@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw Spotify data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str):
    df = read_table(
        os.path.join(raw_data_path, "data.csv")
    )
    X = df.drop(columns='liked')
    y = df['liked']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    dv = DictVectorizer()
    std_scaler = StandardScaler()
    
    X_train, dv, std_scaler = preprocess(X_train, dv, fit_df=True, std_scaler=std_scaler, fit_std_scaler=True)
    X_test, _, _ = preprocess(X_test, dv, fit_df=False, std_scaler=std_scaler, fit_std_scaler=False)
    
    os.makedirs(dest_path, exist_ok=True)
    
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle(std_scaler, os.path.join(dest_path, "std_scaler.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

if __name__ == '__main__':
    run_data_prep()
