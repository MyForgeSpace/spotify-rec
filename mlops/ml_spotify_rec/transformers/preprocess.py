import requests
import pickle
import pandas as pd
import numpy as np

def from_pkl(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        obj = pickle.loads(response.content)
        print("Объект успешно загружен из удалённого файла!")
        return obj
    else:
        print(f"Ошибка при скачивании: {response.status_code}")

@transformer
def transform(df, *args, **kwargs):
    # urls
    dv_url = "https://drive.google.com/uc?export=download&id=1kHs9nF9drKKGBRm7Wh7K8RAW9q7u0oD9"
    std_scaller_url = "https://drive.google.com/uc?export=download&id=1VU_xhtpgLEST1zjpXg0hKqtuNvzL_spk"
    
    # Download dv and std_scaller
    dv = from_pkl(dv_url)
    std_scaller = from_pkl(std_scaller_url)
    
    # StandartScaller()
    X_std = std_scaller.transform(df)
    X_std_df = pd.DataFrame(X_std, columns=df.columns)

    # DictVectorizer
    dicts = X_std_df.to_dict(orient='records')
    X = dv.transform(dicts)

    return X