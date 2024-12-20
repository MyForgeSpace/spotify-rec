import pandas as pd

@data_loader
def load_data(*args, **kwargs):
    df = pd.read_csv("https://drive.google.com/uc?export=download&id=1JxKeacy_5Oe4SynDFUJHD0yal7yjeBVr")
    
    df = df.drop(columns='liked') # for test
    
    return df