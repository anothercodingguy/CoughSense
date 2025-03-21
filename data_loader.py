import opendatasets as od
import pandas as pd

def load_data():
    # Requires kaggle.json in root folder
    od.download('https://www.kaggle.com/datasets/andreafrancia/coughvid-v2')
    return pd.read_csv('coughvid-v2/metadata_compiled.csv')