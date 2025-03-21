import opendatasets as od
import pandas as pd

# Download dataset without local storage
od.download('https://www.kaggle.com/datasets/andreafrancia/coughvid-v2')

# Load metadata
metadata = pd.read_csv('coughvid-v2/metadata_compiled.csv')
print(f"Loaded {len(metadata)} samples")