import pandas as pd
import numpy as np

df = pd.read_parquet('data/platinum/features/AAPL.parquet')
print(df.columns)