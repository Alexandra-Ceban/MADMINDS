import numpy as np
import pandas as pd
import sklearn


df = pd.read_csv("train-input.csv")

print(df.shape)
print(df.columns.tolist())