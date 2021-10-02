import numpy as np
import pandas as pd

n_keys = 3
n_cols = 9000
n_rows = 50000
n_csv = 3

df = pd.DataFrame()

for i in range(n_keys):
    key_name = f'key_{i}'
    df[key_name] = [f'key_{i}_{j}' for j in range(n_rows)]

for i in range(n_cols):
    col_name = f'col_{i}'
    df[col_name] = np.random.rand(n_rows) + i + 10

for i in range(n_csv):
    file_name = f'huge_csv_{i}.csv'
    col_name = f'col_{i}_only'
    df[col_name] = np.random.rand(n_rows) + 10000
    df_rand = df.sample(frac=1, axis=1)
    df_rand.to_csv(file_name, index=False)
    df.drop(col_name, axis=1)
