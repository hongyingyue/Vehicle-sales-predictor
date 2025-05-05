import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(grandparent_dir)

import pandas as pd

from vehicle_ml.feature.ts_feature import add_lagging_feature

data = [
    {
        "Date": 201601 + i,
        "province": "Shanghai",
        "provinceId": 310000,
        "popularity": 1000 + i * 10,
        "model": "modelA",
        "bodyType": "SUV",
        "salesVolume": 200 + i * 5,
    }
    for i in range(5)
] + [
    {
        "Date": 201601 + i,
        "province": "Beijing",
        "provinceId": 110000,
        "popularity": 2000 + i * 15,
        "model": "modelB",
        "bodyType": "Sedan",
        "salesVolume": 300 + i * 7,
    }
    for i in range(5)
]

df = pd.DataFrame(data)
df = df.sort_values(["provinceId", "Date"])

print(df)

df = add_lagging_feature(df, groupby_column=["provinceId", "model"], value_columns=["salesVolume"], lags=[1, 2, 3])

print(df)
