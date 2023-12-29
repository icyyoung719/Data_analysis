import numpy as np
import pandas as pd

data = pd.read_csv("data.csv")
genrelist = []

for i in range(len(data)):
    parts = data.iloc[i,2].split('|')
    for part in parts:
        if part not in genrelist:
            genrelist.append(part)
for i in range(len(genrelist)):
    data.insert(len(data.columns), genrelist[i], value=[0]*len(data))

for i in range(len(data)):
    parts = data.iloc[i, 2].split('|')
    for part in parts:
        data.loc[i,part] = 1
data = data.drop('genres',axis = 1)
data.to_csv('data2.csv')