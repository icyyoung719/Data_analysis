import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['font.sans-serif']=['simhei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus']=False

data = pd.read_csv("data.csv")

max_num =data['ratingnum'].max()
plt.figure(figsize=(8, 8))
plt.subplot(211)
print(max_num)
num_arr = [0]*(max_num+1)
for i in range(len(data)):
    num_arr[data.iloc[i, 3]] += 1
plt.plot(np.arange(max_num+1),num_arr)
plt.plot(np.arange(max_num+1-80)+80,num_arr[80:])
plt.title('评分数', fontsize=16, weight='bold', color='black')
plt.show()
