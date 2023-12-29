import numpy as np
import pandas as pd
import sklearn.linear_model as LM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif']=['simhei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus']=False
import warnings
warnings.filterwarnings('ignore')



def my_linermodel(i):
    data = pd.read_csv('data2.csv')
    data = data.dropna(subset = ['avg_rating'])
    data = data.loc[(data['ratingnum'] >= i)]
    con_ls = []
    con_ls.append('ratingnum')
    for i in range(-20, 0):
        con_ls.append(data.columns[i])

    X = data[con_ls]
    y = data['avg_rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    modelLR = LM.LinearRegression()
    modelLR.fit(X_train, y_train)
    y_pred = modelLR.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# RMSE_ls = []
# for i in range(1,201):
#     answer = 0
#     for ii in range(3):
#         answer += my_linermodel(i)
#     RMSE_ls.append(answer/3)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(211)
# plt.plot(np.arange(200),RMSE_ls)
# plt.xlabel('丢弃的M值')
# plt.ylabel('RMSE')
# plt.title('RMSE-M关系图', fontsize=16, weight='bold', color='black')
# plt.show()

answer = 0
for ii in range(3):
    answer += my_linermodel(80)
print(answer/3)