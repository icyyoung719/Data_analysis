import numpy as np
import pandas as pd
import sklearn.linear_model as LM
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import classification_report,mean_squared_error
from sklearn import neighbors,preprocessing
from sklearn.datasets import make_regression
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif']=['simhei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus']=False
import warnings
warnings.filterwarnings('ignore')

def KNN_model(i):
    data = pd.read_csv('data2.csv')
    data = data.dropna(subset = ['avg_rating'])
    data = data.loc[(data['ratingnum'] >= i)]
    con_ls = []
    con_ls.append('ratingnum')
    for i in range(-20, 0):
        con_ls.append(data.columns[i])

    X = data[con_ls]
    y = data['avg_rating']
    testErr = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    Ntrain = len(y_train)
    K = np.arange(1, int(Ntrain*0.10), 1)
    for k in K:
        modelKNN = neighbors.KNeighborsRegressor(n_neighbors = k)
        modelKNN.fit(X_train, y_train)
        testErr.append(1 - modelKNN.score(X_test, y_test))
    bestK = K[testErr.index(np.min(testErr))]
    # print('最优K值：', bestK)
    modelKNN = neighbors.KNeighborsRegressor(n_neighbors = bestK)
    modelKNN.fit(X_train, y_train)

    # print('K-近邻：验证集误差 = %f;总预测误差 = %f' % ((1 - modelKNN.score(X_test, y_test)), 1 - modelKNN.score(X, y)))

    y_pred = modelKNN.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# RMSE_ls = []
# for i in range(1,101):
#     answer = 0
#     for ii in range(3):
#         answer += KNN_model(i)
#     RMSE_ls.append(answer/3)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(211)
# plt.plot(np.arange(100),RMSE_ls)
# plt.xlabel('丢弃的M值')
# plt.ylabel('RMSE')
# plt.title('RMSE-M关系图', fontsize=16, weight='bold', color='black')
# plt.show()
answer = 0
for ii in range(3):
    answer += KNN_model(80)
print(answer/3)