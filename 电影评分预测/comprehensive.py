# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn.linear_model as LM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif']=['simhei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['axes.unicode_minus']=False
import warnings
warnings.filterwarnings('ignore')

def comprehensive_model(i):
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
    y_pred_LR = modelLR.predict(X_test)

    testErr = []
    Ntrain = len(y_train)
    K = np.arange(1, int(Ntrain * 0.10), 1)
    for k in K:
        modelKNN = neighbors.KNeighborsRegressor(n_neighbors = k)
        modelKNN.fit(X_train, y_train)
        testErr.append(1 - modelKNN.score(X_test, y_test))
    bestK = K[testErr.index(np.min(testErr))]
    modelKNN = neighbors.KNeighborsRegressor(n_neighbors = bestK)
    modelKNN.fit(X_train, y_train)
    y_pred_KNN = modelKNN.predict(X_test)

    y_pred_comprehensive = (y_pred_LR+y_pred_KNN)/2
    mse = mean_squared_error(y_test, y_pred_comprehensive)
    rmse = np.sqrt(mse)
    return rmse

answer = 0
for ii in range(3):
    answer += comprehensive_model(80)
print(answer/3)