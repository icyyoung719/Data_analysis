# -*- coding:utf-8 -*-
import pandas as pd

df = pd.read_csv("archive/ratings.csv",encoding="utf-8")
#求出用户数和电影数，评分数目, 平均评分, 最大评分, 最小评分
def stat_data(train_data):
    num_user=train_data['userId'].nunique()
    num_movie=train_data['movieId'].nunique()
    num_rating=train_data['rating'].size
    avg_rating=train_data['rating'].mean()
    max_rating=train_data['rating'].max()
    min_rating=train_data['rating'].min()

    return num_user, num_movie, num_rating, avg_rating, max_rating, min_rating

num_user, num_movie, num_rating, avg_rating, max_rating, min_rating =stat_data(df)
print("num_user, num_movie, num_rating, avg_rating, max_rating, min_rating")
print(num_user, num_movie, num_rating, avg_rating, max_rating, min_rating)
