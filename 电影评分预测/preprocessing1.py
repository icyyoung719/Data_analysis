# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

def stat_data(train_data):
    num_user=train_data['userId'].nunique()
    num_movie=train_data['movieId'].nunique()
    num_rating=train_data['rating'].size
    avg_rating=train_data['rating'].mean()
    max_rating=train_data['rating'].max()
    min_rating=train_data['rating'].min()

    return num_user, num_movie, num_rating, avg_rating, max_rating, min_rating

df_ratings = pd.read_csv("archive/ratings.csv")
num_user, num_movie, num_rating, avg_rating, max_rating, min_rating =stat_data(df_ratings)
df_movies = pd.read_csv("archive/movies.csv")
data = df_movies#以movies.csv为基础创建数据
del data['title']
data['ratingnum'] = [0]*len(data)
data['avg_rating'] = [0]*len(data)
movieId = 5
data.loc[data['movieId']==movieId,'ratingnum'] += 1
for i in range(num_rating):
    movieId = df_ratings.iloc[i, 1]
    data.loc[data['movieId']==movieId,'ratingnum'] += 1
    data.loc[data['movieId']==movieId,'avg_rating'] += df_ratings.iloc[i, 2]

for i in range(0,len(data)):
    data.iloc[i,3] = data.iloc[i,3]/data.iloc[i,2]
data.to_csv('data.csv')
