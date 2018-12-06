import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%matplotlib inline

movies_df = pd.read_csv('ml-1m/movies.dat',sep='::', header = None)
ratings_df = pd.read_csv('ml-1m/ratings.dat',sep='::', header = None)
movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'TimeStamp']
movies_df['List Index'] = movies_df.index
merged_df = movies_df.merge(ratings_df, on='MovieID')#merge movie dataframe with ratings
merged_df = merged_df.drop('TimeStamp', axis = 1).drop('Title', axis = 1).drop('Genres', axis = 1)
user_Group = merged_df.groupby('UserID')
movies_df.to_pickle("./movies.pkl")
ratings_df.to_pickle("./rating.pkl")
merged_df.to_pickle("./merged.pkl")
#amount of users used for training
