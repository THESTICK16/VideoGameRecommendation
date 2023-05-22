# Data Citation
# Tamber. (2017). Steam Video Games. Retrieved May 22, 2023,.

import pandas
from torch import save, load
# from MatrixFactorization import *
# from Loader import *
from sklearn.cluster import KMeans

"""
# Drop the irrelevant data, such as purchase type and the extraneous column to clean the dataset
games_df = pandas.read_csv('steam-200k.csv')
print(games_df.info())
dropped_purchases_games_df = games_df[games_df.behavior_name != 'purchase']
print(dropped_purchases_games_df.head())
dropped_purchases_games_df = dropped_purchases_games_df.drop(['behavior_name', 'number_zero'], axis=1)
print(dropped_purchases_games_df.head())
print(dropped_purchases_games_df.info())
playtime_only_games_csv = dropped_purchases_games_df.to_csv('playtime_only_games.csv', index=False)
"""

games_df = pandas.read_csv('playtime_only_games.csv')
print(games_df.head())

