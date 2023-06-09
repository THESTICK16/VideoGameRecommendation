# Data Citation
# Tamber. (2017). Steam Video Games. Retrieved May 22, 2023,.

import numpy
import pandas
from torch import save, load
from torch.utils.data import DataLoader

from MatrixFactorization import *
from Loader import *
from sklearn.cluster import KMeans

# This will cause the model to train if true. If training data has been succesfully saved, set to false and it will not train again
conduct_model_training = False

"""
# This code block cleans the dataset by dropping the irrelevant data, such as purchase type and the extraneous column.
# It then creates a new CSV file with the clean data
# This block needs only be run once

def clean_dataset():
    games_df = pandas.read_csv('steam-200k.csv')
    print(games_df.info())
    dropped_purchases_games_df = games_df[games_df.behavior_name != 'purchase']
    print(dropped_purchases_games_df.head())
    dropped_purchases_games_df = dropped_purchases_games_df.drop(['behavior_name', 'number_zero'], axis=1)
    print(dropped_purchases_games_df.head())
    print(dropped_purchases_games_df.info())
    playtime_only_games_csv = dropped_purchases_games_df.to_csv('playtime_only_games.csv', index=False)
    
    
clean_dataset()
"""

games_df = pandas.read_csv('playtime_only_games.csv')
# print(games_df.head())

num_users = len(games_df.userID.unique())
num_items = len(games_df.game_name.unique())
print('Number of unique users:', num_users)
print('Number of unique games:', num_items)
print('The full rating matrix will have', num_users * num_items, 'elements.')
print('.................................')
print('Number of playtimes:', len(games_df))
print('Therefore,', len(games_df) / (num_users * num_items) * 100, '% of the matrix is filled.')

num_epochs = 128  # epochs are the number of times that the training will pass over the entire data set.
batch_size = 128 # batches are a subset of data objects in the entire data pool. The weights of the NN are updated after passing through each batch

cuda = torch.cuda.is_available()
print('Is running on GPU:', cuda)

model = MatrixFactorization(num_users, num_items, num_factors=8)

# Prints the current wights in the Pytorch Tensor objects
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
# print('\n')

# gpu enable if you have cuda
model = model.to("cuda") if cuda else model.to("cpu")

# MSE Loss: Mean Squared Error loss, i.e. the difference between the expected error and the actual error
loss_fn = torch.nn.MSELoss()

# ADAM optimizer: This is the equation that updates the weights during NN training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_set = Loader(games_df, 'userID', 'game_name', 'hours_played')
train_loader = DataLoader(train_set, batch_size, shuffle=True)


def train_model():
    # Train data


    for it in range(num_epochs):
        losses = []
        for x, y in train_loader:
            if cuda:
                x, y = x.cuda(), y.cuda()
            else:
                x, y = x.to('cpu'), y.to('cpu')
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
            # loss = loss_fn(outputs, y.type(torch.float32))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print("iter# {}".format(it), "Loss:", sum(losses) / len(losses))


if conduct_model_training:
    train_model()

    with open('model_state.pt', 'wb') as f:
        save(model.state_dict(), f)
else:
    with open('model_state.pt', 'rb') as f:
        model.load_state_dict(load(f))

# Compare the new weights after training with the old
# print("New Weights:")
# c = 0
# uw = 0
# iw = 0
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
#         if c == 0:
#             uw = param.data
#             c += 1
#         else:
#             iw = param.data
#         print('param data:', param.data)

trained_game_embeddings = model.item_factors.weight.data.cpu().numpy()

# Fit the clusters based on game weights
kmeans = KMeans(n_init=10, n_clusters=10, random_state=0).fit(trained_game_embeddings)


def print_recommendations(cluster_size: int):
    for cluster in range(cluster_size):
        print("Cluster = #{}".format(cluster))
        games = []
        for gameidx in numpy.where(kmeans.labels_ == cluster)[0]:
            gameid = train_set.idx2materialid[gameidx]
            rat_count = games_df.loc[games_df['game_name'] == gameid].count()[0]
            games.append(gameid + str(rat_count))
        for game in sorted(games, key=lambda tup: tup[1], reverse=True)[:10]:
            print("\t", game)


print("\nThe following clusters represent collections of games that users may like.\n"
      "If a user likes one game in a cluster, they will very likely enjoy other games in that cluster.\n"
      "Due to using playtime as the primary comparison metric, this program has a tendency to prefer games with "
      "longer completion times in its output.\n")
print_recommendations(10)
