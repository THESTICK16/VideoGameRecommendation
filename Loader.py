# The DataLoader that will be used by pytorch

import torch
from torch.utils.data.dataset import Dataset


class Loader(Dataset):

    def __init__(self, data_frame, user_reference: str, rated_material_reference: str, rating_reference: str):
        """
        :param data_frame: the data frame being used for the ML process
        :param user_reference: the key for the 'user id' that is used by the dataset
        :param rated_material_reference: the key for the target media that is being recommended (i.e. book name, player name, game name, etc.)
        :param rating_reference: the key for the rating for the given material from the dataset
        """
        self.ratings = data_frame.copy()

        # Extract all user IDs and rated material IDs
        users = data_frame[user_reference].unique()  # FIXME make sure referencing the data frame keys like this still works
        rated_materials = data_frame[rated_material_reference].unique()

        # .......Producing new continuous IDs for materials and users........

        # Unique values : index
        self.userid2idx = {o:i for i,o in enumerate(users)}
        self.materialid2idx = {o:i for i,o in enumerate(rated_materials)}

        # Create continuous IDs for users and rated materials
        self.idx2userid = {i:o for o,i in self.userid2idx.items()}
        self.idx2materialid = {i:o for o,i in self.materialid2idx.items()}

        # return the id from the indexed values in the lambda function below
        self.ratings[rated_material_reference] = data_frame[user_reference].apply(lambda x: self.materialid2idx[x])
        self.ratings[user_reference] = data_frame[user_reference].apply(lambda x: self.userid2idx[x])

        self.x = self.ratings.drop([rating_reference], axis=1,).values
        self.y = self.ratings[rating_reference].values
        self.x, self.y = torch.tensor(self.x), torch.tensor(self.y)  # Transforms the data to tensors (ready for torch models)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.ratings)

