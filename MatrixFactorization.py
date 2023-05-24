import torch


class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_items, num_factors=20):
        super().__init__()
        # creating the user embeddings
        self.user_factors = torch.nn.Embedding(num_users, num_factors)  # like a lookup table for the input
        # creating the item embeddings
        self.item_factors = torch.nn.Embedding(num_items, num_factors)  # like a lookup table for the input
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, data):
        # matrix multiplication
        users, items = data[:, 0], data[:, 1]
        return (self.user_factors(users) * self.item_factors(items)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)

