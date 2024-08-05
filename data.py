import torch
from torch.utils.data import TensorDataset, DataLoader


def get_data(batch_size=32, data_size=1000):
    xs = torch.randn(data_size, 128, 16)  # (batch_size, sequence_length, feature_dim)
    ys = torch.randn(data_size, 128, 1)  # (batch_size, sequence_length, target_dim)
    dataset = TensorDataset(xs, ys)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader