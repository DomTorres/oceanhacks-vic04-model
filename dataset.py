import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class CurrentsDataset(Dataset):
    def __init__(self):
        self.base_dir = "./processed-data/"

        self.currents = np.load(self.base_dir + "currents.npz", allow_pickle=True)['currents']
        self.river_heights = np.load(self.base_dir + "river_heights.npz", allow_pickle=True)['river_heights']


    def __len__(self):
        return len(self.currents)


    def __getitem__(self, idx):
        return self.currents[idx], self.river_heights[idx]


dataset = CurrentsDataset()

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size

train_dataset = Subset(dataset, range(0, train_size))
test_dataset = Subset(dataset, range(train_size, len(dataset)))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

print(train_size, test_size)
