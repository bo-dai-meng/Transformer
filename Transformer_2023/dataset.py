import numpy as np
import pandas
import torch
from torch.utils import data
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import os

class Read_Data():
    def __init__(self, filepath="D:\\Al-ion\\Transformer-data\\1-train.xlsx", sheet_name="Sheet1", time_step = 10):
        print(f"reading{filepath},sheet={sheet_name}")
        df = pandas.read_excel(
            filepath,
            names={"cycle", "Charging specific energy", "Charging  energy", "Charge specific capacity", "Disharge specific capacity"},
            dtype ={"cycle": np.float32, "Charging specific energy": np.float32, "Charging  energy": np.float32, "Charge specific capacity": np.float32,
                    "Disharge specific capacity": np.float32}
        ).values

        self.dataset = torch.from_numpy(df)
        self.timestep = time_step

    def get_dataloader(self, batch_sizes):
        sequence_feature_list = []
        for index in range(len(self.dataset) - self.timestep):
            sequence_feature_list.append(self.dataset[index: index + self.timestep])
        sequence_feature = torch.stack(sequence_feature_list)

        sequence_label_list = []
        for index in range(len(self.dataset) - self.timestep):
            sequence_label_list.append(self.dataset[index + self.timestep, 4])
        sequence_label = torch.stack(sequence_label_list)
        dataset = data.TensorDataset(sequence_feature, sequence_label)
        dataloader = DataLoader(dataset, batch_size=batch_sizes, shuffle=False, drop_last=True)
        return dataloader

# if __name__ == "__main__":
#     read_data = Read_Data()
#     train_data_loader = read_data.get_dataloader(batch_sizes=1714)
#     for data in train_data_loader:
#         feature, label = data
#         print(f"feature:{feature}, label:{label}")
