import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# label = "condition"
# dir_csv = args.dir_csv

# device = "cuda" if torch.cuda.is_available() else "cpu"

# seed = 42
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

# data = pd.read_excel(os.path.join(args.dir_csv, "Diagnostics.xlsx"))
# data = data.loc[data.PatientAge >= 18].copy()
# data["age_bucket"] = pd.cut(
#     data.PatientAge, bins=[17, 34, 44, 49, 54, 59, 64, 69, 74, 79, 84, 100]
# )
# data["group"] = pd.Categorical(
#     data.age_bucket.astype(str) + data.Gender.astype(str)
# ).codes


class ECGDataset(Dataset):
    def __init__(self, label, dir_csv, df):  # , augment=False):
        self.label = label
        self.dir_csv = dir_csv
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # load ECG
        file = row["FileName"]
        fname = os.path.join(self.dir_csv, f"{file}.csv")

        x = pd.read_csv(fname, header=None).values.astype(np.float32)

        if self.label == "condition":
            y = 0 + 1 * (row["Beat"] != "NONE")

        x = normalize_frame(x)

        group = row["group"]

        return x.T, y, group


# ecg_data = ECGDataset("condition", args.dir_csv, data)
# batch_size = 32

# ecg_loader = DataLoader(ecg_data, batch_size=batch_size)
# for x, y, group in ecg_loader:
#     equality_matrix = 1 * (group[None, :] == group[:, None]) - torch.eye(
#         batch_size, batch_size
#     )
#     print(equality_matrix.sum(0))
#     break

def normalize_frame(frame):
    if isinstance(frame, np.ndarray):
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-8)
    elif isinstance(frame, torch.Tensor):
        frame = (frame - torch.min(frame)) / (
            torch.max(frame) - torch.min(frame) + 1e-8
        )
    return frame

