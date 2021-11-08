import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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

def normalize_frame(frame):
    if isinstance(frame, np.ndarray):
        frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-8)
    elif isinstance(frame, torch.Tensor):
        frame = (frame - torch.min(frame)) / (
            torch.max(frame) - torch.min(frame) + 1e-8
        )
    return frame

