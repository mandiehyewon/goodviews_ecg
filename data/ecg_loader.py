import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans

from .augment import augment



class ECGDataset(Dataset):
    def __init__(self, args, df):  # , augment=False):
        self.args = args
        self.label = self.args.label
        self.dir_csv = self.args.dir_csv
        self.df = df
        self.viewtype = self.args.viewtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # load ECG
        file = row["FileName"]
        fname = os.path.join(self.dir_csv, "ECGDataDenoised", f"{file}.csv")

        x = pd.read_csv(fname, header=None).values.astype(np.float32)

        y = row["y"]
        x = normalize_frame(x)

        if self.viewtype == 'demos':
            group = row["group"]
        elif self.viewtype == 'rhythm':
            group = row["y"]
        elif self.viewtype == 'simclr':
            group = random.randint(1,3)
            print(type(x))
            x = augment(self.args, group, x)
        elif self.viewtype == 'attr':
            group = row["group"]

        return x.T, y, group, fname

def normalize_frame(frame):
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy((frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-8))
    elif isinstance(frame, torch.Tensor):
        frame = (frame - torch.min(frame)) / (
            torch.max(frame) - torch.min(frame) + 1e-8
        )
    return frame
