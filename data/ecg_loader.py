import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, args, df):  # , augment=False):
        self.args = args
        self.dir_csv = self.args.dir_csv
        self.df = df
        self.viewtype = self.args.viewtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.viewtype != 'simclr': # simclr's row already includes x values in augmented format.
            # load ECG
            file = row["FileName"]
            fname = os.path.join(self.dir_csv, "ECGDataDenoised", f"{file}.csv")
            x = pd.read_csv(fname, header=None).values.astype(np.float32)
            x = normalize_frame(x)

        y = row["y"]
        
        if self.viewtype == 'sup':
#             label = np.zeros(4)
#             label[int(y)] = 1
#             y = label
            
            return x.T, y, fname

        if self.viewtype in ['demos', 'attr']:
            group = row["group"]
        elif self.viewtype == 'rhythm':
            group = row["y"]
        elif self.viewtype == 'simclr':
            file = row["FileName"]
            fname = os.path.join(self.dir_csv, "ECGDataDenoised", f"{file}.csv")

            x = row["x"]
            x = x.astype(np.float32)
            x = normalize_frame(x)
            group = row["group"]  # simclr already given 1 and 0 as a group. 1 is for pos samples, 0 for neg samples.

        return x.T, y, group, fname


def normalize_frame(frame):
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy((frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-8))
    elif isinstance(frame, torch.Tensor):
        frame = (frame - torch.min(frame)) / (
            torch.max(frame) - torch.min(frame) + 1e-8
        )
    return frame


class CLOCSDataset(Dataset):
    def __init__(self, args, df):  # , augment=False):
        self.args = args
        self.dir_csv = self.args.dir_csv
        self.df = df
        self.viewtype = self.args.viewtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = row["y"]
        group = idx

        file = row["FileName"]
        fname = os.path.join(self.dir_csv, "ECGDataDenoised", f"{file}.csv")
        x = pd.read_csv(fname, header=None).values.astype(np.float32)
        x = normalize_frame(x)

        if self.viewtype == 'clocstime':
            half = int(x.size(0)/2)
            x1 = x[:half]
            x2 = x[half:]

        elif self.viewtype == 'clocslead':
            half = int(x.size(1)/2)
            x1 = x[:,:half]
            x2 = x[:,half:]

        return x1.T, x2.T, y, group, fname







