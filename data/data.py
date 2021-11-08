import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config import args
from .ecg_loader import ECGDataset


def get_data(args):
    df_tab = pd.read_excel(os.path.join(args.dir_csv, "Diagnostics.xlsx"))
    
    df_tab = df_tab.loc[df_tab.PatientAge >= 18].copy()
    df_tab['age_bucket'] = pd.cut(df_tab.PatientAge, bins=[17,34,44,49,54,59,64,69,74,79,84,100])
    df_tab["group"] = pd.Categorical(df_tab.age_bucket.astype(str) + df_tab.Gender.astype(str)).codes

    train_ids = np.load("./stores/train_ids.npy", allow_pickle=True)
    val_ids = np.load("./stores/val_ids.npy", allow_pickle=True)
    test_ids = np.load("./stores/test_ids.npy", allow_pickle=True)

    train_df = df_tab[df_tab["FileName"].isin(train_ids)]
    val_df = df_tab[df_tab["FileName"].isin(val_ids)]
    test_df = df_tab[df_tab["FileName"].isin(test_ids)]
    print(len(train_df), len(val_df), len(test_df))

    train_loader = DataLoader(
        ECGDataset(args, train_df),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        ECGDataset(args, val_df),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        ECGDataset(args, test_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, val_loader, test_loader


def save_trainid(args):
    print(args.dir_csv)
    df_tab = pd.read_excel(os.path.join(args.dir_csv, "Diagnostics.xlsx"))

    frac_train = 0.6  # t:t:v = 6:2:2
    frac_val = 0.2

    rng = np.random.RandomState(seed=args.seed)

    uids = df_tab["FileName"].unique()
    rng.shuffle(uids)

    train_ids = uids[: int((frac_train) * len(uids))]
    val_ids = uids[
        int((frac_train) * len(uids)) : int((frac_train + frac_val) * len(uids))
    ]
    test_ids = uids[int((frac_train + frac_val) * len(uids)) :]

    np.save("./stores/train_ids.npy", train_ids)
    np.save("./stores/val_ids.npy", val_ids)
    np.save("./stores/test_ids.npy", test_ids)

    return
