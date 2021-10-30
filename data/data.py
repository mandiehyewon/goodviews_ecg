import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config import args
from .apollo import ECGDataset


def get_data(args):
    df_tab = pd.read_csv(os.path.join(args.dir_csv, "tabular_data.csv"))
    train_ids = np.load("./stores/train_ids.npy")
    val_ids = np.load("./stores/val_ids.npy")
    test_ids = np.load("./stores/test_ids.npy")

    train_df = df_tab[df_tab["QuantaID"].isin(train_ids)]
    val_df = df_tab[df_tab["QuantaID"].isin(val_ids)]
    test_df = df_tab[df_tab["QuantaID"].isin(test_ids)]
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
        ECGDataset(args, train_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    return train_loader, val_loader, test_loader


def save_trainid(args):
    df_tab = pd.read_csv(os.path.join(args.dir_csv, "tabular_data.csv"))
    frac_train = 0.6  # t:t:v = 6:2:2
    frac_val = 0.2

    rng = np.random.RandomState(seed=args.seed)

    uids = df_tab["QuantaID"].unique()
    rng.shuffle(uids)

    # DO 5 FOLD HERE - save train ids for future use
    # fold = args.seed
    # test_start_idx = int((frac_train+frac_val)*len(uids)) #int(fold*0.2*len(uids))
    # test_end_idx = int((fold+1)*0.2*len(uids))

    train_ids = uids[: int((frac_train) * len(uids))]
    val_ids = uids[
        int((frac_train) * len(uids)) : int((frac_train + frac_val) * len(uids))
    ]
    test_ids = uids[int((frac_train + frac_val) * len(uids)) :]

    # TRAINVAL_END = int(args.train_samp*len(train_val_ids)) #args.train_samp: when we use partial dataset
    # train_val_ids = train_val_ids[:TRAINVAL_END]

    # train_ids = train_val_ids[:int(frac_train*len(train_val_ids))]
    # val_ids = train_val_ids[int(frac_train*len(train_val_ids)):]
    # test_ids = uids[int(frac_val_end*len(uids)):]

    np.save("./stores/train_ids.npy", train_ids)
    np.save("./stores/val_ids.npy", val_ids)
    np.save("./stores/test_ids.npy", test_ids)

    return


def normalize(args):
    df_tab = pd.read_csv(os.path.join(args.dir_csv, "tabular_data.csv"))

    return
