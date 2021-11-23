import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config import args
from .ecg_loader import ECGDataset, normalize_frame


def get_data(args):
    df_tab = pd.read_excel(os.path.join(args.dir_csv, "Diagnostics.xlsx"))
    exclude_list = ['MUSE_20180712_152022_92000', 'MUSE_20180712_151351_36000', 'MUSE_20180712_151353_58000', 'MUSE_20180712_153632_30000', 
    'MUSE_20180712_152114_47000', 'MUSE_20180712_151357_86000', 'MUSE_20180712_152024_00000', 'MUSE_20180712_152014_31000', 
    'MUSE_20180113_181145_89000', 'MUSE_20180712_153140_95000', 'MUSE_20180113_180425_75000', 'MUSE_20180712_152019_73000']
    df_tab = df_tab.loc[~df_tab.FileName.isin(exclude_list)]

    df_tab = df_tab.loc[df_tab.PatientAge >= 18].copy()
    df_tab['age_bucket'] = pd.cut(df_tab.PatientAge, bins=[17,34,44,49,54,59,64,69,74,79,84,100])

    df_tab.loc[df_tab['Rhythm'].isin(["AF", "AFIB"]), 'y'] = 1
    df_tab.loc[df_tab['Rhythm'].isin(["SVT", "AT", "SAAWR", "ST", "AVNRT", "AVRT"]), 'y'] = 2
    df_tab.loc[df_tab['Rhythm'].isin(["SB"]), 'y'] = 3
    df_tab.loc[df_tab['Rhythm'].isin(["SR", "SI", "SA"]), 'y'] = 4
    print(df_tab.groupby('y').agg('count'))
    df_tab['y'] = df_tab.copy()['y'] - 1

    if args.viewtype=="demos":
        df_tab["group"] = pd.Categorical(df_tab.age_bucket.astype(str) + df_tab.Gender.astype(str)).codes
    elif args.viewtype=="rhythm":
        df_tab["group"] = df_tab.copy().y

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
    fname = os.path.join(args.dir_csv, "ECGDataDenoised", f"{MUSE_20180112_073319_29000}.csv")

def screen_out(args):
    rm_pts = []
    df_tab = pd.read_excel(os.path.join(args.dir_csv, "Diagnostics.xlsx"))
    for file in df_tab["FileName"]:
        fname = os.path.join(args.dir_csv, "ECGDataDenoised", f"{file}.csv")
        x = pd.read_csv(fname, header=None).values.astype(np.float32)
        x = normalize_frame(x).T
        
        if x.shape == (12, 5000):
            pass
        else:
            print('-------Different Shape------'+file)
            rm_pts.append(file)

    np.save('./stores/rm_pts.npy', np.array(rm_pts))

    return