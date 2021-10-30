import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, args, df, df_demo=None):#, augment=False):
        # self.augment = augment
        self.args = args
        self.df = df
        self.df_demo = df_demo

    def __len__(self):
        return len(self.df)

    # def do_augment(self, x):
    #     x = _rand_add_noise(x)
    #     x = _rand_crop_ecg(x)
    #     return x
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # load ECG
        qid = row['QuantaID']
        doc = row['Date_of_Cath']
        fname = os.path.join(self.args.dir_csv, f'{qid}_{doc}.csv')
        # fname = f'/storage/shared/apollo/same-day/{qid}_{doc}.csv'

        x = pd.read_csv(fname).values[::2,1:].astype(np.float32)
        
        if self.args.label == 'pcwp':
            if self.args.train_mode == 'regression':
                y = row['PCWP_mean']
            else:
                y = row['PCWP_mean'] > self.args.pcwp_th
        elif self.args.label == 'age':
            if self.args.train_mode == 'regression':
                y = row['Age_at_Cath'] #regression
            else:
                y = row['PCWP_mean'] > self.args.pcwp_th

        elif self.args.label == 'gender':
            y = row['Sex']

        x = x / 1000
#             x = (x-x.mean())/(x.std())
        # if self.augment:
        #     x = self.do_augment(x).astype(np.float32)
        sample = (x[:2496,:].T, y)

        return sample