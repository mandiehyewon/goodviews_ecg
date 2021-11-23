import os
import pandas as pd
from config import args


df_tab = pd.read_excel(os.path.join(args.dir_csv, "Diagnostics.xlsx"))
exclude_list = ['MUSE_20180712_152022_92000', 'MUSE_20180712_151351_36000', 'MUSE_20180712_151353_58000',
                'MUSE_20180712_153632_30000',
                'MUSE_20180712_152114_47000', 'MUSE_20180712_151357_86000', 'MUSE_20180712_152024_00000',
                'MUSE_20180712_152014_31000',
                'MUSE_20180113_181145_89000', 'MUSE_20180712_153140_95000', 'MUSE_20180113_180425_75000',
                'MUSE_20180712_152019_73000']
df_tab = df_tab.loc[~df_tab.FileName.isin(exclude_list)]



if __name__ == "__main__":
    pass
