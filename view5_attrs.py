import os
import pandas as pd
from config import args

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df_tab = pd.read_excel(os.path.join(args.dir_csv, "Diagnostics.xlsx"))
exclude_list = [
    "MUSE_20180712_152022_92000",
    "MUSE_20180712_151351_36000",
    "MUSE_20180712_151353_58000",
    "MUSE_20180712_153632_30000",
    "MUSE_20180712_152114_47000",
    "MUSE_20180712_151357_86000",
    "MUSE_20180712_152024_00000",
    "MUSE_20180712_152014_31000",
    "MUSE_20180113_181145_89000",
    "MUSE_20180712_153140_95000",
    "MUSE_20180113_180425_75000",
    "MUSE_20180712_152019_73000",
]
df_tab = df_tab.loc[~df_tab.FileName.isin(exclude_list)]

attrs = [
    "VentricularRate",
    "AtrialRate",
    "QRSDuration",
    "QTInterval",
    "QTCorrected",
    "RAxis",
    "TAxis",
    "QRSCount",
    "QOnset",
    "QOffset",
    "TOffset",
]

df_attrs = df_tab[attrs]

normalized_df_attrs = (df_attrs - df_attrs.mean()) / df_attrs.std()

normalized_df_attrs.to_csv("attrs.tsv", header=False, index=False, sep="\t")

Sum_of_squared_distances = []
K = range(1, 40)
for num_clusters in K:
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(normalized_df_attrs)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.plot(K, Sum_of_squared_distances, "bx-")
plt.xlabel("Values of K")
plt.ylabel("Sum of squared distances/Inertia")
plt.title("Elbow Method For Optimal k")
plt.show()


if __name__ == "__main__":
    pass
