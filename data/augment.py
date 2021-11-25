import pandas as pd

import utils.augmentation as aug

from config import args


def replicates_row(row, num_augments):
    return pd.concat([row] * num_augments, axis=1)


def make_batch(row, df_tab, num_augments):
    pos_sample_idx = [i for i in range(row.name, row.name + num_augments)]
    pos_samples = df_tab.iloc[pos_sample_idx, :]
    pos_samples['group'] = 1
    neg_samples_population = df_tab.drop(pos_sample_idx)
    neg_samples = neg_samples_population.sample(n=args.batch_size - args.num_augments)
    neg_samples['group'] = 0
    batch = pd.concat([pos_samples, neg_samples],  ignore_index=True)
    batch = batch.sample(frac=1)
    return batch


def augment(augment_type, x):
    """
    https://github.com/uchidalab/time_series_augmentation/blob/master/docs/AugmentationMethods.md
    """
    if augment_type == 0:
        x = aug.jitter(x, sigma=0.03)
    elif augment_type == 1:  # shifting
        x = aug.scaling(x, sigma=0.1)
    elif augment_type == 2:  # shifting
        x = aug.rotation(x)  # flipping as well as axis shuffling.
    elif augment_type == 3:  # shifting
        x = aug.time_warp(x, sigma=0.2, knot=4)  # "Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks," in ACM ICMI, pp. 216-220, 2017.
    return x
