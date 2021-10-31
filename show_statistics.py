import os
import argparse

import pandas as pd

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dir-result', type=str, default='runs')
parser.add_argument('--dataset', type=str, default='pascal')
parser.add_argument('--stat', type=str, default='iou', choices=['iou', 'acc'])
parser.add_argument('--semi-ratio', type=float, default=0.05)
parser.add_argument('--nrow', type=int, default=5)
parser.add_argument('--collect', action='store_true', default=False)
parser.add_argument('--all', action='store_true', default=False)
parser.add_argument('--all-seeds', action='store_true', default=False)
args = parser.parse_args()

dir_results = [args.dir_result]

if args.collect:
    dfs = {}
else:
    df = pd.DataFrame(index=names)


for dir_result in dir_results:
    for name in sorted(os.listdir(dir_result)):
        parsed = name.split('_')
        result_ckpt = os.path.join(args.dir_result, name, 'test_result.pth')

        if not os.path.exists(result_ckpt):
            continue
        if (not args.all) and parsed[0] != '{}{}'.format(args.dataset, args.semi_ratio):
            continue
        if (not args.collect) and (not args.all_seeds) and (not args.all) and parsed[-1][:4] == 'seed':
            continue

        if args.collect:
            model = '_'.join([x for x in parsed[1:] if x[:4] != 'seed'])
            try:
                df = dfs[model]
            except:
                df = pd.DataFrame(index=names)
                dfs[model] = df
        elif args.all_seeds:
            name = '_'.join(parsed[1:])
        elif not args.all:
            name = '_'.join(parsed[1:-1])

        rdict = torch.load(result_ckpt, map_location=device)

        if args.stat == 'iou':
            df.loc[names[0]:names[-1], name] = rdict['IoU']
            df.loc['MEAN', name] = rdict['mIoU']
        else:
            df.loc[names[0]:names[-1], name] = rdict['cAcc']
            df.loc['MEAN', name] = rdict['mAcc']

if args.collect:
    df_collect = pd.DataFrame(index=names+['mean'])
    for model, df in dfs.items():
        if len(df.columns) < 2:
            continue
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        df_collect[model] = ["{:.4f} +- {:.4f}".format(mean[i], std[i]) for i in df.index]
    columns = df_collect.columns
    df = df_collect
else:
    columns = df.columns

while len(columns) > args.nrow:
    print(df[columns[:args.nrow]])
    columns = columns[args.nrow:]
print(df[columns])

torch.save(df, 'test_results.pth')
