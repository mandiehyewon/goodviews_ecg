import os
import argparse
from tqdm import tqdm
import numpy as np

import torch

from config import args
from data import get_data
from model import get_model
from utils.loss import get_loss
from utils.metrics import Evaluator
from utils.utils import set_devices

'''
CUDA_VISIBLE_DEVICES={} python test.py --name {name} --model {cnn} --
'''
# Get Dataloader, Model
name = args.name
train_loader, val_loader, test_loader = get_data(args)
device = set_devices(args)

model = get_model(args, device=device)
evaluator = Evaluator(args)
criterion = get_loss(args)

# Check if checkpoint exists
if args.last:
    ckpt_path = os.path.join(args.dir_result, name, 'ckpts/last.pth')
else:
    ckpt_path = os.path.join(args.dir_result, name, 'ckpts/best.pth')

if not os.path.exists(ckpt_path):
    print("invalid checkpoint path : {}".format(ckpt_path))

# Load checkpoint, model
ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt['model']
model.load_state_dict(state)
model.eval()
print('loaded model')

evaluator.reset()

with torch.no_grad():
    for i, test_batch in tqdm(enumerate(test_loader), total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
        test_x, test_y = test_batch
        test_x, test_y = test_x.to(device), test_y.to(device)
        logits = model(test_x)

        loss = criterion(logits.float(), test_y.unsqueeze(1).float())
        evaluator.add_batch(test_y.cpu(), logits.cpu(), loss, test=True)

    if args.train_mode == 'binary_class':
        f1, auc, apr, acc = evaluator.performance_metric()
        print ('f1: {}, auc: {}, apr: {}, acc: {}'.format(f1, auc, apr, acc))

    elif args.train_mode == 'regression':
        loss = evaluator.performance_metric()
        print ('loss: {}'.format(loss))