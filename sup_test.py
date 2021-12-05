import os
import sys
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from config import args
from data import get_data
from model import get_model
from utils.metrics import Evaluator
from utils.logger import Logger
from utils.utils import set_seeds, set_devices
from utils.loss import get_contrastive_loss
from utils.lr_scheduler import LR_Scheduler
from sklearn.metrics import roc_auc_score

# Get Dataloader, Model
name = args.name
device = set_devices(args)

# Load Data, Create Model
train_loader, val_loader, test_loader = get_data(args)
model = get_model(args, device=device)

nlabels = 4
classifier = nn.Linear(args.embed_size, nlabels).to(device)

# Check if checkpoint exists
ckpt_path = os.path.join(args.dir_result, name, 'ckpts/model.pth'.format(str(args.load_step)))

if not os.path.exists(ckpt_path):
    print("invalid checkpoint path : {}".format(ckpt_path))

# Load checkpoint, model
ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt['model']
model.load_state_dict(state)
model.eval()
print('loaded model')

with torch.no_grad():
    model.eval()
    classifier.eval()
    y_pred = []
    y_target = []

    for (i,test_batch) in enumerate(test_loader):
        if args.viewtype in ['clocstime', 'clocslead']:
            test_x1, test_x2, test_y, test_group, test_fnames = test_batch

            test_x = torch.cat((test_x1, test_x2),dim=0)
            test_y = torch.cat((test_y, test_y),dim=0)
            test_group = torch.cat((test_group, test_group),dim=0)

        else:
            test_x, test_y, test_fnames = test_batch
        
        test_x = test_x.to(device)
        test_pred = classifier(model(test_x))
        y_pred.append(test_pred.cpu())
        y_target.append(test_y)

    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_target = nn.functional.one_hot(torch.cat(y_target,dim=0).to(torch.int64), num_classes=nlabels).numpy()

    test_auc = roc_auc_score(y_true=y_target, y_score=y_pred)
    print(f"Test AUC:{test_auc}")
