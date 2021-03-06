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

seed = set_seeds(args)
device = set_devices(args)
logger = Logger(args)

# Load Data, Create Model
train_loader, val_loader, test_loader = get_data(args)
model = get_model(args, device=device)

nlabels = 4
classifier = nn.Linear(args.embed_size, nlabels).to(device)

criterion = nn.CrossEntropyLoss(reduction='none').to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = LR_Scheduler(optimizer, args.scheduler, args.lr, args.epochs, from_iter=args.lr_sch_start, warmup_iters=args.warmup_iters, functional=True)

### TRAINING
pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for epoch in range(1, args.epochs + 1):
    loss = 0
    for (idx, train_batch) in enumerate(train_loader):
        train_x, train_y, train_fnames = train_batch

        train_x= train_x.to(device)
        logit = model(train_x)

        loss = criterion(logit, train_y.long().to(device)).mean()
        
        print(loss)
        logger.loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
    ### VALIDATION
    model.eval()
    logger.evaluator.reset()
    with torch.no_grad():
        for batch in val_loader:
            val_x, val_y, _= batch
            val_x, val_y = val_x.to(device), val_y.long().to(device)

            logits = model(val_x)

            loss = criterion(logits, val_y.long()).mean()
            print (loss)
    model.train()

if args.epochs > 0:
    ckpt = logger.save(model, optimizer, epoch, last=True)
    logger.writer.close()

print("\n Finished training..........Starting Test")

with torch.no_grad():
    model.eval()
    classifier.eval()
    y_pred = []
    y_target = []

    for (i,test_batch) in enumerate(test_loader):
        test_x, test_y, test_fnames = test_batch
        
        test_x = test_x.to(device)
        test_pred = classifier(model(test_x))
        y_pred.append(test_pred.cpu())
        y_target.append(train_y.unsqueeze(1).long())

    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_target = nn.functional.one_hot(torch.cat(y_target,dim=0).to(torch.int64), num_classes=nlabels).numpy()

    test_auc = roc_auc_score(y_true=y_target, y_score=y_pred)
    print(f"Test AUC:{test_auc}")
