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

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = LR_Scheduler(optimizer, args.scheduler, args.lr, args.epochs, from_iter=args.lr_sch_start, warmup_iters=args.warmup_iters, functional=True)

### TRAINING
pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for epoch in range(1, args.epochs + 1):
    loss = 0
    for train_batch in train_loader:
        train_x, train_y, train_group, train_fnames = train_batch
        train_x, train_group = train_x.to(device), train_group.to(device)
        encoded = model(train_x)

        loss = get_contrastive_loss(args, encoded, train_group, device)
        print(loss)
        logger.loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

    # ## LOGGING
    # if epoch % args.log_iter == 0:
    #     logger.log_tqdm(pbar)
    #     logger.log_scalars(epoch)
    #     logger.loss_reset()

    # ### VALIDATION
    # if epoch % args.val_iter == 0:
    #     model.eval()
    #     logger.evaluator.reset()
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             val_x, val_y, _ = batch
    #             val_x, val_y = val_x.to(device), val_y.to(device)

    #             encoded = model(val_x)
    #             loss = get_contrastive_loss(args, encoded, val_y, device)
    #             logger.evaluator.add_batch(val_y.cpu(), encoded.cpu(), loss)

    #         logger.add_validation_logs(epoch, loss)
    #     model.train()
    # logger.save(model, optimizer, epoch)
    # pbar.update(1)

# ckpt = logger.save(model, optimizer, epoch, last=True)
# logger.writer.close()

# Downstream Training
model.eval()
dw_learning_rate = 1e-2
dw_epochs = -1
dw_criterion = nn.CrossEntropyLoss()
dw_optimizer = torch.optim.SGD(classifier.parameters(), lr=dw_learning_rate)

for epoch in range(1, dw_epochs + 1):
    loss = 0
    classifier.train()
    for train_batch in train_loader:
        train_x, train_y, train_group, train_fnames = train_batch
        train_x = train_x.to(device)

        dw_pred = classifier(model(train_x))
        dw_loss = dw_criterion(dw_pred, train_y.to(torch.long))
        
        print(f"downstream_loss:{dw_loss}")
        
        dw_optimizer.zero_grad()
        dw_loss.backward()
        dw_optimizer.step()

print("\n Finished training..........Starting Test")


with torch.no_grad():
    model.eval()
    classifier.eval()
    y_pred = []
    y_target = []

    for (i,test_batch) in enumerate(test_loader):
        test_x, test_y, test_group, test_fnames = test_batch
        test_x = test_x.to(device)
        test_pred = classifier(model(test_x))
        y_pred.append(test_pred)
        y_target.append(test_y)
        if i > 3:
            break

    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_target = nn.functional.one_hot(torch.cat(y_target,dim=0).to(torch.int64), num_classes=nlabels).numpy()

    test_auc = roc_auc_score(y_true=y_target, y_score=y_pred)
    print(f"Test AUC:{test_auc}")
