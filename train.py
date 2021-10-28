import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from config import args
from data import get_data
from model import get_model
from utils.loss import get_loss
from utils.metrics import Evaluator
from utils.logger import Logger
from utils.utils import set_seeds, set_devices

seed = set_seeds(args)
device = set_devices(args)
logger = Logger(args)

# Load Data, Create Model
train_loader, val_loader, test_loader = get_data(args)
model = get_model(args, device=device)

criterion = get_loss(args)
optimizer = optim.Adam(model.parameters(), lr = args.lr)


### TRAINING
pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
for epoch in range(1, args.epochs + 1):
    loss = 0
    for train_batch in train_loader:
        train_x, train_y = train_batch
        train_x, train_y = train_x.to(device), train_y.to(device)

        if args.model == 'cnn':
            logits = model(train_x.unsqueeze(1))
        else:
            logits = model(train_x)

        loss = criterion(logits.float(), train_y.unsqueeze(1).float())
        logger.loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ## LOGGING
    if epoch % args.log_iter == 0:
        logger.log_tqdm(pbar)
        logger.log_scalars(epoch)
        logger.loss_reset()

    ### VALIDATION
    if epoch % args.val_iter == 0:
        model.eval()
        logger.evaluator.reset()
        with torch.no_grad():
            for batch in val_loader:
                val_x, val_y = batch
                val_x, val_y = val_x.to(device), val_y.to(device)
        
                if args.model == 'cnn':
                    logits = model(val_x.unsqueeze(1))
                else:
                    logits = model(val_x)
                
                loss = criterion(logits.float(), val_y.unsqueeze(1).float())
                logger.evaluator.add_batch(val_y.cpu(), logits.cpu(), loss)
            logger.add_validation_logs(epoch, loss)
        model.train()
    logger.save(model, optimizer, epoch)
    pbar.update(1)

ckpt = logger.save(model, optimizer, epoch, last=True)
logger.writer.close()

## Testing
print('\n Finished training.......... Start Testing..........')

# Load checkpoint
state = ckpt['model']
model = get_model(args).to(device)

model.load_state_dict(state)
model.eval()

print('loaded checkpoint')
test_evaluator = Evaluator(args)
logger.evaluator.reset()

with torch.no_grad():
    for i, test_batch in tqdm(enumerate(test_loader), total=len(test_loader), bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
        test_x, test_y = test_batch
        test_x, test_y = test_x.to(device), test_y.to(device)
        logits = model(test_x.unsqueeze(1))
        
        loss = criterion(logits.float(), test_y.unsqueeze(1).float())
        logger.evaluator.add_batch(test_y.cpu(), logits.cpu(), loss, test=True)

if args.train_mode == 'binary_class':
    f1, auc, apr, acc = logger.evaluator.performance_metric()
    print ('f1: {}, auc: {}, apr: {}, acc: {}'.format(f1, auc, apr, acc))
    
elif args.train_mode == 'regression':
    loss = logger.evaluator.performance_metric()
    print ('loss: {}'.format(loss))
