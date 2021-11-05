#!/usr/bin/env python3
import os
import sys
import shutil
import copy
import logging
import logging.handlers
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from utils.metrics import Evaluator, metric_names


class Logger:
    def __init__(self, args):
        self.args = args
        self.args_save = copy.deepcopy(args)

        # Evaluator
        self.evaluator = Evaluator(self.args)

        # Checkpoint and Logging Directories
        self.dir_root = os.path.join(args.dir_result, args.name)
        self.dir_log = os.path.join(self.dir_root, 'logs')
        self.dir_save = os.path.join(self.dir_root, 'ckpts')

        self.log_iter = args.log_iter

        if args.reset and os.path.exists(self.dir_root):
            shutil.rmtree(self.dir_root, ignore_errors=True)
        if not os.path.exists(self.dir_root):
            os.makedirs(self.dir_root)
        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)
        elif os.path.exists(os.path.join(self.dir_save, 'last.pth')) and os.path.exists(self.dir_log):
            shutil.rmtree(self.dir_log, ignore_errors=True)
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        # Tensorboard Writer
        self.writer = SummaryWriter(logdir=self.dir_log, flush_secs=60)
        
        # Log variables
        self.loss = 0
        self.best_auc = 0
        self.best_iter = 0
        self.best_results = []
        self.best_loss = 0

        # print(self.args_save)

    def log_tqdm(self, pbar):
        if self.args.train_mode =='regression':
            tqdm_log = 'loss: {:.5f}, best_loss: {:.5f}, best_iter: {}'.format(self.loss/self.log_iter, self.best_loss, self.best_iter)
        else:
            tqdm_log = 'loss: {:.5f}, auc: {:.5f}, best_iter: {}'.format(self.loss/self.log_iter, self.best_auc, self.best_iter)
        pbar.set_description(tqdm_log)
        
    def log_scalars(self, step):
        self.writer.add_scalar('loss', self.loss / self.log_iter, global_step=step)

    def loss_reset(self):
        self.loss = 0

    def add_validation_logs(self, step, loss):

        if self.args.train_mode == 'regression':
            # loss = self.evaluator.performance_metric()
            if self.best_loss == 0.0:
                self.best_loss = loss
                self.best_iter = step
            else:
                if self.best_loss > loss:
                    self.best_loss = loss
                    self.best_iter = step
            self.writer.add_scalar('val/loss', loss, global_step=step)
            self.best_results = [loss]

        else:
            auc = self.evaluator.performance_metric()
            if self.best_auc < auc:
                self.best_iter = step
                self.best_auc = auc
                self.best_results = [auc]

            self.writer.add_scalar('val/auroc', auc, global_step=step)
            self.writer.flush()

    def save(self, model, optimizer, step, last=None, k_fold_num=0):
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_results': self.best_results, 'best_step': step, 'last_step' : last}

        if step == self.best_iter:
            self.save_ckpt(ckpt, 'best.pth')
        if last:
            self.save_ckpt(ckpt, 'last.pth')
        elif step % self.args.save_iter == 0:
            self.save_ckpt(ckpt, '{}.pth'.format(step))

        return ckpt

    def save_ckpt(self, ckpt, name):
        torch.save(ckpt, os.path.join(self.dir_save, name))
