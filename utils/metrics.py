import os
import random
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    roc_curve,
)

import torch

metric_names = [
    "tpr",
    "tnr",
    "fpr",
    "fnr",
    "fdr",
    "ppv",
    "f1",
    "auc",
    "apr",
    "acc",
    "loss",
]
score_to_dict = lambda name, score: dict((name[i], score[i]) for i in range(len(score)))


class Evaluator(object):
    """Evaluator Object for 
    prediction performance"""

    def __init__(self, args=None):
        if args != None:
            self.args = args
            self.batch_size = args.batch_size
        self.confusion_matrix = np.zeros((2, 2))
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
        self.loss = 0
        self.best_auc = 0
        self.threshold = 0.5

    def add_batch(self, y_true, y_pred, loss, test=False):
        if self.args.train_mode == "regression":
            if test == True:
                self.loss += loss
            else:
                self.loss = loss

        elif self.args.train_mode == "binary_class":
            self.y_true.append(y_true)
            self.y_pred_proba.append(y_pred)
            self.y_pred.append(np.array(y_pred > self.threshold).astype(int))

            self.confusion_matrix += confusion_matrix((y_pred > self.threshold), y_true)

    def performance_metric(self):
        if self.args.train_mode == "regression":  ##can be deleted later
            loss = self.loss

            return loss

        elif self.args.train_mode == "binary_class":
            #  np.concatenate transforms list of ndarrays from each batch into one ndarray having a shape of (n_samples,)
            self.y_true = np.concatenate(self.y_true, axis=0)
            self.y_pred_proba = np.concatenate(self.y_pred_proba, axis=0)
            self.y_pred = np.concatenate(self.y_pred, axis=0)
            auc = roc_auc_score(self.y_true, self.y_pred_proba)
            apr = average_precision_score(self.y_true, self.y_pred_proba)
            acc = accuracy_score(self.y_true, self.y_pred)
            f1 = f1_score(self.y_true, self.y_pred)

            return f1, auc, apr, acc

    def reset(self):
        self.confusion_matrix = np.zeros((2,) * 2)
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
        self.loss = np.inf
        self.threshold = 0.5
