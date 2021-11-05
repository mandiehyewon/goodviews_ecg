import os
import random
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)

import torch

metric_names = ["auc"]
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
        self.best_auc = 0
        self.threshold = 0.5

    def add_batch(self, y_true, y_pred, loss, test=False):
        self.y_true.append(y_true)
        self.y_pred_proba.append(y_pred)
        self.y_pred.append(np.array(y_pred > self.threshold).astype(int))

        self.confusion_matrix += confusion_matrix((y_pred > self.threshold), y_true)

    def performance_metric(self):
        auc = roc_auc_score(self.y_true, self.y_pred_proba)
        return auc

    def reset(self):
        self.confusion_matrix = np.zeros((2,) * 2)
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
        self.loss = np.inf
        self.threshold = 0.5
