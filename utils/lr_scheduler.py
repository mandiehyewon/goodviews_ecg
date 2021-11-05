##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math

class LR_Scheduler(object):
    """Learning Rate Scheduler
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    """
    def __init__(self, optimizer, mode, base_lr, num_iters,
                 from_iter=0, warmup_iters=0, decay_degree=0.9, functional=False):
        self.optimizer = optimizer
        self.mode = mode
        # print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        self.iter = from_iter
        self.N = num_iters + 1 
        self.warmup_iters = warmup_iters
        self.decay_degree = decay_degree
        self.functional = functional

    def step(self):
        self.iter += 1
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * self.iter / self.N * math.pi))
        elif self.mode == 'poly':
            if self.iter == self.N:
                lr = 0.0
            else:
                lr = self.lr * pow((1 - 1.0 * self.iter / self.N), self.decay_degree)
        elif self.mode == 'constant':
            return
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and self.iter < self.warmup_iters:
            lr = lr * 1.0 * self.iter / self.warmup_iters
        assert lr >= 0
        self._adjust_learning_rate(self.optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if self.functional:
            optimizer.lr = lr
        else:
            if len(optimizer.param_groups) == 1:
                optimizer.param_groups[0]['lr'] = lr
            else:
                # enlarge the lr at the head
                optimizer.param_groups[0]['lr'] = lr
                for i in range(1, len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr * 10