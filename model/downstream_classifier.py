# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 20:14
直接修改torch的resnet
@ author: javis

Adopted From:
https://github.com/Amadeuszhao/SE-ECGNet/blob/master/models/resnet.py
'''

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['linear_classifier']

class linear_classifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(linear_classifier, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layer(x)
        return x
