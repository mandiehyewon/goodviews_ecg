import torch
import torch.nn as nn

def get_loss(args):
    if args.train_mode == 'regression':
        return RMSELoss(args)
        # return nn.MSELoss()
        
    elif args.train_mode == 'binary_class':
        # return nn.CrossEntropyLoss()
        return nn.BCEWithLogitsLoss()
        # return nn.BCELoss(reduction='mean')


def get_pclr_loss(device):
    return torch.nn.CrossEntropyLoss().to(device)

class RMSELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = args.eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss