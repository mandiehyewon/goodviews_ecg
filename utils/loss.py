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


def get_contrastive_loss(device):
    return torch.nn.CrossEntropyLoss().to(device)


def contrastive_logits(features, device):
    labels = torch.cat([torch.arange(args.batch_size) for i in range(args.batch_size//2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)  # features are normalized first so that simple inner product by matmul
    # produce the cosine similarity.

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix - because we don't use (z_i, z_j) pairs
    # where i=j.
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    logits = logits / self.args.temperature
    return logits, labels


class RMSELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = args.eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss