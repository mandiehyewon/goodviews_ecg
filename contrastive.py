import torch
import torch.nn.functional as F
import numpy as np
from config import args


def get_contrastive_loss(features, labels, args):
    features = F.normalize(
        features, p=2, dim=1
    )  # normalize for each row's L-2 norm become 1; this makes torch.matmul produce cosine similiarity matrix without scaling factors
    similarity_matrix = torch.matmul(
        features, features.T
    )  # similiarty matrix.shape = [batch_size, batch_size]

    similarity_matrix = similarity_matrix * (
        1 - torch.eye(args.batch_size, args.batch_size)
    )  # make each diagonal to be zero
    similarity_matrix = similarity_matrix / args.temperature
    similarity_matrix_exp = torch.exp(similarity_matrix)
    # print(labels)
    # neg_labels = (labels == 0).type(torch.uint8)
    numerators = torch.sum(torch.mul(similarity_matrix_exp, labels), dim=1)
    # denominators = torch.sum(torch.mul(similarity_matrix_exp, neg_labels), dim=1)
    denominators = torch.sum(similarity_matrix_exp, dim=1)  # denominator should be > numerators
    p = torch.div(numerators, denominators)
    # print("p", p, 0 < p, p <1)
    eps = 1e-7
    loss = - torch.log(p+eps)
    loss = loss.sum(dim=0)
    loss /= args.batch_size
    return loss


if __name__ == "__main__":
    temperature = 0.07  #
    batch_size = 64
    dim = 100
    labels = torch.randint(2, [args.batch_size, args.batch_size])
    features = torch.rand(batch_size, dim)
    get_contrastive_loss(features, labels, args)
