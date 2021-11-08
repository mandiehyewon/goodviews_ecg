import torch
import torch.nn.functional as F
import numpy as np


def get_contrastive_loss(features, labels, args):
    # temperature = 0.07
    # batch_size = 64
    features = F.normalize(
        features, p=2, dim=1
    )  # normalize for each row's L-2 norm become 1; this makes torch.matmul produce cosine similiarity matrix without scaling factors
    similarity_matrix = torch.matmul(
        features, features.T
    )  # similiarty matrix.shape = [batch_size, batch_size]
    # labels = torch.randint(2, similarity_matrix.shape)

    similarity_matrix = similarity_matrix * (
        1 - torch.eye(args.batch_size, args.batch_size)
    )  # make each diagonal to be zero
    similarity_matrix = similarity_matrix / args.temperature
    similarity_matrix_exp = torch.exp(similarity_matrix)
    numerators = torch.sum(torch.mul(similarity_matrix_exp, labels), dim=1)
    denominators = torch.sum(similarity_matrix_exp, dim=1)
    loss = -torch.log(torch.div(numerators, denominators))
    loss = loss.sum(dim=0)
    return loss


if __name__ == "__main__":
    temperature = 0.07  #
    batch_size = 64
    dim = 100
    features = torch.rand(batch_size, dim)
    features = F.normalize(
        features, p=2, dim=1
    )  # normalize for each row's L-2 norm become 1; this makes torch.matmul produce cosine similiarity matrix without scaling factors
    similarity_matrix = torch.matmul(
        features, features.T
    )  # similiarty matrix.shape = [batch_size, batch_size]
    labels = torch.randint(2, similarity_matrix.shape)

    similarity_matrix = similarity_matrix * (
        1 - torch.eye(batch_size, batch_size)
    )  # make each diagonal to be zero
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix_exp = torch.exp(similarity_matrix)
    numerators = torch.sum(torch.mul(similarity_matrix_exp, labels), dim=1)
    denominators = torch.sum(similarity_matrix_exp, dim=1)
    loss = -torch.log(torch.div(numerators, denominators))
    loss = loss.sum(dim=0)
    loss.backward()
