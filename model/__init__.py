import importlib
import torch.nn as nn
import torchvision.models as models

from model.resnet import resnet18, resnet34


def get_model(args, device=None):

    if args.model == "cnn":
        model = resnet18()
        pass

    else:
        raise NotImplementedError

    model = model.to(device)
    model = nn.DataParallel(model)

    return model
