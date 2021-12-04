import importlib
import torch.nn as nn
import torchvision.models as models

from model.resnet import resnet18, resnet34, resnet18_contrastive
import model.resnetsup

def get_model(args, device=None):

    if args.model == "cnn":
        model = resnetsup.resnet18(args)
        pass

    elif args.model == "contrastive":
        if args.viewtype == 'clocslead':
            model = resnet18_contrastive(embed_size = args.embed_size, nleads=6)
        else:
            model = resnet18_contrastive(embed_size = args.embed_size)
        pass

    else:
        raise NotImplementedError

    model = model.to(device)
    model = nn.DataParallel(model)

    return model