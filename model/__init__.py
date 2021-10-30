import importlib
import torch.nn as nn
import torchvision.models as models

from model.resnet import resnet18, resnet34

def get_model(args, device=None):

    if args.model == 'cnn':
        model = resnet18()
        pass

    elif args.model == 'cnn_prev':
        model = models.resnet18(pretrained=False).to(device)
        # model.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        # model.avgpool = nn.AdaptiveAvgPool2d(1)
        # model.fc = nn.Linear(in_features=512, out_features=1, bias=True)

        # if args.regression:
            # model.

    else:
        model_module = importlib.import_module("model." + args.model)
        model_class = getattr(model_module, args.model.upper())
        model = model_class(args, device)

    model = model.to(device)
    model = nn.DataParallel(model)

    return model
