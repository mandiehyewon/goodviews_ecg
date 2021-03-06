import os
import yaml
import argparse

### CONFIGURATIONS
parser = argparse.ArgumentParser()

# General Parameters
parser.add_argument("--device", type=int, default=None, nargs="+")
parser.add_argument("--cpu", default=False, action="store_true")
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--reset", default=False, action="store_true")
parser.add_argument("--seed", type=int, default=926)

# Training Parameters
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--dw-epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--dw-lr", type=float, default=1e-2)
parser.add_argument("--scheduler", type=str, default="poly", choices=["poly", "cos"])
parser.add_argument('--lr-sch-start', type=int, default=0)
parser.add_argument('--warmup-iters', type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--decay-rate", type=int, default=0.1)
parser.add_argument("--decay-iter", type=int, default=56000)
parser.add_argument("--clip", type=float, default=1)

# Training Parameters for Contrastive Learning
parser.add_argument("--temperature", type=float, default=0.07)


# Data Parameters
parser.add_argument('--data', type=str, default="whole")
parser.add_argument("--normalize", default=False, action="store_true")

# Augmentation Parameters
parser.add_argument('--no-preaug', action="store_false", help='do not use priorly augmented data')
parser.add_argument('--num-augments', type=int, default=4, help='number of types of augments to use')
parser.add_argument('--amp-min', type=float, default=0.5)
parser.add_argument('--amp-max', type=int, default=2)
parser.add_argument('--tshift-min', type=int, default=-50, help='number of samples')
parser.add_argument('--tshift-max', type=int, default=50, help='number of samples')
parser.add_argument('--mask-min', type=int, default=0)
parser.add_argument('--mask-max', type=int, default=150)

# Model Parameters
parser.add_argument("--model", type=str, default="contrastive")  # model name
parser.add_argument("--viewtype", type=str, default="simclr", choices=["demos", "rhythm", "clocstime", "clocslead", "simclr", "attr", "sup"])
parser.add_argument("--num-kmeans-clusters", type=int, default=40)

# Architecture Parameters
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--hidden-dim", type=int, default=100)
parser.add_argument("--embed-size", type=int, default=128)

# Loss Parameters
parser.add_argument("--eps", type=float, default=1e-6)  # eps for RMSE
parser.add_argument("--log-iter", type=int, default=10)
parser.add_argument("--save-iter", type=int, default=50)

# Testing Parameters
parser.add_argument("--load-step", type=int, default=50)

args = parser.parse_args()

# Dataset Path settings
with open("path_configs.yaml") as f:
    path_configs = yaml.safe_load(f)
    args.dir_csv = path_configs["dir_csv"]
    args.dir_result = path_configs["dir_result"]
    args.preaug_fname = path_configs["preaug_fname"]

# Device Settings
if args.device is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = str(args.device[0])
    for i in range(len(args.device) - 1):
        device += "," + str(args.device[i + 1])
    os.environ["CUDA_VISIBLE_DEVICES"] = device
