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
parser.add_argument("--train-type", type=str, default="supervised")  # ssl
parser.add_argument("--train-mode", type=str, default="regression", choices=["regression", "binary_class"])
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--scheduler", type=str, default="poly", choices=["poly", "cos"])
parser.add_argument('--lr-sch-start', type=int, default=0)
parser.add_argument('--warmup-iters', type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--decay-rate", type=int, default=0.1)
parser.add_argument("--decay-iter", type=int, default=56000)

# Data Parameters
parser.add_argument('--data', type=str, default="whole") #mgh, bgw
parser.add_argument("--normalize-label", default=False, action="store_true")  # used to normalize labels (pcwp)
parser.add_argument("--label", type=str, default="pcwp", choices=["pcwp", "age", "gender"])
parser.add_argument("--pcwp-th", type=int, default=18)

# Model Parameters
parser.add_argument("--model", type=str, default="cnn1d_lstm")  # model name

# Architecture Parameters
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--hidden-dim", type=int, default=100)

# Loss Parameters
parser.add_argument("--eps", type=float, default=1e-6)  # eps for RMSE

# Logging Parameters
parser.add_argument("--log-iter", type=int, default=10)
parser.add_argument("--val-iter", type=int, default=10)
parser.add_argument("--save-iter", type=int, default=100)

# Test / Store Parameters
parser.add_argument("--best", default=True, action="store_true")
parser.add_argument("--last", default=False, action="store_true")
parser.add_argument("--plot-prob", default=False, action="store_true")
args = parser.parse_args()

# Dataset Path settings
with open("path_configs.yaml") as f:
    path_configs = yaml.safe_load(f)
    args.dir_csv = path_configs["dir_csv"]
    # args.dir_root = path_configs['dir_root']
    args.dir_result = path_configs["dir_result"]

# Device Settings
if args.device is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = str(args.device[0])
    for i in range(len(args.device) - 1):
        device += "," + str(args.device[i + 1])
    os.environ["CUDA_VISIBLE_DEVICES"] = device
