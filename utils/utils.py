import random
import numpy as np

import torch
import torch.nn as nn

def set_seeds(args):
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

def set_devices(args):
	if args.cpu or not torch.cuda.is_available():
		return torch.device('cpu')
	else:
		return torch.device('cuda')
