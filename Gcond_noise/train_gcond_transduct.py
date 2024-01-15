# test_idx, train_idx, val_idx - <dataset,r, seed>
# Random seed, args, epoch wise metrics, accuracy metrics,
# Evaluation Model.pt : At test time
from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from gcond_agent_transduct import GCond
from utils_graphsaint import DataGraphSAINT
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=1)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--one_step', type=int, default=0)

parser.add_argument('--noise_type', type=str, default='edge_add')
parser.add_argument('--noise', type=float, default=0)

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# all output to txt file
import sys

# Define a function to redirect stdout to a file
def redirect_stdout_to_file(filename):
    original_stdout = sys.stdout
    sys.stdout = open(filename, 'a')  # Use 'a' (append) mode to append to the file
    return original_stdout

# Restore the original stdout
def restore_stdout(original_stdout):
    sys.stdout.close()
    sys.stdout = original_stdout

file_name_out = f'print_output/print_{args.dataset}_{args.reduction_rate}_{args.seed}_{args.one_step}_{args.noise_type}_transduct.txt'
original_stdout = redirect_stdout_to_file(file_name_out)

print(args)
# save args
# Convert the argparse namespace to a dictionary
args_dict = vars(args)

# # Specify the file name for saving the arguments
# args_file = f'args_parsed/{args.dataset}_{args.reduction_rate}_{args.seed}_{args.one_step}_{args.noise_type}_transduct.json'

# # Save the arguments to a JSON file
# with open(args_file, 'w') as file:
#     json.dump(args_dict, file, indent=4)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset, args.normalize_features, noise_type=args.noise_type, noise=args.noise)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
# saving the data indices 
# file_name_idx = f'Index/{args.dataset}_{args.reduction_rate}_{args.seed}_induct.npz'
# np.savez(file_name_idx, idx_train=data.idx_train, idx_test=data.idx_test, idx_val=data.idx_val)

agent = GCond(data, args, device='cuda')

agent.train()

# restore the original standard output
restore_stdout(original_stdout)