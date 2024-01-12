from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from SGDD_agent import SGDD
from utils_graphsaint import DataGraphSAINT
from output import datastorage

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=7, help='gpu id')
# parser.add_argument('--dataset', type=str, default='cora')
# parser.add_argument('--dataset', type=str, default='citeseer')
# parser.add_argument('--dataset', type=str, default='flickr')
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
# parser.add_argument('--dataset', type=str, default='yelpchi')
# parser.add_argument('--dataset', type=str, default='sbm')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=2000)
# parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=1e-4)
parser.add_argument('--lr_feat', type=float, default=1e-4)
# parser.add_argument('--lr_adj', type=float, default=0.01)
# parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=15, help='Random seed.') # [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
parser.add_argument('--beta', type=float, default=0.5, help='regularization term.')
parser.add_argument('--ep_ratio', type=float, default=0.5, help='control the ratio of direct \
                     edges predict term in the graph.')
parser.add_argument('--sinkhorn_iter', type=int, default=5, help='use sinkhorn iteration to \
                    warm-up the transport plan.')
parser.add_argument('--opt_scale', type=float, default=1e-10, help='control the scale of the opt loss')
parser.add_argument("--ignr_epochs", type=int, default=400, help="use the few epochs to warm-up structure learning")
# parser.add_argument('--mx_size', type=int, default=2708, help='max size of the matrix to')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--option', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--one_step', type=int, default=0)
parser.add_argument('--mode', type=str, default='disabled', help='whether to use the wandb')
parser.add_argument('--inittype', type=str, default=None)
parser.add_argument('--coreset_init_path', type=str, default="saved_core")

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

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

# change the standard output to store in file
file_name_out = f'print_output/{args.dataset}_{args.reduction_rate}_{args.seed}__{args.one_step}sgc.txt'
original_stdout = redirect_stdout_to_file(file_name_out)

print(args) # print it

seed_list = [1,15,85,120,1000]
init_list = ["herding", "kcenter"]
compr_list = {"cora": [0.25,0.5], "citeseer": [0.25,0.5],"reddit":[0.001,0.005,0.01]}
data_list = ["cora", "citeseer","reddit"]
method_list = ["gcond", "sgdd"]

myresults = datastorage(seed_list,init_list,compr_list,data_list,method_list)
args.out = myresults
args.out.load_table()


data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

if data_full.adj.shape[0] < 5000:
    args.mx_size = data_full.adj.shape[0]
else:
    args.mx_size = 5000
    data_full.adj_mx = data_full.adj[:args.mx_size, :args.mx_size]
agent = SGDD(data, args, device='cuda')

agent.train()
args.out.display_save()


# restore the original standard output
restore_stdout(original_stdout)
#hello