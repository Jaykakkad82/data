from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
import sys
from deeprobust.graph import utils
from deeprobust.graph.utils import *
import torch.nn.functional as F
from configs import load_config
from utils import *
from utils_graphsaint import DataGraphSAINT
from models.gcn import GCN
from coreset import KCenter, Herding, Random, kmeans
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--nlayers', type=int, default=2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--inductive', type=int, default=1)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--method', type=str, choices=['kcenter', 'herding', 'random', 'kmeans'])
parser.add_argument('--reduction_rate', type=float, required=True)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
args = load_config(args)


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

file_name_out = f'Coreset_print_output/{args.method}_{args.dataset}_{args.reduction_rate}_{args.seed}_induct.txt'
original_stdout = redirect_stdout_to_file(file_name_out)




print(args)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

features = data_full.features
adj = data_full.adj
labels = data_full.labels
idx_train = data_full.idx_train
idx_val = data_full.idx_val
idx_test = data_full.idx_test

# Setup GCN Model
device = 'cuda'
model = GCN(nfeat=features.shape[1], nhid=256, nclass=labels.max()+1, device=device, weight_decay=args.weight_decay)

model = model.to(device)
model.fit_with_val(features, adj,labels, data,train_iters=80)
#model.fit(features, adj, labels, idx_train, idx_val, train_iters=600, verbose=False)
# optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# for e in range(args.epochs + 1):
#     model.train()
#     optimizer_model.zero_grad()
#     embed, output = model.forward(features, adj)
#     loss = F.nll_loss(output[idx_train], labels[idx_train])
#     acc = utils.accuracy(output[idx_train], labels[idx_train])

#     # print('Epochs:', e, 'Full graph train set results: loss = ',loss.item(), 'accuracy=',acc.item())
#     print('=========Train===============')
#     print( 'Epochs={}: Full graph train set results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss.item(), acc.item()))
#     loss.backward()
#     optimizer_model.step()
#     if e % 10 == 0:
#         model.eval()
#         # You can use the inner function of model to test
#         _, output_test = model.forward(features, adj)
#         loss_test = F.nll_loss(output_test[idx_test], labels_test)
#         acc_test = utils.accuracy(output_test[idx_test], labels_test)
#         print('=============Testing===============')
#         #logging.info('=========Testing===============')
#         #print('Epochs:', e, 'Test results: loss = ',loss_test.item(), 'accuracy=',acc_test.item())
#         print('Epochs={}: Test results: loss = {:.4f}, accuracy = {:.4f}'.format(e, loss_test.item(), acc_test.item()))

#     embed_out = embed


model.eval()
# You can use the inner function of model to test
model.test(idx_test)

embeds = model.predict().detach()

if args.method == 'kcenter':
    agent = KCenter(data, args, device='cuda')
if args.method == 'herding':
    agent = Herding(data, args, device='cuda')
if args.method == 'random':
    agent = Random(data, args, device='cuda')
if args.method == 'kmeans':
    agent = kmeans(data, args, device='cuda')

idx_selected = agent.select(embeds)


feat_train = features[idx_selected]
adj_train = adj[np.ix_(idx_selected, idx_selected)]

labels_train = labels[idx_selected]

if args.save:
    np.save(f'saved_core/idx_{args.dataset}_{args.reduction_rate}_{args.method}_{args.seed}.npy', idx_selected)


res = []
runs = 10
for _ in tqdm(range(runs)):
    model.initialize()
    model.fit_with_val(feat_train, adj_train, labels_train, data,
                 train_iters=600, normalize=True, verbose=False)

    model.eval()
    labels_test = torch.LongTensor(data.labels_test).cuda()

    # Full graph
    output = model.predict(data.feat_full, data.adj_full)
    loss_test = F.nll_loss(output[data.idx_test], labels_test)
    acc_test = utils.accuracy(output[data.idx_test], labels_test)
    res.append(acc_test.item())

res = np.array(res)
print('Mean accuracy:', repr([res.mean(), res.std()]))
torch.save(model.state_dict(), f'Eval_coreset_model/model__{args.method}_{args.dataset}_{args.reduction_rate}_{args.seed}_trans.pt')
# restore the original standard output
restore_stdout(original_stdout)

