import sys
from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
import torch.nn.functional as F
import os
import datetime
import deeprobust.graph.utils as utils
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.myappnp import APPNP
from models.myappnp1 import APPNP1
from models.mycheby import Cheby
from models.mygraphsage import GraphSage
#from models.gat2 import GAT, GraphData, Dpr2Pyg
from models.gat import GAT,GraphData, Dpr2Pyg
import scipy.sparse as sp
from utils_graphsaint import DataGraphSAINT
from utils import *
from gntk_cond import GNTK
import logging
from tensorboardX import SummaryWriter
from sklearn.neighbors import kneighbors_graph


# random seed setting
def main(args):
    #global args
    #args = train_args
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(args.device)
    #logging.info('start!')
    if args.dataset in ['cora', 'citeseer']:
        args.epsilon = 0.05
    else:
        args.epsilon = 0.01

    data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
    if args.dataset in data_graphsaint:
        data = DataGraphSAINT(args.dataset)
        data_full = data.data_full
    else:
        data_full = get_dataset(args.dataset)
        data = Transd2Ind(data_full)

    if args.test_model_type == 'GAT':
        res_val = []
        res_test = []
        nlayer = 2
        for i in range(args.nruns):
            best_acc_val, best_acc_test = test_gat(args, data, device, model_type='GAT',nruns=i)
            res_val.append(best_acc_val)
            res_test.append(best_acc_test)
        res_val = np.array(res_val)
        res_test = np.array(res_test)
        logging.info('Model:{}, Layer: {}'.format(args.test_model_type, nlayer))
        logging.info('TEST: Full Graph Mean Accuracy: {:.6f}, STD: {:.6f}'.format(res_test.mean(), res_test.std()))
        logging.info('TEST: Valid Graph Mean Accuracy: {:.6f}, STD: {:.6f}'.format(res_val.mean(), res_val.std()))

    else:
        logging.info('please identify the model type as GAT')

    return best_acc_val, best_acc_test, args

def get_syn_data(args, data, device, model_type=None):
    if args.best_ntk_score==1:
        adj_syn = torch.load(f'{args.load_path}/adj_{args.dataset}_{args.reduction_rate}_best_ntk_score_{args.tr_seed}.pt',
                             map_location='cpu')
        feat_syn = torch.load(f'{args.load_path}/feat_{args.dataset}_{args.reduction_rate}_best_ntk_score_{args.tr_seed}.pt',
                              map_location='cpu')
        labels_syn = torch.load(
            f'{args.load_path}/label_{args.dataset}_{args.reduction_rate}_best_ntk_score_{args.tr_seed}.pt',
            map_location='cpu')
    elif args.best_test_acc==1:
        adj_syn = torch.load(f'{args.load_path}/adj_{args.dataset}_{args.reduction_rate}_best_test_{args.tr_seed}.pt',
                             map_location='cpu')
        feat_syn = torch.load(f'{args.load_path}/feat_{args.dataset}_{args.reduction_rate}_best_test_{args.tr_seed}.pt',
                              map_location='cpu')
        labels_syn = torch.load(
            f'{args.load_path}/label_{args.dataset}_{args.reduction_rate}_best_test_{args.tr_seed}.pt',
            map_location='cpu')
    elif args.best_val_acc==1:
        adj_syn = torch.load(f'{args.load_path}/adj_{args.dataset}_{args.reduction_rate}_best_val_{args.tr_seed}.pt',
                             map_location='cpu')
        feat_syn = torch.load(f'{args.load_path}/feat_{args.dataset}_{args.reduction_rate}_best_val_{args.tr_seed}.pt',
                              map_location='cpu')
        labels_syn = torch.load(
            f'{args.load_path}/label_{args.dataset}_{args.reduction_rate}_best_val_{args.tr_seed}.pt',
            map_location='cpu')

    elif args.init == 1:
        adj_syn = torch.load(f'{args.load_path}/adj_{args.dataset}_{args.reduction_rate}_initial_{args.tr_seed}.pt',
                             map_location='cpu')
        feat_syn = torch.load(f'{args.load_path}/feat_{args.dataset}_{args.reduction_rate}_initial_{args.tr_seed}.pt',
                              map_location='cpu')
        labels_syn = torch.load(
            f'{args.load_path}/label_{args.dataset}_{args.reduction_rate}_initial_{args.tr_seed}.pt',
            map_location='cpu')

    elif args.test_it != None:

        adj_syn = torch.load(
            f'{args.load_path}/adj_{args.dataset}_{args.reduction_rate}_{str(args.test_it)}_{args.tr_seed}.pt',
            map_location='cpu')
        feat_syn = torch.load(
            f'{args.load_path}/feat_{args.dataset}_{args.reduction_rate}_{str(args.test_it)}_{args.tr_seed}.pt',
            map_location='cpu')
        labels_syn = torch.load(
            f'{args.load_path}/label_{args.dataset}_{args.reduction_rate}_{str(args.test_it)}_{args.tr_seed}.pt',
            map_location='cpu')

    feat_syn = feat_syn.to(device)
    labels_syn = labels_syn.to(device)
    adj_syn = adj_syn.to(device)

    return feat_syn, adj_syn, labels_syn


def generate_labels_syn(args, data):
    from collections import Counter
    counter = Counter(data.labels_train)
    num_class_dict = {}
    n = len(data.labels_train)

    sorted_counter = sorted(counter.items(), key=lambda x: x[1])
    sum_ = 0
    labels_syn = []
    syn_class_indices = {}
    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * args.reduction_rate) - sum_
            syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]
        else:
            num_class_dict[c] = max(int(num * args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]

    return labels_syn


def test_gat(args, data, device, model_type,nruns):
    feat_syn, adj_syn, labels_syn = get_syn_data(args, data, device, model_type)
    if type(adj_syn) is not torch.Tensor:
        feat_syn, adj_syn, labels_syn = utils.to_tensor(feat_syn, adj_syn, labels_syn, device=device)
    else:
        feat_syn, adj_syn, labels_syn = feat_syn.to(device), adj_syn.to(device), labels_syn.to(device)


    if utils.is_sparse_tensor(adj_syn):
        adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=True)
    else:
        adj_syn_norm = utils.normalize_adj_tensor(adj_syn)

    adj_syn = adj_syn_norm
    weight_decay = args.test_wd
    lr = args.test_lr_model


    logging.info('======= testing {}'.format(model_type))


    dropout = args.test_dropout

    data_train = GraphData(feat_syn, adj_syn, labels_syn)

    data_train = Dpr2Pyg(data_train)[0]

    data_full = GraphData(data.feat_full, data.adj_full, None)
    data_full = Dpr2Pyg(data_full)[0]

    data_val = GraphData(data.feat_val, data.adj_val, None)
    data_val = Dpr2Pyg(data_val)[0]

    data_test = GraphData(data.feat_test, data.adj_test, None)
    data_test = Dpr2Pyg(data_test)[0]


    model = GAT(nfeat=feat_syn.shape[1], nhid=16, heads=16, dropout=dropout,
                nclass=data.nclass, device=device, dataset=args.dataset).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc_val = best_acc_test = best_acc_it = 0

    train_iters = args.test_model_iters

    for iter in range(train_iters):
        model.train()
        optimizer.zero_grad()
        output_syn = model.forward(data_train)
        loss_train = F.nll_loss(output_syn, data_train.y)
        acc_syn = utils.accuracy(output_syn, data_train.y)

        loss_train.backward()
        optimizer.step()

        writer.add_scalar('train/loss_val_curve_' + str(nruns), loss_train.item(), iter)
        writer.add_scalar('train/acc_val_curve_' + str(nruns), acc_syn.item(), iter)

        logging.info('Epoch {}, training loss: {}, training acc: {}'.format(iter, loss_train.item(), acc_syn.item()))

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).to(device)
        labels_val = torch.LongTensor(data.labels_val).to(device)
        if args.dataset in ['reddit', 'flickr']:
            output_val = model.predict(data_val)
            loss_val = F.nll_loss(output_val, labels_val)
            acc_val = utils.accuracy(output_val, labels_val)

            output_test = model.predict(data_test)
            loss_test = F.nll_loss(output_test, labels_test)
            acc_test = utils.accuracy(output_test, labels_test)

            logging.info(
                "Validation set results: loss= {:.4f},accuracy= {:.4f}".format(loss_val.item(), acc_val.item()))
            logging.info(
                "Test full set results with best validation performance: loss= {:.4f}, accuracy= {:.4f}".format(
                    loss_test.item(),
                    acc_test.item()))
            writer.add_scalar('val/loss_val_curve_' + str(nruns), loss_val.item(), iter)
            writer.add_scalar('val/acc_val_curve_' + str(nruns), acc_val.item(), iter)
            writer.add_scalar('test/loss_test_curve_' + str(nruns), loss_test.item(), iter)
            writer.add_scalar('test/acc_test_curve_' + str(nruns), acc_test.item(), iter)

            if acc_val.item() > best_acc_val:
                best_acc_val = acc_val.item()
                best_acc_test = acc_test.item()
                best_acc_it = iter

        else:
            # Full graph
            output = model.predict(data_full)
            loss_val = F.nll_loss(output[data.idx_val], labels_val)
            acc_val = utils.accuracy(output[data.idx_val], labels_val)
            loss_test = F.nll_loss(output[data.idx_test], labels_test)
            acc_test = utils.accuracy(output[data.idx_test], labels_test)

            logging.info(
                "Validation set results: loss= {:.4f},accuracy= {:.4f}".format(loss_val.item(), acc_val.item()))
            logging.info(
                "Test full set results with best validation performance: loss= {:.4f}, accuracy= {:.4f}".format(
                    loss_test.item(),
                    acc_test.item()))

            writer.add_scalar('val/loss_val_curve_' + str(nruns), loss_val.item(), iter)
            writer.add_scalar('val/acc_val_curve_' + str(nruns), acc_val.item(), iter)
            writer.add_scalar('test/loss_test_curve_' + str(nruns), loss_test.item(), iter)
            writer.add_scalar('test/acc_test_curve_' + str(nruns), acc_test.item(), iter)

            if acc_val.item() > best_acc_val:
                best_acc_val = acc_val.item()
                best_acc_test = acc_test.item()
                best_acc_it = iter
        

    logging.info('FINAL BEST ACC TEST: {:.6f} with in {}-iteration'.format(best_acc_test, best_acc_it))
    return best_acc_val, best_acc_test



def nearest_neighbors(feat_syn, k, metric):
    adj = kneighbors_graph(feat_syn, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--test_nlayers', type=int, default=2)
    parser.add_argument('--test_hidden', type=int, default=256)
    parser.add_argument('--reduction_rate', type=float, default=0.25)
    parser.add_argument('--test_wd', type=float, default=5e-4)
    parser.add_argument('--test_lr_model', type=float, default=0.01)
    parser.add_argument('--test_dropout', type=float, default=0.0)
    parser.add_argument('--tr_seed', type=int, default=15, help='Random seed in condensation.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--test_model_iters', type=int, default=600)
    parser.add_argument('--epsilon', type=float, default=-1)
    parser.add_argument('--nruns', type=int, default=10)
    parser.add_argument('--test_model_type', type=str, default='GAT')
    parser.add_argument('--save_log', type=str, default='logs', help='path to save log')
    parser.add_argument('--load_path', type=str, default='./logs/Distill/cora-reduce_0.25-20221026-135148-821946',
                        help='path to load dataset')
    parser.add_argument('--best_ntk_score', type=int, default=1, help='whether use the best condensed graph data')
    parser.add_argument('--best_test_acc', type=int, default=0, help='whether use the best condensed graph data')
    parser.add_argument('--best_val_acc', type=int, default=0, help='whether use the best condensed graph data')
    parser.add_argument('--test_it', type=int, default=None, help='test evaluation iteration feat and adj')
    parser.add_argument('--init', type=int, default=0, help='test evaluation iteration feat and adj')
    parser.add_argument('--whole_data', type=int, default=0, help='whether run whole data')
    parser.add_argument('--lr_decay', type=int, default=0, help='whether half epoch decay lr')
    parser.add_argument('--test_opt_type', type=str, default='Adam',help='choosing the optimizer type')

    args = parser.parse_args()

    log_dir = './' + args.save_log + '/Test/{}-model_{}-reduce_{}-{}'.format(args.dataset, args.test_model_type,
                                                                             str(args.reduction_rate),
                                                                             datetime.datetime.now().strftime(
                                                                                 "%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'test.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))
    writer = SummaryWriter(log_dir + '/tbx_log')
    main(args)
    logging.info(args)
    logging.info('Finish!, Log_dir: {}'.format(log_dir))
