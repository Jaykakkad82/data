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
from models.gat import GAT
import scipy.sparse as sp
from utils_graphsaint import DataGraphSAINT
from utils import *
from gntk_cond import GNTK
import logging
from tensorboardX import SummaryWriter
from sklearn.neighbors import kneighbors_graph


# random seed setting
def main(args):
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

    res_val = []
    res_test = []
    nlayer = 2
    for i in range(args.nruns):
        best_acc_val, best_acc_test = test(args, data, device, nlayer, model_type=args.test_model_type, nruns=i)
        res_val.append(best_acc_val)
        res_test.append(best_acc_test)
    res_val = np.array(res_val)
    res_test = np.array(res_test)
    logging.info('Model:{}, Layer: {}'.format(args.test_model_type, nlayer))
    logging.info('TEST: Full Graph Mean Accuracy: {:.6f}, STD: {:.6f}'.format(res_test.mean(), res_test.std()))
    logging.info('TEST: Valid Graph Mean Accuracy: {:.6f}, STD: {:.6f}'.format(res_val.mean(), res_val.std()))

    return best_acc_val, best_acc_test, args


def test(args, data, device, nlayers, model_type, nruns):

    if args.whole_data != 1:
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

    else:
        logging.info('THIS IS THE ORIGINAL WHOLE DATA...')
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        features, adj, labels = utils.to_tensor(features, adj, labels, device=device)
        feat_syn, labels_syn = features, labels
        if utils.is_sparse_tensor(adj):
            adj_syn_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_syn_norm = utils.normalize_adj_tensor(adj)

        adj_syn = adj_syn_norm
        weight_decay = 5e-4
        lr = 0.01

    if model_type == 'MLP':
        adj_syn = adj_syn - adj_syn
        model_class = GCN
    else:
        model_class = eval(model_type)

    dropout = args.test_dropout

    model = model_class(nfeat=feat_syn.shape[1], nhid=args.test_hidden, dropout=dropout, nlayers=args.test_nlayers,
                        nclass=data.nclass, device=device).to(device)

    logging.info(model)

    logging.info('=== training {} model ==='.format(model_type))

    if args.test_opt_type=='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif args.test_opt_type=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)

    best_acc_val = best_acc_test = best_epoch = 0

    train_iters = args.test_model_iters

    for i in range(train_iters):
        if i == train_iters // 2 and args.lr_decay == 1:
            lr = args.test_lr_model * 0.1
            if args.test_opt_type == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif args.test_opt_type == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        if args.whole_data == 1:
            model.train()
            optimizer.zero_grad()
            output_syn = model.forward(feat_syn, adj_syn)
            loss_train = F.nll_loss(output_syn[data.idx_train], labels_syn[data.idx_train])
            acc_syn = utils.accuracy(output_syn[data.idx_train], labels_syn[data.idx_train])
        else:
            model.train()
            optimizer.zero_grad()
            output_syn = model.forward(feat_syn, adj_syn)
            loss_train = F.nll_loss(output_syn, labels_syn)
            acc_syn = utils.accuracy(output_syn, labels_syn)

        loss_train.backward()
        optimizer.step()

        writer.add_scalar('train/loss_val_curve_' + str(nruns), loss_train.item(), i)
        writer.add_scalar('train/acc_val_curve_' + str(nruns), acc_syn.item(), i)

        if i % 1 == 0:

            logging.info('Epoch {}, training loss: {}, training acc: {}'.format(i, loss_train.item(), acc_syn.item()))
            model.eval()
            labels_test = torch.LongTensor(data.labels_test).to(device)
            labels_val = torch.LongTensor(data.labels_val).to(device)

            if args.dataset in ['reddit', 'flickr']:
                output = model.predict(data.feat_val, data.adj_val)
                loss_val = F.nll_loss(output, labels_val)
                acc_val = utils.accuracy(output, labels_val)

                output_test = model.predict(data.feat_test, data.adj_test)
                loss_test = F.nll_loss(output_test, labels_test)
                acc_test = utils.accuracy(output_test, labels_test)

                logging.info(
                    "Validation set results: loss= {:.4f},accuracy= {:.4f}".format(loss_val.item(), acc_val.item()))
                logging.info(
                    "Test full set results with best validation performance: loss= {:.4f}, accuracy= {:.4f}".format(
                        loss_test.item(),
                        acc_test.item()))
                writer.add_scalar('val/loss_val_curve_' + str(nruns), loss_val.item(), i)
                writer.add_scalar('val/acc_val_curve_' + str(nruns), acc_val.item(), i)
                writer.add_scalar('test/loss_test_curve_' + str(nruns), loss_test.item(), i)
                writer.add_scalar('test/acc_test_curve_' + str(nruns), acc_test.item(), i)

                if acc_val.item() > best_acc_val:
                    best_acc_val = acc_val.item()
                    best_acc_test = acc_test.item()
                    best_epoch = i

            # if not args.dataset in ['reddit', 'flickr']:
            else:
                # Full graph
                output = model.predict(data.feat_full, data.adj_full)
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

                writer.add_scalar('val/loss_val_curve_' + str(nruns), loss_val.item(), i)
                writer.add_scalar('val/acc_val_curve_' + str(nruns), acc_val.item(), i)
                writer.add_scalar('test/loss_test_curve_' + str(nruns), loss_test.item(), i)
                writer.add_scalar('test/acc_test_curve_' + str(nruns), acc_test.item(), i)

                if acc_val.item() > best_acc_val:
                    best_acc_val = acc_val.item()
                    best_acc_test = acc_test.item()
                    best_epoch = i
                #if acc_test.item() > best_acc_test:

    logging.info('FINAL BEST ACC TEST: {:.6f} within BEST EPOCH: {}'.format(best_acc_test,best_epoch))
    return best_acc_val, best_acc_test


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
    parser.add_argument('--test_model_type', type=str, default='GCN')
    parser.add_argument('--save_log', type=str, default='logs', help='path to save log')
    parser.add_argument('--load_path', type=str, default='./logs/Distill/cora-reduce_0.25-20221026-135148-821946',
                        help='path to load dataset')
    parser.add_argument('--best_ntk_score', type=int, default=0, help='whether use the best condensed graph data')
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
