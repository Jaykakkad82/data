import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import get_eval_pool
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
from models.reparam_module import ReparamModule
import scipy.sparse as sp
from torch_sparse import SparseTensor
import logging
import os
import random
import copy
from sklearn.neighbors import kneighbors_graph
from collections import Counter
import scipy
from gntk_cond import GNTK
from tqdm import tqdm


class MetaGtt:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

        n = int(data.feat_train.shape[0] * args.reduction_rate)

        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))

        if args.optimizer_con == 'Adam':
            self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        elif args.optimizer_con == 'SGD':
            self.optimizer_feat = torch.optim.SGD([self.feat_syn], lr=args.lr_feat, momentum=0.9)

        logging.info('adj_syn: {}, feat_syn: {}'.format((n, n), self.feat_syn.shape))

    def expert_load(self):
        args = self.args
        expert_dir = args.buffer_path
        logging.info("Expert Dir: {}".format(expert_dir))

        if args.load_all:
            buffer = []
            n = 0
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))

        else:
            expert_files = []
            n = 0
            while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
                expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))
            file_idx = 0
            expert_idx = 0
            random.shuffle(expert_files)
            if args.max_files is not None:
                expert_files = expert_files[:args.max_files]
            # print("loading file {}".format(expert_files[file_idx]))
            buffer = torch.load(expert_files[file_idx])
            if args.max_experts is not None:
                buffer = buffer[:args.max_experts]
            random.shuffle(buffer)
            self.buffer = buffer

        return file_idx, expert_idx, expert_files

    def evaluate_synset_knn(self, model_eval):
        args = self.args
        eval_labs = self.labels_syn
        with torch.no_grad():
            feat_save = self.feat_syn

        feat_syn_eval, label_syn_eval = copy.deepcopy(feat_save.detach()), copy.deepcopy(
            eval_labs.detach())  # avoid any unaware modification

        adj_syn_eval = torch.from_numpy(self.nearest_neighbors(feat_syn_eval, args.k)).to(self.device)

        data, device = self.data, self.device

        args = self.args
        if model_eval == 'MLP':
            adj_syn_eval = adj_syn_eval - adj_syn_eval
            model_class = GCN
        else:
            model_class = eval(model_eval)

        adj_syn_norm = utils.normalize_adj_tensor(adj_syn_eval, sparse=False)

        dropout = 0.5 if args.dataset in ['reddit'] else 0

        net_eval = model_class(nfeat=feat_syn_eval.shape[1], nhid=args.student_hidden, dropout=dropout,
                               nlayers=args.student_nlayers,
                               nclass=data.nclass, device=self.device).to(self.device)

        if args.optimizer_eval == 'Adam':
            optimizer = optim.Adam(net_eval.parameters(), lr=args.eval_lr, weight_decay=args.eval_wd)
        elif args.optimizer_eval == 'SGD':
            optimizer = optim.SGD(net_eval.parameters(), lr=args.eval_lr, momentum=0.9, weight_decay=args.eval_wd)

        train_iters = args.eval_train_iters
        lr_schedule = [train_iters // 2 + 1]
        logging.info('Training Synthetic set ...')

        best_val_acc = 0
        best_test_acc = 0

        for i in range(train_iters):
            net_eval.train()
            optimizer.zero_grad()
            _, output_syn = net_eval.forward(feat_syn_eval, adj_syn_norm)
            loss_syn = F.nll_loss(output_syn, label_syn_eval)
            loss_syn.backward()
            optimizer.step()
            acc_syn = utils.accuracy(output_syn, label_syn_eval)

            if i in lr_schedule:
                lr = args.eval_lr * 0.1
                optimizer = torch.optim.SGD(net_eval.parameters(), lr=lr, momentum=0.9, weight_decay=args.eval_wd)

            # if i % 50 == 0:
            #    logging.info(
            #       "Synthetic set results: loss= {:.4f}, accuracy= {:.4f} ".format(loss_syn.item(), acc_syn.item()))

            if i % 10 == 0:
                net_eval.eval()
                '''
                labels_train = torch.LongTensor(data.labels_train).to(self.device)
                output = net_eval.predict(data.feat_train, data.adj_train)
                loss_train = F.nll_loss(output, labels_train)
                acc_train = utils.accuracy(output, labels_train)
                if verbose:
                    logging.info(
                        "Train set results: loss= {:.4f}, accuracy= {:.4f} ".format(loss_train.item(), acc_train.item()))
                '''
                labels_val = torch.LongTensor(data.labels_val).to(self.device)
                _, output_val = net_eval.predict(data.feat_val, data.adj_val)
                loss_val = F.nll_loss(output_val, labels_val)
                acc_val = utils.accuracy(output_val, labels_val)

                if acc_val.item() > best_val_acc:
                    best_val_acc = acc_val.item()
                    # Full graph
                    labels_test = torch.LongTensor(data.labels_test).to(self.device)
                    _, output = net_eval.predict(data.feat_full, data.adj_full)
                    loss_test = F.nll_loss(output[data.idx_test], labels_test)
                    acc_test = utils.accuracy(output[data.idx_test], labels_test)
                    if acc_test.item() > best_test_acc:
                        best_test_acc = acc_test.item()

        logging.info("Best val set results: accuracy= {:.2f} ".format(best_val_acc * 100.0))
        logging.info("Best test set results: accuracy= {:.2f} ".format(best_test_acc * 100.0))

        return best_val_acc, best_test_acc, feat_syn_eval, adj_syn_eval, label_syn_eval

    def synset_save(self):
        args = self.args
        eval_labs = self.labels_syn
        with torch.no_grad():
            feat_save = self.feat_syn

        feat_syn_eval, label_syn_eval = copy.deepcopy(feat_save.detach()), copy.deepcopy(
            eval_labs.detach())  # avoid any unaware modification

        adj_syn_eval = torch.eye(feat_syn_eval.shape[0]).to(self.device)
        #feat_knn = F.normalize(feat_syn_eval, p=1).to(self.device)
        #adj_syn_cal = torch.from_numpy(self.nearest_neighbors(feat_knn, args.k)).to(self.device)
        #adj_syn_mix = adj_syn_cal + self.adj_syn_init
        #adj_syn_mix[torch.where(adj_syn_mix.gt(1))] = 1
        #adj_syn_eval = adj_syn_mix

        return feat_syn_eval, adj_syn_eval, label_syn_eval

    def evaluate_synset_ntk(self):
        args = self.args
        eval_labs = self.labels_syn
        data = self.data
        layers = 3
        gntk = GNTK(num_layers=layers, num_mlp_layers=2, jk=0, scale='degree')
        features_0, adj_0, labels_0 = data.feat_val, data.adj_val, data.labels_val
        num_class = max(labels_0) + 1


        feat_syn_eval, adj_syn_eval, labels_syn_eval = self.synset_save()
        feat_syn, adj_syn, labels_syn = np.array(feat_syn_eval.cpu()), np.array(adj_syn_eval.cpu()), np.array(labels_syn_eval.cpu())
        adj_syn = adj_syn + scipy.sparse.identity(adj_syn.shape[0])


        diag_syn = gntk.diag(feat_syn, adj_syn)

        labels_syn = one_hot(labels_syn, num_class)

        _, sigma_syn_syn, ntk_syn_syn, dotsigma_syn_syn = calc(gntk, feat_syn, feat_syn, diag_syn, diag_syn,
                                                               adj_syn, adj_syn)
        if args.dataset in ['ogbn-arxiv','flickr', 'reddit']:
            score_syn_ls = []
            acc_syn_ls = []
            for k in range(args.samp_iter):
                node_idx = data.retrieve_class_sampler_val(transductive=False, num_per_class=args.samp_num_per_class)
                re_adj_0 = adj_0[np.ix_(node_idx, node_idx)]
                re_feat_0 = features_0[node_idx, :]
                re_adj_0 = re_adj_0 + scipy.sparse.identity(re_adj_0.shape[0])
                re_labels_0 = one_hot(labels_0, num_class)[node_idx, :]
                diag_val = gntk.diag(re_feat_0, re_adj_0)
                _, sigma_val_syn, ntk_val_syn, dotsigma_val_syn = calc(gntk, re_feat_0, feat_syn, diag_val, diag_syn,
                                                                       re_adj_0,
                                                                       adj_syn)
                score_syn, acc_syn = loss_acc_fn_eval(data, ntk_syn_syn[-1], ntk_val_syn[-1], labels_syn, re_labels_0,
                                                      reg=args.ntk_reg)

                score_syn_ls.append(score_syn.item())
                acc_syn_ls.append(acc_syn.item())
                logging.info(
                    'The graph ntk KRR score within the {}-th sampling in validation score = {:.4f}, acc = {:.2f}'.format(
                        k,
                        score_syn,
                        acc_syn * 100.))
            score_syn_np = np.array(score_syn_ls)
            acc_syn_np = np.array(acc_syn_ls)

            score_syn_mean = np.mean(score_syn_np)
            acc_syn_mean = np.mean(acc_syn_np)
            logging.info('AVG KRR score = {:.4f}, acc = {:.2f}'.format(
                score_syn_mean,
                acc_syn_mean * 100.))

        else:
            adj_0 = adj_0 + scipy.sparse.identity(adj_0.shape[0])
            labels_0 = one_hot(labels_0, num_class)
            diag_val = gntk.diag(features_0, adj_0)
            _, sigma_val_syn, ntk_val_syn, dotsigma_val_syn = calc(gntk, features_0, feat_syn, diag_val, diag_syn,
                                                                   adj_0,
                                                                   adj_syn)
            score_syn, acc_syn = loss_acc_fn_eval(data, ntk_syn_syn[-1], ntk_val_syn[-1], labels_syn, labels_0,
                                                  reg=args.ntk_reg)

            logging.info(
                'The graph ntk KRR score within in validation score = {:.4f}, acc = {:.2f}'.format(
                    score_syn,
                    acc_syn * 100.))

            score_syn_mean = score_syn
            acc_syn_mean = acc_syn

        return score_syn_mean, acc_syn_mean, feat_syn_eval, adj_syn_eval, labels_syn_eval

    def distill(self, writer):

        args = self.args
        data = self.data

        #features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        feat_test, adj_test, labels_test = data.feat_test, data.adj_test, data.labels_test
        feat_init, adj_init, labels_init = self.get_coreset_init(features, adj, labels)
        feat_init, adj_init, labels_init = utils.to_tensor(feat_init, adj_init, labels_init, device=self.device)

        feat_test, adj_test, labels_test = utils.to_tensor(feat_test, adj_test, labels_test, device=self.device)
        #features_tensor, adj_tensor, labels_tensor = utils.to_tensor(features, adj, labels, device=self.device)
        #adj_tensor_norm = utils.normalize_adj_tensor(adj_tensor, sparse=True)

        self.feat_syn.data.copy_(feat_init)
        self.labels_syn = labels_init
        self.adj_syn_init = adj_init

        file_idx, expert_idx, expert_files = self.expert_load()

        syn_lr = torch.tensor(args.lr_student).to(self.device)

        if args.optim_lr == 1:
            syn_lr = syn_lr.detach().to(self.device).requires_grad_(True)
            if args.optimizer_lr == 'Adam':
                optimizer_lr = torch.optim.Adam([syn_lr], lr=args.lr_lr)
            elif args.optimizer_lr == 'SGD':
                optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

        eval_it_pool = np.arange(0, args.ITER + 1, args.eval_interval).tolist()
        model_eval_pool = get_eval_pool(args.eval_type, args.condense_model, args.eval_model)
        accs_all_exps = dict()  # record performances of all experiments
        for key in model_eval_pool:
            accs_all_exps[key] = []

        best_ntk_score_eval = {m: 10 for m in model_eval_pool}
        best_ntk_score_eval_iter = {m: 0 for m in model_eval_pool}
        best_ntk_accs_eval = {m: 0 for m in model_eval_pool}
        best_ntk_accs_eval_iter = {m: 0 for m in model_eval_pool}

        best_model_acc_eval = {m: 0 for m in model_eval_pool}
        best_model_acc_eval_iter = {m: 0 for m in model_eval_pool}
        best_model_std_eval = {m: 0 for m in model_eval_pool}

        best_model_acc_test = {m: 0 for m in model_eval_pool}
        best_model_acc_test_iter = {m: 0 for m in model_eval_pool}
        best_model_std_test = {m: 0 for m in model_eval_pool}

        best_loss = 1.0
        best_loss_it = 0
        adj_syn_norm_key = {'0': 0}

        for it in tqdm(range(0, args.ITER + 1), desc='Distill'):
            #logging.info(adj_syn_norm_key['0'])
            if args.dataset in ['ogbn-arxiv']:
                #model = SGC1(nfeat=feat_syn.shape[1], nhid=self.args.student_hidden,
                #             dropout=0.0, with_bn=False,
                #             weight_decay=0e-4, nlayers=2,
                #             nclass=data.nclass,
                #             device=self.device).to(self.device)
                model = GCN(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                            nclass=data.nclass, dropout=args.student_dropout, nlayers=args.student_nlayers,
                            device=self.device).to(self.device)
            else:
                if args.condense_model == 'SGC':
                    model = SGC(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                                nclass=data.nclass, dropout=args.student_dropout,
                                nlayers=args.student_nlayers, with_bn=False,
                                device=self.device).to(self.device)
                elif args.condense_model == 'GCN':
                    model = GCN(nfeat=data.feat_train.shape[1], nhid=args.student_hidden,
                                nclass=data.nclass, dropout=args.student_dropout, nlayers=args.student_nlayers,
                                device=self.device).to(self.device)
            # model.initialize()

            model = ReparamModule(model)

            model.train()

            num_params = sum([np.prod(p.size()) for p in (model.parameters())])

            if args.load_all:
                expert_trajectory = self.buffer[np.random.randint(0, len(self.buffer))]
            else:
                expert_trajectory = self.buffer[expert_idx]
                expert_idx += 1
                if expert_idx == len(self.buffer):
                    expert_idx = 0
                    file_idx += 1
                    if file_idx == len(expert_files):
                        file_idx = 0
                        random.shuffle(expert_files)
                    # print("loading file {}".format(expert_files[file_idx]))
                    if args.max_files != 1:
                        del self.buffer
                        self.buffer = torch.load(expert_files[file_idx])
                    if args.max_experts is not None:
                        self.buffer = self.buffer[:args.max_experts]
                    random.shuffle(self.buffer)

            if args.rand_start == 1:
                if args.interval_buffer == 1:
                    start = np.linspace(0, args.start_epoch, num=args.start_epoch // 10 + 1)
                    start_epoch = int(np.random.choice(start, 1)[0])
                    start_epoch = start_epoch//10
                else:
                    start_epoch = np.random.randint(0, args.start_epoch)
            elif args.rand_start == 0:
                if args.interval_buffer == 1:
                    start_epoch = args.start_epoch // 10
                else:
                    start_epoch = args.start_epoch

            starting_params = expert_trajectory[start_epoch]

            if args.interval_buffer == 1:
                #print(start_epoch + args.expert_epochs // 10, len(expert_trajectory))
                target_params = expert_trajectory[start_epoch + args.expert_epochs // 10]
            else:
                target_params = expert_trajectory[start_epoch + args.expert_epochs]

            target_params = torch.cat([p.data.to(self.device).reshape(-1) for p in target_params], 0)

            student_params = [
                torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params], 0)

            param_loss_list = []
            param_dist_list = []
            #adj_syn_norm_list = []

            logging.info('it:{}--feat_max = {:.4f}, feat_min = {:.4f}'.format(it, torch.max(self.feat_syn),
                                                                              torch.min(self.feat_syn)))

            if it == 0:
                feat_syn = self.feat_syn
                adj_syn_norm = utils.normalize_adj_tensor(self.adj_syn_init, sparse=True)
                adj_syn_input = adj_syn_norm
            else:
                feat_syn = self.feat_syn
                adj_syn = torch.eye(feat_syn.shape[0]).to(self.device)
                adj_syn_cal_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                adj_syn_input = adj_syn_cal_norm
                #feat_knn = F.normalize(self.feat_syn.detach(), p=1).to(self.device)
                #adj_syn_cal = torch.from_numpy(self.nearest_neighbors(feat_knn, args.k)).to(self.device)
                #adj_syn_mix = adj_syn_cal + self.adj_syn_init
                #adj_syn_mix[torch.where(adj_syn_mix.gt(1))] = 1
                #adj_syn_cal_norm = utils.normalize_adj_tensor(adj_syn_mix, sparse=False)
                #adj_syn_input = adj_syn_cal_norm

            for step in range(args.syn_steps):
                forward_params = student_params[-1]
                feat_out, output_syn = model.forward(feat_syn, adj_syn_input, flat_param=forward_params)
                loss_syn = F.nll_loss(output_syn, self.labels_syn)
                grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)[0]
                acc_syn = utils.accuracy(output_syn, self.labels_syn)
                student_params.append(student_params[-1] - syn_lr * grad)
                if step % 500 == 0:
                    model.eval()
                    #feat_test, adj_test, labels_test = utils.to_tensor(feat_test, adj_test, labels_test,
                    #                                                   device=self.device)
                    adj_test_norm = utils.normalize_adj_tensor(adj_test, sparse=True)
                    _, output_test = model.forward(feat_test, adj_test_norm, flat_param=student_params[-1])
                    acc_test = utils.accuracy(output_test, labels_test)
                    logging.info('loss = {:.4f},acc_syn = {:.4f},acc_test = {:.4f}'.format(loss_syn.item(),
                                                                                           acc_syn.item(),
                                                                                           acc_test.item()))
                    model.train()

            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)

            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            grand_loss = param_loss
            # total_loss = grand_loss + ntk_loss
            total_loss = grand_loss
            self.optimizer_feat.zero_grad()

            if args.optim_lr == 1:
                optimizer_lr.zero_grad()

            # grand_loss.backward()
            total_loss.backward()
            # print(torch.min(self.feat_syn), torch.max(self.feat_syn), torch.min(self.feat_syn.grad), torch.max(self.feat_syn.grad))
            self.optimizer_feat.step()
            logging.info('torch.sum(self.feat_syn) = {}'.format(torch.sum(self.feat_syn)))
            if args.optim_lr == 1:
                optimizer_lr.step()
                writer.add_scalar('student_lr_change', syn_lr.item(), it)

            if it % 1 == 0:
                logging.info(
                    "Iteration {}: Total_Loss = {:.4f}, Grand_Loss={:.4f}, Start_Epoch= {}, Student_LR = {:6f}".format(
                        it,
                        total_loss.item(),
                        grand_loss.item(),
                        start_epoch,
                        syn_lr.item()))


            if it in eval_it_pool and it > 0:
                for model_eval in model_eval_pool:
                    logging.info(
                        'Evaluation: model_train = {}, model_eval = {}, iteration = {}'.format(args.condense_model,
                                                                                               model_eval,
                                                                                               it))
                    ntk_score_eval = []
                    ntk_accs_eval = []


                    # for it_eval in range(args.eval_nums):
                    ntk_score_eval_o, ntk_acc_test_o, ntk_feat_syn_save, ntk_adj_syn_save, ntk_label_syn_save = self.evaluate_synset_ntk()
                    ntk_score_eval.append(ntk_score_eval_o)
                    ntk_accs_eval.append(ntk_acc_test_o)

                    logging.info(
                        'This is learned adj_syn INFO with {}-th iters: Shape: {}, Sum: {}, Avg_value: {}, Sparsity :{}'
                        .format(it, ntk_adj_syn_save.shape, ntk_adj_syn_save.sum(),
                                ntk_adj_syn_save.sum() / (ntk_adj_syn_save.shape[0] ** 2),
                                ntk_adj_syn_save.nonzero().shape[0] / (ntk_adj_syn_save.shape[0] ** 2)))

                    ntk_score_eval = np.array(ntk_score_eval)
                    ntk_accs_eval = np.array(ntk_accs_eval)

                    ntk_accs_eval_mean = np.mean(ntk_accs_eval)
                    # acc_test_std = np.std(accs_test)
                    writer.add_scalar('ntk_acc_eval_curve', ntk_accs_eval_mean, it)

                    ntk_score_eval_mean = np.mean(ntk_score_eval)
                    # score_test_std = np.std(score_test)
                    writer.add_scalar('ntk_score_test_curve', ntk_score_eval_mean, it)

                    if ntk_score_eval_mean < best_ntk_score_eval[model_eval]:
                        best_ntk_score_eval[model_eval] = ntk_score_eval_mean
                        best_ntk_score_eval_iter[model_eval] = it
                        torch.save(ntk_adj_syn_save,
                                   f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_best_ntk_score_{args.seed_student}.pt')
                        torch.save(ntk_feat_syn_save,
                                   f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_best_ntk_score_{args.seed_student}.pt')
                        torch.save(ntk_label_syn_save,
                                   f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_best_ntk_score_{args.seed_student}.pt')

                    if ntk_accs_eval_mean > best_ntk_accs_eval[model_eval]:
                        best_ntk_accs_eval[model_eval] = ntk_accs_eval_mean
                        best_ntk_accs_eval_iter[model_eval] = it
                        torch.save(ntk_adj_syn_save,
                                   f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_best_ntk_acc_{args.seed_student}.pt')
                        torch.save(ntk_feat_syn_save,
                                   f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_best_ntk_acc_{args.seed_student}.pt')
                        torch.save(ntk_label_syn_save,
                                   f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_best_ntk_acc_{args.seed_student}.pt')

                    logging.info(
                        'Evaluate ntk {}, score_mean = {:.4f}, acc_mean = {:.2f}'.format(model_eval, ntk_score_eval_mean, ntk_accs_eval_mean * 100.0))

            '''
            if it in eval_it_pool and it > 0:
                for model_eval in model_eval_pool:
                    logging.info(
                        'Evaluation: model_train = {}, model_eval = {}, iteration = {}'.format(args.condense_model,
                                                                                               model_eval,
                                                                                               it))
                    accs_test = []
                    accs_val = []

                    for it_eval in range(args.eval_nums):
                        args.eval_lr = syn_lr.item()
                        out_acc_val, out_acc_test, feat_syn_save, adj_syn_save, label_syn_save = self.evaluate_synset_knn(model_eval)
                        accs_test.append(out_acc_test)
                        accs_val.append(out_acc_val)

                    logging.info(
                        'This is learned adj_syn INFO with {}-th iters: Shape: {}, Sum: {}, Avg_value: {}, Sparsity :{}'
                        .format(it, adj_syn_save.shape, adj_syn_save.sum(),
                                adj_syn_save.sum() / (adj_syn_save.shape[0] ** 2),
                                adj_syn_save.nonzero().shape[0] / (adj_syn_save.shape[0] ** 2)))

                    accs_test = np.array(accs_test)
                    accs_val = np.array(accs_val)

                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    writer.add_scalar('acc_test_curve', acc_test_mean, it)

                    acc_val_mean = np.mean(accs_val)
                    acc_val_std = np.std(accs_val)
                    writer.add_scalar('acc_val_curve', acc_val_mean, it)

                    if acc_val_mean > best_model_acc_eval[model_eval]:
                        best_model_acc_eval[model_eval] = acc_val_mean
                        best_model_std_eval[model_eval] = acc_val_std
                        best_model_acc_eval_iter[model_eval] = it
                        torch.save(adj_syn_save,
                                   f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_best_val_{args.seed_student}.pt')
                        torch.save(feat_syn_save,
                                   f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_best_val_{args.seed_student}.pt')
                        torch.save(label_syn_save,
                                   f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_best_val_{args.seed_student}.pt')

                    if acc_test_mean > best_model_acc_test[model_eval]:
                        best_model_acc_test[model_eval] = acc_test_mean
                        best_model_std_test[model_eval] = acc_test_std
                        best_model_acc_test_iter[model_eval] = it
                        torch.save(adj_syn_save,
                                   f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_best_test_{args.seed_student}.pt')
                        torch.save(feat_syn_save,
                                   f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_best_test_{args.seed_student}.pt')
                        torch.save(label_syn_save,
                                   f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_best_test_{args.seed_student}.pt')

                    logging.info(
                        'Evaluate {} random {}, val_mean = {:.2f}, val_std = {:.2f}, test_mean = {:.2f}, test_std={:.2f}'.format(
                            len(accs_test), model_eval, acc_val_mean * 100.0, acc_val_std * 100.0,
                                                        acc_test_mean * 100.0, acc_test_std * 100.0))

                    if it % 1000 == 0:
                        torch.save(adj_syn_save,
                                   f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}.pt')
                        torch.save(feat_syn_save,
                                   f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}.pt')
                        torch.save(label_syn_save,
                                   f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}.pt')
            
            '''
            if it % 1000 == 0 or it==args.ITER:
                feat_syn_save, adj_syn_save, label_syn_save = self.synset_save()
                torch.save(adj_syn_save,
                           f'{args.log_dir}/adj_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}.pt')
                torch.save(feat_syn_save,
                           f'{args.log_dir}/feat_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}.pt')
                torch.save(label_syn_save,
                           f'{args.log_dir}/label_{args.dataset}_{args.reduction_rate}_{it}_{args.seed_student}.pt')
            for _ in student_params:
                del _

            if grand_loss.item() < best_loss:
                best_loss = grand_loss.item()
                best_loss_it = it

            writer.add_scalar('grand_loss_curve', grand_loss.item(), it)


        for model_eval in model_eval_pool:
            logging.info('Evaluation NTK: {} for eval_ntk_score_best: Score_ntk={:.5f}, within {} iter, ACC_ntk = {:.2f}, withtin {} iter'.format(
                model_eval, best_ntk_score_eval[model_eval], best_ntk_score_eval_iter[model_eval],best_ntk_accs_eval[model_eval], best_ntk_accs_eval_iter[model_eval]))
        
        #    logging.info(
        #        'Evaluation MODEL: {} for_eval/test_acc_best: ACC_eval={:.2f}, within {} iter, ACC_test={:.2f}, within {} iter'.format(
        #            model_eval, best_model_acc_eval[model_eval] * 100.0, best_model_acc_eval_iter[model_eval],best_model_acc_test[model_eval] * 100.0, best_model_acc_test_iter[model_eval]))

        logging.info('This is the smallest loss: {:.06f} within {} iteration'.format(best_loss, best_loss_it))

    def get_coreset_init(self, features, adj, labels):
        logging.info('Loading from: {}'.format(self.args.coreset_init_path))
        idx_selected_train = np.load(
            f'{self.args.coreset_init_path}/idx_{self.args.dataset}_{self.args.reduction_rate}_{self.args.coreset_method}_{self.args.coreset_seed}.npy')
        feat_train = features[idx_selected_train]
        adj_train = adj[np.ix_(idx_selected_train, idx_selected_train)]
        labels_train = labels[idx_selected_train]
        return feat_train, adj_train, labels_train

    def nearest_neighbors(self, features, k=2):
        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        # print(torch.isnan(features+1e-10).any())
        # print(torch.isinf(features+1e-10).any())
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
            sims[i, indices_argsort[-k:]] = 1
        adj_knn = torch.FloatTensor(sims).to(self.device)
        diag = torch.diag(adj_knn)
        a_diag = torch.diag_embed(diag)
        adj_knn = adj_knn - a_diag
        adj_knn = adj_knn.cpu().numpy()
        #adj_knn += np.eye(adj_knn.shape[0])
        return adj_knn


def one_hot(x,
            num_classes,
            center=True,
            dtype=np.float32):
    assert len(x.shape) == 1
    one_hot_vectors = np.array(x[:, None] == np.arange(num_classes), dtype)
    if center:
        one_hot_vectors = one_hot_vectors - 1. / num_classes
    return one_hot_vectors


def calc(gntk, feat1, feat2, diag1, diag2, A1, A2):
    return gntk.gntk(feat1, feat2, diag1, diag2, A1, A2)


def loss_acc_fn_train(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
    # print(k_ss.device, torch.abs(torch.tensor(reg)).to(k_ss.device),torch.trace(k_ss).device, torch.eye(k_ss.shape[0]).device)
    k_ss_reg = (k_ss + torch.abs(torch.tensor(reg)).to(k_ss.device) * torch.trace(k_ss).to(k_ss.device) * torch.eye(
        k_ss.shape[0]).to(k_ss.device) / k_ss.shape[0])
    # print(torch.linalg.inv(k_ss_reg).dtype,torch.from_numpy(y_support).dtype,k_ts[data.idx_train, :].dtype )
    # print(torch.linalg.inv(k_ss_reg).shape, torch.from_numpy(y_support).shape, k_ts[data.idx_train, :].shape)
    pred = torch.matmul(k_ts[data.idx_train, :].cuda(), torch.matmul(torch.linalg.inv(k_ss_reg).cuda(),
                                                                     torch.from_numpy(y_support).to(
                                                                         torch.float64).cuda()))
    mse_loss = torch.nn.functional.mse_loss(pred.to(torch.float64).cuda(),
                                            torch.from_numpy(y_target).to(torch.float64).cuda(), reduction="mean")
    # mse_loss = 0.5 * np.mean((pred - y_target) ** 2)
    # print(mse_loss)
    # acc = torch.argmax(pred, dim=1).eq(torch.argmax(y_target, dim=1))
    acc = 0
    return mse_loss, acc


def loss_acc_fn_eval(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
    k_ss_reg = (k_ss + np.abs(reg) * np.trace(k_ss) * np.eye(k_ss.shape[0]) / k_ss.shape[0])
    pred = np.dot(k_ts, np.linalg.inv(k_ss_reg).dot(y_support))
    mse_loss = 0.5 * np.mean((pred - y_target) ** 2)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_target, axis=1))
    return mse_loss, acc
