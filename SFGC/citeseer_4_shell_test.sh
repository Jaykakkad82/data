#! /bin/bash

#@citeseer-#@citeseer-r0125
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.125 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 1000 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_0.125-20231226-182346-939479' \
--tr_seed=5 --seed=5 \
 --uid=04d95411-efee-4f77-9e2c-cf24123a70d9 --noise-type=edge_type --noise=0.1

#@citeseer-r025
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.25 \
--test_lr_model=0.003 --test_wd=0.005 --test_model_iters 1000 --nruns 1  --test_dropout=0.3 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_0.25-20231226-204027-725505' \
--tr_seed=5 --seed=5 \
 --uid=04d95411-efee-4f77-9e2c-cf24123a70d9 --noise-type=edge_type --noise=0.1

#@citeseer-r05
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.5 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 1000 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_0.5-20231226-213000-886503' \
--tr_seed=5 --seed=5 \
 --uid=04d95411-efee-4f77-9e2c-cf24123a70d9 --noise-type=edge_type --noise=0.1
