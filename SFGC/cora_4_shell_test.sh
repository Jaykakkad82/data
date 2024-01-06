#! /bin/bash

#@cora-r0125
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset cora \
--reduction_rate 0.125 \
--test_lr_model=0.01 --test_wd=0.0005 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/cora-reduce_0.125-20231228-014922-844592' \
--tr_seed=61 --seed=61

#@cora-r025
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset cora \
--reduction_rate 0.25 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/cora-reduce_0.25-20231227-211224-227281' \
--tr_seed=61 --seed=61

#@cora-r05
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset cora \
--reduction_rate 0.5 \
--test_lr_model=0.001 --test_wd=0.001 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/cora-reduce_0.5-20231227-090006-653478' \
--tr_seed=61 --seed=61