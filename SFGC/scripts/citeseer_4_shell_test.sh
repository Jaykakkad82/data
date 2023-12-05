#! /bin/bash

#@citeseer-r025
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.25 \
--test_lr_model=0.003 --test_wd=0.005 --test_model_iters 1000 --nruns 1  --test_dropout=0.3 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_0.25-20221118-154807-807563'

#@citeseer-r05
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.5 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 1000 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_0.5-20230917-054656-180042'

#@citeseer-r1
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 1 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 1000 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--load_path='./logs/Distill/citeseer-reduce_1.0-20221108-104444-053909'

