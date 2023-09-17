#! /bin/bash

#@citeseer-r025
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 \
--lr_feat=0.0005 --optimizer_con Adam \
--expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=200 \
--buffer_path './logs/Buffer/citeseer-20221012-091556-763255' \
--coreset_init_path './logs/Coreset/citeseer-reduce_0.25-20221106-171050-606991' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.25 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10

#@citeseer-r05
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 \
--lr_feat=0.001 --optimizer_con Adam \
--expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=200 \
--buffer_path './logs/Buffer/citeseer-20230917-041317-699068' \
--coreset_init_path './logs/Coreset/citeseer-reduce_0.5-20230917-051821-869434' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.5 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10

#@citeseer-r1
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 \
--lr_feat=0.001 --optimizer_con Adam \
--expert_epochs=400 --lr_student=0.6 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=300 \
--buffer_path './logs/Buffer/citeseer-20221012-091556-763255' \
--coreset_init_path './logs/Coreset/citeseer-reduce_1.0-20221106-171317-449627' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=1 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10

