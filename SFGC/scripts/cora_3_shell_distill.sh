#!/bin/bash

#@cora-r0125
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam --expert_epochs=2000 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=500 --buffer_path './logs/Buffer/cora-20230917-201947-933578' --coreset_init_path './logs/Coreset/cora-reduce_0.125-20230917-224931-192462' --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.125 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student 37 --coreset_seed 37 > ./logs/cora_distill_transduct_adj_identity_0125

#@cora-r025
CUDA_VISIBLE_DEVICES=0 python  distill_transduct_adj_identity.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam --expert_epochs=1500 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=50 --syn_steps=400 --buffer_path './logs/Buffer/cora-20230917-201947-933578' --coreset_init_path './logs/Coreset/cora-reduce_0.25-20230917-220643-568519' --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.25 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student 37 --coreset_seed 37 > ./logs/cora_distill_transduct_adj_identity_025

#@cora-r05
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam --expert_epochs=1200 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=500 --buffer_path './logs/Buffer/cora-20230917-201947-933578' --coreset_init_path 'logs/Coreset/cora-reduce_0.5-20230917-220727-749680' --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.5 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student 37 --coreset_seed 37 > ./logs/cora_distill_transduct_adj_identity_05

