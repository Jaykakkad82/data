#!/bin/bash

#@citeseer-r0125
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 --lr_feat=0.001 --optimizer_con Adam --expert_epochs=400 --lr_student=0.6 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=300 --buffer_path './logs/Buffer/citeseer-20230917-201850-856819' --coreset_init_path './logs/Coreset/citeseer-reduce_0.125-20230917-224708-727330' --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.125 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student 37 --coreset_seed 37 > ./logs/citeseer_distill_transduct_adj_identity2_0125

#@citeseer-r025
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 --lr_feat=0.0005 --optimizer_con Adam --expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=200 --buffer_path './logs/Buffer/citeseer-20230917-201850-856819' --coreset_init_path './logs/Coreset/citeseer-reduce_0.25-20230917-210853-306871' --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.25 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student 37 --coreset_seed 37 > ./logs/citeseer_distill_transduct_adj_identity2_025

#@citeseer-r05
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 --lr_feat=0.001 --optimizer_con Adam --expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=200 --buffer_path './logs/Buffer/citeseer-20230917-201850-856819' --coreset_init_path './logs/Coreset/citeseer-reduce_0.5-20230917-210758-675885' --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.5 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student 37 --coreset_seed 37 > ./logs/citeseer_distill_transduct_adj_identity2_05
