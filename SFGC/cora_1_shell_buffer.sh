#! /bin/bash

#@cora-buff
CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 0.4 --teacher_epochs 2500 --dataset cora --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' --num_experts=200 --wd_teacher 0 --mom_teacher 0 --optim SGD --seed_teacher 5

#@cora-r025
# CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1200 --lr 0.01 --lr_coreset 0.01 --weight_decay 5e-4 --wd_coreset 1e-4  --save 1 --method kcenter --reduction_rate 0.25 --load_npy '' --runs 1 --seed 5 > ./logs/cora_train_coreset_025
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1200 --lr 0.01 --lr_coreset 0.01 --weight_decay 5e-4 --wd_coreset 1e-4  --save 1 --method kcenter --reduction_rate 0.25 --load_npy '' --seed 5

#@cora-r05
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1000 --lr 0.01 --lr_coreset 0.005 --weight_decay 5e-4 --wd_coreset 5e-4 --save 1 --method kcenter --reduction_rate 0.5 --seed 5
#CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 600 --lr_coreset 0.005 \
#--wd_coreset 5e-4 --method kcenter --reduction_rate 0.5 \
#--load_npy 'logs/Coreset/cora-reduce_0.5-20221024-112028-667459'

#@cora-r0125
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 600 --lr 0.01 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.125 --load_npy '' --seed 5

#CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 600 --lr_coreset 0.01 \
#--wd_coreset 5e-4  --save 1 --method kcenter --reduction_rate 1 \
#--load_npy './logs/Coreset/cora-reduce_1.0-20221025-163255-954853' --runs 1