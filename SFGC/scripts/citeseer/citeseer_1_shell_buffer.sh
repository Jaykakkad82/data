#! /bin/bash

#@citeseer-buff
CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 0.001 --teacher_epochs 800 --dataset citeseer --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' --num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0 --seed_teacher 5 --uid=04d95411-efee-4f77-9e2c-cf24123a70d9 --noise-type=edge_type --noise=0.1

#@citeseer-r05
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.5 --load_npy '' --seed 5 --uid=04d95411-efee-4f77-9e2c-cf24123a70d9 --noise-type=edge_type --noise=0.1

#@citeseer-r025
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.25 --load_npy '' --seed 5 --uid=04d95411-efee-4f77-9e2c-cf24123a70d9 --noise-type=edge_type --noise=0.1

#@citeseer-r0125
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.125 --load_npy '' --seed 5 --uid=04d95411-efee-4f77-9e2c-cf24123a70d9 --noise-type=edge_type --noise=0.1
