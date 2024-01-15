#! /bin/bash

#@flickr-buff
CUDA_VISIBLE_DEVICES=0 python buffer_inductive.py --device cuda:0 --lr_teacher 0.001 --teacher_epochs 1000 --dataset flickr --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' --num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0 --seed_teacher 31 > ./logs/flickr_buffer_inductive

#@flickr-r0001
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 1000 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.001 --load_npy '' --seed 31 > ./logs/flickr_train_coreset_inductive_0001
#./logs/Coreset/flickr-reduce_0.001-20221109-095434-604761

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 1 --method kcenter --reduction_rate 0.001 \
#--load_npy './logs/Coreset/flickr-reduce_0.001-20221109-095434-604761' --runs 1

#@flickr-r0005
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.005 --load_npy '' --seed 31 > ./logs/flickr_train_coreset_inductive_0005
#./logs/Coreset/flickr-reduce_0.005-20221109-101214-051861

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 1 --method kcenter --reduction_rate 0.005 \
#--load_npy './logs/Coreset/flickr-reduce_0.005-20221109-101214-051861' --runs 1

#@flickr-r001
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.01 --load_npy '' --seed 31 > ./logs/flickr_train_coreset_inductive_001
#./logs/Coreset/flickr-reduce_0.01-20221109-101537-112764

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 1 --method kcenter --reduction_rate 0.01 \
#--load_npy './logs/Coreset/flickr-reduce_0.01-20221109-101537-112764' --runs 1

#========================
