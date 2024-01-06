#! /bin/bash

#@reddit-buff
CUDA_VISIBLE_DEVICES=0 python buffer_inductive.py --device cuda:0 --lr_teacher 0.001 --teacher_epochs 1000 --dataset reddit --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' --num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0 --seed_teacher 31 > ./logs/reddit_buffer_inductive

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 0 --method kcenter --reduction_rate 0.0005 \
#--load_npy './logs/Coreset/reddit-reduce_0.0005-20221113-153007-643703' --runs 1

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 0 --method kcenter --reduction_rate 0.001 \
#--load_npy './logs/Coreset/reddit-reduce_0.001-20221113-150848-219047' --runs 1

#@reddit-r0002
# CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 1000 --lr 0.001 --weight_decay 0.0005  --save 1 --method kcenter --reduction_rate 0.002 --load_npy '' --seed 37 > ./logs/reddit_train_coreset_inductive_0002
#./logs/Coreset/reddit-reduce_0.002-20230203-205724-419628


#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 0 --method kcenter --reduction_rate 0.005 \
#--load_npy './logs/Coreset/reddit-reduce_0.005-20221112-121718-604171' --runs 1






#@reddit-r0001
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.001 --load_npy '' --seed 31
#./logs/Coreset/reddit-reduce_0.001-20221113-150848-219047

#@reddit-r0005
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 1000 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.005 --load_npy '' --seed 31
#./logs/Coreset/reddit-reduce_0.005-20221112-121718-604171

#@reddit-r00005
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 1000 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.0005 --load_npy '' --seed 31
#./logs/Coreset/reddit-reduce_0.0005-20221113-153007-643703