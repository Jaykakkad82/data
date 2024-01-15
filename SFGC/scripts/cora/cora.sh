#! /bin/bash

#@cora-buff
CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 0.4 --teacher_epochs 2500 --dataset cora --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' --num_experts=200 --wd_teacher 0 --mom_teacher 0 --optim SGD --seed_teacher 5


#@cora-r025
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1200 --lr 0.01 --lr_coreset 0.01 --weight_decay 5e-4 --wd_coreset 1e-4  --save 1 --method kcenter --reduction_rate 0.25 --load_npy '' --seed 5

#@cora-r05
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1000 --lr 0.01 --lr_coreset 0.005 --weight_decay 5e-4 --wd_coreset 5e-4 --save 1 --method kcenter --reduction_rate 0.5 --seed 5

#@cora-r0125
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 600 --lr 0.01 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.125 --load_npy '' --seed 5


#@cora-r0125
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam --expert_epochs=2000 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=500 --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.125 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student 5 --coreset_seed 5

#@cora-r025
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam --expert_epochs=1500 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=50 --syn_steps=400 --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.25 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student 5 --coreset_seed 5

#@cora-r05
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam --expert_epochs=1200 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=500 --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.5 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student 5 --coreset_seed 5


#@cora-r0125
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset cora \
--reduction_rate 0.125 \
--test_lr_model=0.01 --test_wd=0.0005 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--tr_seed=5 --seed=5

#@cora-r025
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset cora \
--reduction_rate 0.25 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--tr_seed=5 --seed=5

#@cora-r05
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset cora \
--reduction_rate 0.5 \
--test_lr_model=0.001 --test_wd=0.001 --test_model_iters 600 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--tr_seed=5 --seed=5