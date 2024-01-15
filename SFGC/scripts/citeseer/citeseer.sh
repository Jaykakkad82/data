#! /bin/bash

seed="$1"
echo "seed: "$seed

#@citeseer-buff
CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 0.001 --teacher_epochs 2500 --dataset citeseer --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' --num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0 --seed_teacher $seed --uid="citeseer_"$seed

#@citeseer-r05
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.5 --load_npy '' --seed $seed --uid="citeseer_"$seed

#@citeseer-r025
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 1200 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.25 --load_npy '' --seed $seed --uid="citeseer_"$seed

#@citeseer-r0125
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.125 --load_npy '' --seed $seed --uid="citeseer_"$seed


#@citeseer-r0125
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity.py --dataset citeseer --device cuda:0 --lr_feat=0.001 --optimizer_con Adam --expert_epochs=400 --lr_student=0.6 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=300 --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.125 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student $seed --coreset_seed $seed --uid="citeseer_"$seed

#@citeseer-r025
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity.py --dataset citeseer --device cuda:0 --lr_feat=0.0005 --optimizer_con Adam --expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=200 --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.25 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student $seed --coreset_seed $seed --uid="citeseer_"$seed

#@citeseer-r05
CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity.py --dataset citeseer --device cuda:0 --lr_feat=0.001 --optimizer_con Adam --expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=200 --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.5 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student $seed --coreset_seed $seed --uid="citeseer_"$seed


#@citeseer-#@citeseer-r0125
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.125 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 1000 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--tr_seed=$seed --seed=$seed \
 --uid="citeseer_"$seed

#@citeseer-r025
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.25 \
--test_lr_model=0.003 --test_wd=0.005 --test_model_iters 1000 --nruns 1  --test_dropout=0.3 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--tr_seed=$seed --seed=$seed \
 --uid="citeseer_"$seed

#@citeseer-r05
CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset citeseer \
--reduction_rate 0.5 \
--test_lr_model=0.001 --test_wd=0.0005 --test_model_iters 1000 --nruns 1  --test_dropout=0 \
--best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
--tr_seed=$seed --seed=$seed \
 --uid="citeseer_"$seed
