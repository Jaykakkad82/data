#! /bin/bash

reduction_rate=0.25
noise_type="shuffle_nodes"
dataset="cora"
seed=31
noise_list=(0.1 0.15 0.2)
uid_list=(1016 1017 1018)
for i in {0..2}
do
    noise="${noise_list[$i]}"
    uid="${uid_list[$i]}"

    CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 0.4 --teacher_epochs 2500 --dataset $dataset --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' --num_experts=200 --wd_teacher 0 --mom_teacher 0 --optim SGD --seed_teacher $seed --uid=$uid --noise-type=$noise_type --noise=$noise

    echo "---------------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset $dataset --device cuda:0 --epochs 1200 --lr 0.01 --lr_coreset 0.01 --weight_decay 5e-4 --wd_coreset 1e-4  --save 1 --method kcenter --reduction_rate $reduction_rate --load_npy '' --seed $seed --uid=$uid --noise-type=$noise_type --noise=$noise

    echo "---------------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity.py --dataset $dataset --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam --expert_epochs=1500 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=50 --syn_steps=400 --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=$reduction_rate --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student $seed --coreset_seed $seed --uid=$uid --noise-type=$noise_type --noise=$noise

    echo "---------------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset $dataset \
    --reduction_rate $reduction_rate \
    --test_lr_model=0.01 --test_wd=0.0005 --test_model_iters 600 --nruns 1  --test_dropout=0 \
    --best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
    --tr_seed=$seed --seed=$seed \
    --uid=$uid --noise-type=$noise_type --noise=$noise

    echo "---------------------------------------------------------------------"
    echo "---------------------------------------------------------------------"
done

