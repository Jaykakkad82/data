#! /bin/bash

start="$1"
end="$2"
echo "Start: "$start "end: "$end

reduction_rate=0.25
noise_type="shuffle_nodes"
dataset_list=("citeseer" "citeseer" "citeseer" "citeseer" "citeseer" "citeseer" "citeseer" "citeseer" "citeseer" "cora" "cora" "cora" "cora" "cora" "cora" "cora" "cora" "cora")
seed_list=(37 37 37 15 15 15 31 31 31 37 37 37 15 15 15 31 31 31)
noise_list=(0.1 0.15 0.2 0.1 0.15 0.2 0.1 0.15 0.2 0.1 0.15 0.2 0.1 0.15 0.2 0.1 0.15 0.2)
uid_list=(1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058)
for ((i=$start; i<=$end; i++));
do
    dataset="${dataset_list[$i]}"
    seed="${seed_list[$i]}"
    noise="${noise_list[$i]}"
    uid="${uid_list[$i]}"

    echo "reduction_rate: "$reduction_rate "noise_type: "$noise_type "dataset: "$dataset "seed: "$seed "noise: "$noise "uid: "$uid

    CUDA_VISIBLE_DEVICES=0 python buffer_transduct.py --device cuda:0 --lr_teacher 0.001 --teacher_epochs 800 --dataset $dataset --teacher_nlayers=2 --traj_save_interval=10 --param_save_interval=10 --buffer_model_type 'GCN' --num_experts=200 --wd_teacher 5e-4 --mom_teacher 0 --optim Adam --decay 0 --seed_teacher $seed --uid=$uid --noise-type=$noise_type --noise=$noise

    echo "---------------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset $dataset --device cuda:0 --epochs 800 --lr 0.001 --weight_decay 5e-4  --save 1 --method kcenter --reduction_rate $reduction_rate --load_npy '' --seed $seed --uid=$uid --noise-type=$noise_type --noise=$noise

    echo "---------------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity2.py --dataset $dataset --device cuda:0 --lr_feat=0.0005 --optimizer_con Adam --expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=200 --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=$reduction_rate --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student $seed --coreset_seed $seed --uid=$uid --noise-type=$noise_type --noise=$noise

    echo "---------------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 python test_other_arcs.py --device cuda:0 --dataset $dataset \
    --reduction_rate $reduction_rate \
    --test_lr_model=0.003 --test_wd=0.005 --test_model_iters 1000 --nruns 1  --test_dropout=0.3 \
    --best_ntk_score 1 --test_opt_type Adam --test_model_type GCN \
    --tr_seed=$seed --seed=$seed \
    --uid=$uid --noise-type=$noise_type --noise=$noise

    echo "---------------------------------------------------------------------"
    echo "---------------------------------------------------------------------"
done

# sh scripts/noise/shuffle-exp.sh 0 2
# sh scripts/noise/shuffle-exp.sh 3 5
# sh scripts/noise/shuffle-exp.sh 6 8
# sh scripts/noise/shuffle-exp.sh 9 11
# sh scripts/noise/shuffle-exp.sh 12 14
# sh scripts/noise/shuffle-exp.sh 15 17

# CUDA_VISIBLE_DEVICES=0 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 --lr_feat=0.0005 --optimizer_con Adam --expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 --start_epoch=30 --syn_steps=200 --condense_model GCN --interval_buffer 1 --rand_start 1 --reduction_rate=0.25 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 --seed_student 37 --coreset_seed 37 --uid=1021 --noise-type=shuffle_nodes --noise=0.2