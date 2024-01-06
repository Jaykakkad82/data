#@reddit-r0001
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=0.05 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=50 --syn_steps=900 \
--buffer_path './logs/Buffer/reddit-20231226-211606-330033' \
--coreset_init_path './logs/Coreset/reddit-reduce_0.001-20231231-010054-261123' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg=0.01 --eval_interval 1 --ITER 3000 --samp_iter 1 --samp_num_per_class 50 \
--seed_student 31 --coreset_seed 31

# #@reddit-r0002
# CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
# --lr_feat=${lr_feat} --optimizer_con Adam \
# --expert_epochs=${exp_e} --lr_student=${lr_stu} --optim_lr=0 \
# --start_epoch=${sta_e} --syn_steps=${syn_e} \
# --buffer_path './logs/Buffer/reddit-20231226-211606-330033' \
# --coreset_init_path './logs/Coreset/reddit-reduce_0.002-20230203-205724-419628' \
# --condense_model GCN --interval_buffer 1 --rand_start 1 \
# --reduction_rate=0.002 --ntk_reg=0.01 --eval_interval 5 --ITER 2000 --samp_iter 1 --samp_num_per_class 90 \
# --seed_student 31 --coreset_seed 31

#@reddit-r0005
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=0.2 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.2 --optim_lr=0 \
--start_epoch=30 --syn_steps=900 \
--buffer_path './logs/Buffer/reddit-20231226-211606-330033' \
--coreset_init_path './logs/Coreset/reddit-reduce_0.005-20231231-012048-753848' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.005 --ntk_reg=0.0001 --eval_interval 1 --ITER 1500 --samp_iter 1 --samp_num_per_class=50 \
--seed_student 31 --coreset_seed 31

#@reddit-r00005
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=0.02 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=900 \
--buffer_path './logs/Buffer/reddit-20231226-211606-330033' \
--coreset_init_path './logs/Coreset/reddit-reduce_0.0005-20231231-022500-982742' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.0005 --ntk_reg 0.01 --eval_interval 1 --ITER 3000 --samp_iter 5 --samp_num_per_class 50 \
--seed_student 31 --coreset_seed 31

