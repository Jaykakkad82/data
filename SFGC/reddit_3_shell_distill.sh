#@reddit-r0001
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=0.05 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=50 --syn_steps=900 \
--buffer_path './logs/Buffer/reddit-20240106-084624-088917' \
--coreset_init_path './logs/Coreset/reddit-reduce_0.001-20240106-091615-932621' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg=0.01 --eval_interval 1 --ITER 3000 --samp_iter 1 --samp_num_per_class 50 \
--seed_student 61 --coreset_seed 61


#@reddit-r0005
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=0.2 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.2 --optim_lr=0 \
--start_epoch=30 --syn_steps=900 \
--buffer_path './logs/Buffer/reddit-20240106-084624-088917' \
--coreset_init_path './logs/Coreset/reddit-reduce_0.005-20240106-093556-005281' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.005 --ntk_reg=0.0001 --eval_interval 1 --ITER 1500 --samp_iter 1 --samp_num_per_class=50 \
--seed_student 61 --coreset_seed 61

#@reddit-r00005
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=0.02 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=900 \
--buffer_path './logs/Buffer/reddit-20240106-084624-088917' \
--coreset_init_path './logs/Coreset/reddit-reduce_0.0005-20240106-100010-672313' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.0005 --ntk_reg 0.01 --eval_interval 1 --ITER 3000 --samp_iter 5 --samp_num_per_class 50 \
--seed_student 61 --coreset_seed 61

