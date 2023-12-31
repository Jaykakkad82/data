#@flickr-r0001
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --dataset flickr --device cuda:0 \
--lr_feat=0.1 --optimizer_con Adam \
--expert_epochs=700 --lr_student=0.3 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=600 \
--buffer_path './logs/Buffer/flickr-20231002-004933-697855' \
--coreset_init_path './logs/Coreset/flickr-reduce_0.001-20231001-035846-691719' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter 1 --samp_num_per_class=10 \
--seed_student 31 --coreset_seed 31

#@flickr-r0005
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --dataset flickr --device cuda:0 \
--lr_feat=0.01 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=600 \
--buffer_path './logs/Buffer/flickr-20231002-004933-697855' \
--coreset_init_path './logs/Coreset/flickr-reduce_0.005-20231001-040739-318147' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.005 --ntk_reg 0.01 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10 \
--seed_student 31 --coreset_seed 31

#@flickr-r001
CUDA_VISIBLE_DEVICES=0 python distill_inductive_adj_identity.py --dataset flickr --device cuda:0 \
--lr_feat=0.02 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=900 \
--buffer_path './logs/Buffer/flickr-20231002-004933-697855' \
--coreset_init_path './logs/Coreset/flickr-reduce_0.01-20231001-040502-537652' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.01 --ntk_reg 0.1 --eval_interval 1 --ITER 2000 --samp_iter 5 --samp_num_per_class=10 \
--seed_student 31 --coreset_seed 31