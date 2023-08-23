#@citeseer-r025
CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 \
--lr_feat=0.0005 --optimizer_con Adam \
--expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=200 \
--buffer_path './logs/Buffer/citeseer-20221012-091556-763255' \
--coreset_init_path './logs/Coreset/citeseer-reduce_0.25-20221106-171050-606991' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.25 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10

#@citeseer-r05
CUDA_VISIBLE_DEVICES=3 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 \
--lr_feat=0.001 --optimizer_con Adam \
--expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=200 \
--buffer_path './logs/Buffer/citeseer-20221012-091556-763255' \
--coreset_init_path './logs/Coreset/citeseer-reduce_0.5-20221106-114854-910019' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.5 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10

#@citeseer-r1
CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity2.py --dataset citeseer --device cuda:0 \
--lr_feat=0.001 --optimizer_con Adam \
--expert_epochs=400 --lr_student=0.6 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=300 \
--buffer_path './logs/Buffer/citeseer-20221012-091556-763255' \
--coreset_init_path './logs/Coreset/citeseer-reduce_1.0-20221106-171317-449627' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=1 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10

#@cora-r025
CUDA_VISIBLE_DEVICES=2 python  distill_transduct_adj_identity.py --dataset cora --device cuda:0 \
--lr_feat=0.0001 --optimizer_con Adam \
--expert_epochs=1500 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=50 --syn_steps=400 \
--buffer_path './logs/Buffer/cora-20220925-225653-091173' \
--coreset_init_path './logs/Coreset/cora-reduce_0.25-20221025-155350-254087' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.25 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10

#@cora-r05
CUDA_VISIBLE_DEVICES=3 python distill_transduct_adj_identity.py --dataset cora --device cuda:0 \
--lr_feat=0.0001 --optimizer_con Adam \
--expert_epochs=1200 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=500 \
--buffer_path './logs/Buffer/cora-20220925-225653-091173' \
--coreset_init_path 'logs/Coreset/cora-reduce_0.5-20221024-112028-667459' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.5 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10

#@cora-r1
CUDA_VISIBLE_DEVICES=1 python distill_transduct_adj_identity.py --dataset cora --device cuda:0 \
--lr_feat=0.0001 --optimizer_con Adam \
--expert_epochs=2000 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=500 \
--buffer_path './logs/Buffer/cora-20220925-225653-091173' \
--coreset_init_path './logs/Coreset/cora-reduce_1.0-20221025-163255-954853' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=1 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10

#@ogbn-r0001
CUDA_VISIBLE_DEVICES=2 python distill_transduct_adj_identity2.py --dataset ogbn-arxiv --device cuda:0 \
--lr_feat=0.2 --optimizer_con Adam \
--expert_epochs=1800 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=600 \
--buffer_path './logs/Buffer/ogbn-arxiv-20220925-223423-976629' \
--coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.001-20221025-212029-649373' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg 0.1 --eval_interval 1 --ITER 2000 --samp_iter 1 --samp_num_per_class=50

#@ogbn-r0005
CUDA_VISIBLE_DEVICES=3 python distill_transduct_adj_identity.py --dataset ogbn-arxiv --device cuda:0 \
--lr_feat=0.1 --optimizer_con Adam \
--expert_epochs=1900 --lr_student=0.1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=1200 \
--buffer_path './logs/Buffer/ogbn-arxiv-20220925-223423-976629' \
--coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.005-20221101-133048-529972' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.005 --ntk_reg 0.1 --eval_interval 1 --ITER 2000 --samp_iter 1 --samp_num_per_class=50

#@ogbn-r001
CUDA_VISIBLE_DEVICES=2 python distill_transduct_adj_identity.py --dataset ogbn-arxiv --device cuda:0 \
--lr_feat=0.1 --optimizer_con Adam \
--expert_epochs=1900 --lr_student=0.1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=10 --syn_steps=1000 \
--buffer_path './logs/Buffer/ogbn-arxiv-20220925-223423-976629' \
--coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.01-20221101-133345-976504' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.01 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter 1 --samp_num_per_class=10

#@flickr-r0001
CUDA_VISIBLE_DEVICES=1 python distill_inductive_adj_identity.py --dataset flickr --device cuda:0 \
--lr_feat=0.1 --optimizer_con Adam \
--expert_epochs=700 --lr_student=0.3 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=600 \
--buffer_path './logs/Buffer/flickr-20221108-120636-260182' \
--coreset_init_path './logs/Coreset/flickr-reduce_0.001-20221109-095434-604761' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg 1 --eval_interval 1 --ITER 2000 --samp_iter 1 --samp_num_per_class=10

#@flickr-r0005
CUDA_VISIBLE_DEVICES=3 python distill_inductive_adj_identity.py --dataset flickr --device cuda:0 \
--lr_feat=0.01 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=600 \
--buffer_path './logs/Buffer/flickr-20221108-120636-260182' \
--coreset_init_path './logs/Coreset/flickr-reduce_0.005-20221109-101214-051861' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.005 --ntk_reg 0.01 --eval_interval 1 --ITER 2000 --samp_iter=5 --samp_num_per_class=10

#@flickr-r001
CUDA_VISIBLE_DEVICES=2 python distill_inductive_adj_identity.py --dataset flickr --device cuda:0 \
--lr_feat=0.02 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=900 \
--buffer_path './logs/Buffer/flickr-20221108-120636-260182' \
--coreset_init_path './logs/Coreset/flickr-reduce_0.01-20221109-101537-112764' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.01 --ntk_reg 0.1 --eval_interval 1 --ITER 2000 --samp_iter 5 --samp_num_per_class=10

#@reddit-r00005
CUDA_VISIBLE_DEVICES=1 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=0.02 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=900 \
--buffer_path './logs/Buffer/reddit-20221108-135237-305681' \
--coreset_init_path './logs/Coreset/reddit-reduce_0.0005-20221113-153007-643703' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.0005 --ntk_reg 0.01 --eval_interval 1 --ITER 3000 --samp_iter 5 --samp_num_per_class 50

#@reddit-r0001
CUDA_VISIBLE_DEVICES=1 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=0.05 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=50 --syn_steps=900 \
--buffer_path './logs/Buffer/reddit-20221108-135237-305681' \
--coreset_init_path './logs/Coreset/reddit-reduce_0.005-20221112-121718-604171' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg=0.01 --eval_interval 1 --ITER 3000 --samp_iter 1 --samp_num_per_class 50

#@reddit-r0002
CUDA_VISIBLE_DEVICES=2 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=${lr_feat} --optimizer_con Adam \
--expert_epochs=${exp_e} --lr_student=${lr_stu} --optim_lr=0 \
--start_epoch=${sta_e} --syn_steps=${syn_e} \
--buffer_path './logs/Buffer/reddit-20221108-135237-305681' \
--coreset_init_path './logs/Coreset/reddit-reduce_0.002-20230203-205724-419628' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.002 --ntk_reg=0.01 --eval_interval 5 --ITER 2000 --samp_iter 1 --samp_num_per_class 90

#@reddit-r0005
CUDA_VISIBLE_DEVICES=2 python distill_inductive_adj_identity.py --dataset reddit --device cuda:0 \
--lr_feat=0.2 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.2 --optim_lr=0 \
--start_epoch=30 --syn_steps=900 \
--buffer_path './logs/Buffer/reddit-20221108-135237-305681' \
--coreset_init_path './logs/Coreset/reddit-reduce_0.005-20221112-121718-604171' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.005 --ntk_reg=0.0001 --eval_interval 1 --ITER 1500 --samp_iter 1 --samp_num_per_class=50

