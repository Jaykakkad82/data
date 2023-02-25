for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=2 python distill_transduct.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=0.2 --optimizer_con Adam \
  --expert_epochs=1800 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=30 --syn_steps=600 \
  --buffer_path './logs/Buffer/used/ogbn-arxiv-20220925-223423-976629' \
  --coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.001-20221025-212029-649373' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k} \
  --reduction_rate=0.001 --ntk_reg 0.1 --eval_interval 10 --ITER 200 --samp_iter 1 --samp_num_per_class 50
done
CUDA_VISIBLE_DEVICES=2 python distill_transduct_continue.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=0.2 --optimizer_con Adam \
--expert_epochs=1800 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=600 \
--buffer_path './logs/Buffer/used/ogbn-arxiv-20220925-223423-976629' \
--coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.001-20221025-212029-649373' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg 0.1 --eval_interval 10 --ITER 200 --samp_iter 1 --samp_num_per_class 50

CUDA_VISIBLE_DEVICES=2 python distill_transduct_pge.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=0.2 --optimizer_con Adam \
--expert_epochs=1800 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=600 \
--buffer_path './logs/Buffer/used/ogbn-arxiv-20220925-223423-976629' \
--coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.001-20221025-212029-649373' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg 0.1 --eval_interval 10 --ITER 200 --samp_iter 1 --samp_num_per_class 50

for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=2 python distill_transduct.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=0.1 --optimizer_con Adam \
  --expert_epochs=1900 --lr_student=0.1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=30 --syn_steps=1200 \
  --buffer_path './logs/Buffer/used/ogbn-arxiv-20220925-223423-976629' \
  --coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.005-20221101-133048-529972' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k} \
  --reduction_rate=0.005 --ntk_reg 0.1 --eval_interval 10 --ITER 200 --samp_iter 1 --samp_num_per_class 50
done
CUDA_VISIBLE_DEVICES=2 python distill_transduct_continue.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=0.1 --optimizer_con Adam \
--expert_epochs=1900 --lr_student=0.1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=1200 \
--buffer_path './logs/Buffer/used/ogbn-arxiv-20220925-223423-976629' \
--coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.005-20221101-133048-529972' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.005 --ntk_reg 0.1 --eval_interval 10 --ITER 200 --samp_iter 1 --samp_num_per_class 50

CUDA_VISIBLE_DEVICES=2 python distill_transduct_pge.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=0.1 --optimizer_con Adam \
--expert_epochs=1900 --lr_student=0.1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=1200 \
--buffer_path './logs/Buffer/used/ogbn-arxiv-20220925-223423-976629' \
--coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.005-20221101-133048-529972' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.005 --ntk_reg 0.1 --eval_interval 10 --ITER 200 --samp_iter 1 --samp_num_per_class 50

for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=2 python distill_transduct_1.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=0.1 --optimizer_con Adam \
  --expert_epochs=1900 --lr_student=0.1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=10 --syn_steps=1000 \
  --buffer_path './logs/Buffer/used/ogbn-arxiv-20220925-223423-976629' \
  --coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.01-20221101-133345-976504' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k} \
  --reduction_rate=0.01 --ntk_reg 0.1 --eval_interval 10 --ITER 200 --samp_iter 1 --samp_num_per_class 10
done
CUDA_VISIBLE_DEVICES=2 python distill_transduct_continue_1.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=0.1 --optimizer_con Adam \
--expert_epochs=1900 --lr_student=0.1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=10 --syn_steps=1000 \
--buffer_path './logs/Buffer/used/ogbn-arxiv-20220925-223423-976629' \
--coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.01-20221101-133345-976504' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.01 --ntk_reg 0.1 --eval_interval 10 --ITER 200 --samp_iter 1 --samp_num_per_class 10

CUDA_VISIBLE_DEVICES=2 python distill_transduct_pge_1.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=0.1 --optimizer_con Adam \
--expert_epochs=1900 --lr_student=0.1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=10 --syn_steps=1000 \
--buffer_path './logs/Buffer/ogbn-arxiv-20220925-223423-976629' \
--coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.01-20221101-133345-976504' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.01 --ntk_reg 0.1 --eval_interval 10 --ITER 200 --samp_iter 1 --samp_num_per_class 10
