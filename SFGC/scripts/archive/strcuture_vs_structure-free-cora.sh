for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=1 python distill_transduct_1.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam \
  --expert_epochs=1500 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=30 --syn_steps=400 \
  --buffer_path './logs/Buffer/used/cora-20220925-225653-091173' \
  --coreset_init_path 'logs/Coreset/cora-reduce_0.25-20221025-155350-254087' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k}\
  --reduction_rate=0.25 --ntk_reg 1 --eval_interval 10 --ITER 500
done
CUDA_VISIBLE_DEVICES=1 python distill_transduct_continue_1.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam \
--expert_epochs=1500 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=400 \
--buffer_path './logs/Buffer/used/cora-20220925-225653-091173' \
--coreset_init_path 'logs/Coreset/cora-reduce_0.25-20221025-155350-254087' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.25 --ntk_reg 1 --eval_interval 10 --ITER 500

CUDA_VISIBLE_DEVICES=1 python distill_transduct_pge_1.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam \
--expert_epochs=1500 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=400 \
--buffer_path './logs/Buffer/used/cora-20220925-225653-091173' \
--coreset_init_path 'logs/Coreset/cora-reduce_0.25-20221025-155350-254087' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.25 --ntk_reg 1 --eval_interval 10 --ITER 500

for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=1 python distill_transduct_1.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam \
  --expert_epochs=1200 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=30 --syn_steps=500 \
  --buffer_path './logs/Buffer/used/cora-20220925-225653-091173' \
  --coreset_init_path './logs/Coreset/cora-reduce_0.5-20221024-112028-667459' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k}\
  --reduction_rate=0.5 --ntk_reg 1 --eval_interval 10 --ITER 1500
done
#./logs/Distill/cora-reduce_0.5-20230126-120622-485493
#./logs/Distill/cora-reduce_0.5-20230126-123610-159899
#./logs/Distill/cora-reduce_0.5-20230126-130541-618634
CUDA_VISIBLE_DEVICES=1 python distill_transduct_continue_1.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam \
--expert_epochs=1200 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=500 \
--buffer_path './logs/Buffer/used/cora-20220925-225653-091173' \
--coreset_init_path './logs/Coreset/cora-reduce_0.5-20221024-112028-667459' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.5 --ntk_reg 1 --eval_interval 10 --ITER 1500

CUDA_VISIBLE_DEVICES=1 python distill_transduct_pge_1.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam \
--expert_epochs=1200 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=500 \
--buffer_path './logs/Buffer/used/cora-20220925-225653-091173' \
--coreset_init_path './logs/Coreset/cora-reduce_0.5-20221024-112028-667459' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.5 --ntk_reg 1 --eval_interval 10 --ITER 1500

for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=1 python distill_transduct_1.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam \
  --expert_epochs=2000 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=30 --syn_steps=500 \
  --buffer_path './logs/Buffer/used/cora-20220925-225653-091173' \
  --coreset_init_path './logs/Coreset/cora-reduce_1.0-20221025-163255-954853' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k}\
  --reduction_rate=1 --ntk_reg 1 --eval_interval 10 --ITER 500
done
CUDA_VISIBLE_DEVICES=1 python distill_transduct_continue_1.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam \
--expert_epochs=2000 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=500 \
--buffer_path './logs/Buffer/used/cora-20220925-225653-091173' \
--coreset_init_path './logs/Coreset/cora-reduce_1.0-20221025-163255-954853' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=1 --ntk_reg 1 --eval_interval 10 --ITER 500

CUDA_VISIBLE_DEVICES=1 python distill_transduct_pge_1.py --dataset cora --device cuda:0 --lr_feat=0.0001 --optimizer_con Adam \
--expert_epochs=2000 --lr_student=0.5 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=500 \
--buffer_path './logs/Buffer/used/cora-20220925-225653-091173' \
--coreset_init_path './logs/Coreset/cora-reduce_1.0-20221025-163255-954853' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=1 --ntk_reg 1 --eval_interval 10 --ITER 500
