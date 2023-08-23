for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=1 python distill_transduct.py --dataset citeseer --device cuda:0 --lr_feat=0.0005 --optimizer_con Adam \
  --expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=70 --syn_steps=200 \
  --buffer_path './logs/Buffer/used/citeseer-20221012-091556-763255' \
  --coreset_init_path 'logs/Coreset/citeseer-reduce_0.25-20221106-171050-606991' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k} \
  --reduction_rate=0.25 --ntk_reg 1 --eval_interval 10 --ITER 2500 --samp_iter 5 --samp_num_per_class 10
done
CUDA_VISIBLE_DEVICES=1 python distill_transduct_continue.py --dataset citeseer --device cuda:0 --lr_feat=0.0005 --optimizer_con Adam \
--expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=70 --syn_steps=200 \
--buffer_path './logs/Buffer/used/citeseer-20221012-091556-763255' \
--coreset_init_path 'logs/Coreset/citeseer-reduce_0.25-20221106-171050-606991' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.25 --ntk_reg 1 --eval_interval 10 --ITER 2500 --samp_iter 5 --samp_num_per_class 10

CUDA_VISIBLE_DEVICES=1 python distill_transduct_pge.py --dataset citeseer --device cuda:0 --lr_feat=0.0005 --optimizer_con Adam \
--expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=70 --syn_steps=200 \
--buffer_path './logs/Buffer/used/citeseer-20221012-091556-763255' \
--coreset_init_path 'logs/Coreset/citeseer-reduce_0.25-20221106-171050-606991' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.25 --ntk_reg 1 --eval_interval 10 --ITER 2500 --samp_iter 5 --samp_num_per_class 10

for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=1 python distill_transduct.py --dataset citeseer --device cuda:0 --lr_feat=0.001 --optimizer_con Adam \
  --expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=30 --syn_steps=200 \
  --buffer_path './logs/Buffer/used/citeseer-20221012-091556-763255' \
  --coreset_init_path './logs/Coreset/citeseer-reduce_0.5-20221106-114854-910019' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k} \
  --reduction_rate=0.5 --ntk_reg 1 --eval_interval 10 --ITER 300 --samp_iter 5 --samp_num_per_class 10
done
CUDA_VISIBLE_DEVICES=1 python distill_transduct_continue.py --dataset citeseer --device cuda:0 --lr_feat=0.001 --optimizer_con Adam \
--expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=200 \
--buffer_path './logs/Buffer/used/citeseer-20221012-091556-763255' \
--coreset_init_path './logs/Coreset/citeseer-reduce_0.5-20221106-114854-910019' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.5 --ntk_reg 1 --eval_interval 10 --ITER 300 --samp_iter 5 --samp_num_per_class 10

CUDA_VISIBLE_DEVICES=1 python distill_transduct_pge.py --dataset citeseer --device cuda:0 --lr_feat=0.001 --optimizer_con Adam \
--expert_epochs=500 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=200 \
--buffer_path './logs/Buffer/used/citeseer-20221012-091556-763255' \
--coreset_init_path './logs/Coreset/citeseer-reduce_0.5-20221106-114854-910019' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.5 --ntk_reg 1 --eval_interval 10 --ITER 300 --samp_iter 5 --samp_num_per_class 10

for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=1 python distill_transduct.py --dataset citeseer --device cuda:0 --lr_feat=0.001 --optimizer_con Adam \
  --expert_epochs=400 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=30 --syn_steps=300 \
  --buffer_path './logs/Buffer/used/citeseer-20221012-091556-763255' \
  --coreset_init_path './logs/Coreset/citeseer-reduce_1.0-20221106-171317-449627' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k}\
  --reduction_rate=1 --ntk_reg 1 --eval_interval 10 --ITER 200 --samp_iter 5 --samp_num_per_class 10
done
CUDA_VISIBLE_DEVICES=1 python distill_transduct_continue.py --dataset citeseer --device cuda:0 --lr_feat=0.001 --optimizer_con Adam \
--expert_epochs=400 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=300 \
--buffer_path './logs/Buffer/used/citeseer-20221012-091556-763255' \
--coreset_init_path './logs/Coreset/citeseer-reduce_1.0-20221106-171317-449627' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=1 --ntk_reg 1 --eval_interval 10 --ITER 200 --samp_iter 5 --samp_num_per_class 10

CUDA_VISIBLE_DEVICES=1 python distill_transduct_pge.py --dataset citeseer --device cuda:0 --lr_feat=0.001 --optimizer_con Adam \
--expert_epochs=400 --lr_student=1 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=300 \
--buffer_path './logs/Buffer/used/citeseer-20221012-091556-763255' \
--coreset_init_path './logs/Coreset/citeseer-reduce_1.0-20221106-171317-449627' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=1 --ntk_reg 1 --eval_interval 10 --ITER 200 --samp_iter 5 --samp_num_per_class 10
