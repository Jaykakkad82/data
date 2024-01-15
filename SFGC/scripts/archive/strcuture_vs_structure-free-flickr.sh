for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=1 python distill_inductive_k.py --dataset flickr --device cuda:0 --lr_feat=0.05 --optimizer_con Adam \
  --expert_epochs=700 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=30 --syn_steps=600 \
  --buffer_path './logs/Buffer/used/flickr-20221108-120636-260182' \
  --coreset_init_path 'logs/Coreset/flickr-reduce_0.001-20221109-095434-604761' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k} \
  --reduction_rate=0.001 --ntk_reg 1 --eval_interval 20 --ITER 3000 --samp_iter 1 --samp_num_per_class 10
done
CUDA_VISIBLE_DEVICES=1 python distill_inductive_continue.py --dataset flickr --device cuda:0 --lr_feat=0.05 --optimizer_con Adam \
--expert_epochs=700 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=600 \
--buffer_path './logs/Buffer/used/flickr-20221108-120636-260182' \
--coreset_init_path 'logs/Coreset/flickr-reduce_0.001-20221109-095434-604761' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg 1 --eval_interval 20 --ITER 3000 --samp_iter 1 --samp_num_per_class 10

CUDA_VISIBLE_DEVICES=1 python distill_inductive_pge.py --dataset flickr --device cuda:0 --lr_feat=0.05 --optimizer_con Adam \
--expert_epochs=700 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=600 \
--buffer_path './logs/Buffer/used/flickr-20221108-120636-260182' \
--coreset_init_path 'logs/Coreset/flickr-reduce_0.001-20221109-095434-604761' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.001 --ntk_reg 1 --eval_interval 20 --ITER 3000 --samp_iter 1 --samp_num_per_class 10

for k in 1 2 5
do
  CUDA_VISIBLE_DEVICES=1 python distill_inductive_k.py --dataset flickr --device cuda:0 --lr_feat=0.02 --optimizer_con Adam \
  --expert_epochs=900 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
  --start_epoch=30 --syn_steps=900 \
  --buffer_path './logs/Buffer/used/flickr-20221108-120636-260182' \
  --coreset_init_path './logs/Coreset/flickr-reduce_0.01-20221109-101537-112764' \
  --condense_model GCN --interval_buffer 1 --rand_start 1 --k=${k} \
  --reduction_rate=0.01 --ntk_reg 0.1 --eval_interval 10 --ITER 500 --samp_iter 5 --samp_num_per_class 10
done
CUDA_VISIBLE_DEVICES=1 python distill_inductive_continue.py --dataset flickr --device cuda:0 --lr_feat=0.02 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=900 \
--buffer_path './logs/Buffer/used/flickr-20221108-120636-260182' \
--coreset_init_path './logs/Coreset/flickr-reduce_0.01-20221109-101537-112764' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.01 --ntk_reg 0.1 --eval_interval 10 --ITER 500 --samp_iter 5 --samp_num_per_class 10

CUDA_VISIBLE_DEVICES=1 python distill_inductive_pge.py --dataset flickr --device cuda:0 --lr_feat=0.02 --optimizer_con Adam \
--expert_epochs=900 --lr_student=0.2 --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
--start_epoch=30 --syn_steps=900 \
--buffer_path './logs/Buffer/used/flickr-20221108-120636-260182' \
--coreset_init_path './logs/Coreset/flickr-reduce_0.01-20221109-101537-112764' \
--condense_model GCN --interval_buffer 1 --rand_start 1 \
--reduction_rate=0.01 --ntk_reg 0.1 --eval_interval 10 --ITER 500 --samp_iter 5 --samp_num_per_class 10
