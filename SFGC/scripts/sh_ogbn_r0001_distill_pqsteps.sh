#ogbn-arxiv-r0001-pq_steps
for exp_epochs in 1800 1600 1400 1200 1000
do
  for syn_epochs in 1400 1200 1000 800 600
  do
    for lr_feat in 0.2
    do
      for lr_stu in 0.2
      do
      CUDA_VISIBLE_DEVICES=2 python distill_transduct_adj_identity2.py --dataset ogbn-arxiv --device cuda:0 --lr_feat=${lr_feat} --optimizer_con Adam \
      --expert_epochs=${exp_epochs} --lr_student=${lr_stu} --optim_lr=1 --optimizer_lr SGD --lr_lr 1e-6 \
      --start_epoch=30 --syn_steps=${syn_epochs} \
      --buffer_path './logs/Buffer/used/ogbn-arxiv-20220925-223423-976629' \
      --coreset_init_path './logs/Coreset/ogbn-arxiv-reduce_0.001-20221025-212029-649373' \
      --condense_model GCN --interval_buffer 1 --rand_start 1 \
      --reduction_rate=0.001 --ntk_reg 0.1 --eval_interval 1 --ITER 300 --samp_iter 1 --samp_num_per_class 50
      done
    done
  done
done
