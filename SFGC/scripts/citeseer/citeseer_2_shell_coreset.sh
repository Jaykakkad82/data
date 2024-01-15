#! /bin/bash

#@citeseer-r05
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.5 --load_npy ''

#CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 1 --method kcenter --reduction_rate 0.5 \
#--load_npy './logs/Coreset/citeseer-reduce_0.5-20221106-114854-910019' --runs 1

#@citeseer-r025
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.25 --load_npy ''
#./logs/Coreset/citeseer-reduce_0.25-20221106-171050-606991

#@citeseer-r1
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset citeseer --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 1 --load_npy ''
#./logs/Coreset/citeseer-reduce_1.0-20221106-171317-449627