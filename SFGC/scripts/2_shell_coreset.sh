#@cora-r025
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1200 --lr 0.01 --lr_coreset 0.01 \
--weight_decay 5e-4 --wd_coreset 1e-4  --save 1 --method kcenter --reduction_rate 0.25 \
--load_npy './logs/Coreset/cora-reduce_0.25-20221025-155350-254087' --runs 1

#@cora-r05
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 1000 --lr 0.01 --lr_coreset 0.005 \
--weight_decay 5e-4 --wd_coreset 5e-4 --save 1 --method kcenter --reduction_rate 0.5

#CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 600 --lr_coreset 0.005 \
#--wd_coreset 5e-4 --method kcenter --reduction_rate 0.5 \
#--load_npy 'logs/Coreset/cora-reduce_0.5-20221024-112028-667459'

#@cora-r1
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 600 --lr 0.01 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 1 --load_npy ''

#CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset cora --device cuda:0 --epochs 600 --lr_coreset 0.01 \
#--wd_coreset 5e-4  --save 1 --method kcenter --reduction_rate 1 \
#--load_npy './logs/Coreset/cora-reduce_1.0-20221025-163255-954853' --runs 1

#@obgn-r0001
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset ogbn-arxiv --device cuda:0 --epochs 1000 --lr 0.01 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.001 --load_npy ''

#CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset ogbn-arxiv --device cuda:0 --epochs 1000 --lr_coreset 0.001 \
#--wd_coreset 0  --save 1 --method kcenter --reduction_rate 0.001 \
#--load_npy './logs/Coreset/ogbn-arxiv-reduce_0.001-20221025-212029-649373' --runs 1

#@obgn-r0005
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset ogbn-arxiv --device cuda:0 --epochs 1000 --lr 0.01 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.005 --load_npy ''

#CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset ogbn-arxiv --device cuda:0 --epochs 1000 --lr_coreset 0.001 \
#--wd_coreset 0 --save 1 --method kcenter --reduction_rate 0.005 \
#--load_npy './logs/Coreset/ogbn-arxiv-reduce_0.005-20221101-133048-529972' --runs 1

#@obgn-r001
CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset ogbn-arxiv --device cuda:0 --epochs 1000 --lr 0.01 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.01 --load_npy ''

#CUDA_VISIBLE_DEVICES=0 python train_coreset.py --dataset ogbn-arxiv --device cuda:0 --epochs 2000 --lr_coreset 0.005 \
#--wd_coreset 1e-3 --save 1 --method kcenter --reduction_rate 0.01 \
#--load_npy './logs/Coreset/ogbn-arxiv-reduce_0.01-20221101-133345-976504' --runs 1

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

#@flickr-r0001
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 1000 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.001 --load_npy ''
#./logs/Coreset/flickr-reduce_0.001-20221109-095434-604761

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 1 --method kcenter --reduction_rate 0.001 \
#--load_npy './logs/Coreset/flickr-reduce_0.001-20221109-095434-604761' --runs 1

#@flickr-r0005
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.005 --load_npy ''
#./logs/Coreset/flickr-reduce_0.005-20221109-101214-051861

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 1 --method kcenter --reduction_rate 0.005 \
#--load_npy './logs/Coreset/flickr-reduce_0.005-20221109-101214-051861' --runs 1

#@flickr-r001
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.01 --load_npy ''
#./logs/Coreset/flickr-reduce_0.01-20221109-101537-112764

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset flickr --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 1 --method kcenter --reduction_rate 0.01 \
#--load_npy './logs/Coreset/flickr-reduce_0.01-20221109-101537-112764' --runs 1

#========================

#@reddit-r00005
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 1000 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.0005 --load_npy ''
#./logs/Coreset/reddit-reduce_0.0005-20221113-153007-643703

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 0 --method kcenter --reduction_rate 0.0005 \
#--load_npy './logs/Coreset/reddit-reduce_0.0005-20221113-153007-643703' --runs 1

#@reddit-r0001
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.001 --load_npy ''
#./logs/Coreset/reddit-reduce_0.001-20221113-150848-219047

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 0 --method kcenter --reduction_rate 0.001 \
#--load_npy './logs/Coreset/reddit-reduce_0.001-20221113-150848-219047' --runs 1

#@reddit-r0002
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 1000 --lr 0.001 \
--weight_decay 0.0005  --save 1 --method kcenter --reduction_rate 0.002 --load_npy ''
#./logs/Coreset/reddit-reduce_0.002-20230203-205724-419628


#@reddit-r0005
CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 1000 --lr 0.001 \
--weight_decay 5e-4  --save 1 --method kcenter --reduction_rate 0.005 --load_npy ''
#./logs/Coreset/reddit-reduce_0.005-20221112-121718-604171

#CUDA_VISIBLE_DEVICES=0 python train_coreset_inductive.py --dataset reddit --device cuda:0 --epochs 800 --lr_coreset 0.001 \
#--wd_coreset 5e-3 --save 0 --method kcenter --reduction_rate 0.005 \
#--load_npy './logs/Coreset/reddit-reduce_0.005-20221112-121718-604171' --runs 1







